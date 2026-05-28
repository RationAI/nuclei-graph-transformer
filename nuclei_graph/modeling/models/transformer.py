import torch
from timm.layers.drop import DropPath
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask

from nuclei_graph.configuration import Config
from nuclei_graph.modeling.layers import GeGLU, RotarySparseAttention
from nuclei_graph.nuclei_graph_typing import Outputs


class CNN(nn.Module):
    def __init__(self, out_dim: int = 128):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.GroupNorm(8, 16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.features(x))


class MLPSpatialEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, pos: Tensor) -> Tensor:
        return self.proj(pos)


class Layer(nn.Module):
    def __init__(self, config: Config, drop_path_rate: float = 0.0) -> None:
        super().__init__()
        self.self_attn = RotarySparseAttention(
            dim=config.dim, num_heads=config.num_heads
        )
        self.ffn = GeGLU(dim=config.dim, hidden_dim=config.hidden_dim)

        self.pre_attn_norm = nn.RMSNorm(config.dim)
        self.pre_ffn_norm = nn.RMSNorm(config.dim)

        self.drop_path = (
            DropPath(drop_prob=drop_path_rate)
            if drop_path_rate > 0.0
            else nn.Identity()
        )

    def forward(self, x: Tensor, pos: Tensor, block_mask: BlockMask) -> Tensor:
        y = self.pre_attn_norm(x)
        x = x + self.drop_path(self.self_attn(y, pos, block_mask))

        y = self.pre_ffn_norm(x)
        x = x + self.drop_path(self.ffn(y))
        return x


class Transformer(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        dpr = [
            x.item()
            for x in torch.linspace(0, config.drop_path_rate, config.num_layers)
        ]
        self.layers = nn.ModuleList(
            Layer(config, drop_path_rate=dpr[i]) for i in range(config.num_layers)
        )
        self.pos_encoder = MLPSpatialEmbedding(dim=config.dim)

        self.batch_norm = nn.BatchNorm1d(config.norm_dim)
        self.input_proj = nn.Linear(config.node_features, config.dim)
        self.final_norm = nn.RMSNorm(config.dim)

        self.pos_scale = nn.Parameter(torch.tensor(0.0))

        self.class_head = nn.Linear(config.dim, config.num_classes)

        self.attn_head = nn.Sequential(
            nn.Linear(config.dim, config.dim // 2),
            nn.Tanh(),
            nn.Linear(config.dim // 2, 1),
        )
        self.patch_cnn = CNN(out_dim=config.dim)
        self.cnn_norm = nn.RMSNorm(config.dim)
        self.efd_norm = nn.RMSNorm(config.dim)

    def _prepare_features(
        self,
        x: Tensor,
        bboxes: Tensor,
        pos: Tensor,
        real_seq_len: int,
        chunk_size: int = 1024,
        pos_norm_const: float = 1000.0,
    ) -> Tensor:
        norm_dim = self.batch_norm.num_features
        not_to_norm = x[..., norm_dim:]
        norm_full = torch.zeros_like(x[..., :norm_dim])
        norm_full[:real_seq_len] = self.batch_norm(x[:real_seq_len, :norm_dim])

        x_norm = torch.cat([norm_full, not_to_norm], dim=-1)
        efd_emb = self.efd_norm(self.input_proj(x_norm))

        cnn_outputs = []
        for i in range(0, bboxes.size(0), chunk_size):
            cnn_outputs.append(self.patch_cnn(bboxes[i : i + chunk_size]))

        bbox_emb = self.cnn_norm(torch.cat(cnn_outputs, dim=0))
        return bbox_emb  # + efd_emb

    def _pool_graph_logits(
        self, nuclei_logits: Tensor, attn_scores: Tensor, seq_lens: Tensor
    ) -> tuple[Tensor, Tensor]:
        real_seq_len = seq_lens.sum().item()

        nuclei_logits = nuclei_logits[:real_seq_len]
        attn_scores = attn_scores[:real_seq_len]

        seq_lens_list = seq_lens.tolist()
        attn_scores_split = torch.split(attn_scores, seq_lens_list)
        nuclei_logits_split = torch.split(nuclei_logits, seq_lens_list)

        graph_logits_list = []
        attn_weights_list = []

        for scores, logits in zip(attn_scores_split, nuclei_logits_split, strict=True):
            weights = torch.softmax(scores, dim=0)
            graph_logits_list.append(torch.sum(weights * logits, dim=0))
            attn_weights_list.append(weights)

        graph_logits = torch.stack(graph_logits_list)  # (b, num_classes)
        attn_weights = torch.cat(attn_weights_list)  # (real_seq_len, 1)

        return graph_logits, attn_weights

    def forward(
        self,
        x: Tensor,
        bboxes: Tensor,
        pos: Tensor,
        block_mask: BlockMask,
        seq_lens: Tensor,
    ) -> Outputs:
        """Forward pass of the Transformer model handling packed ragged sequences.

        Args:
            x: Target sequence of shape (N_total, d).
            pos: Target positions of shape (N_total, 2).
            bboxes: Image patches of shape (N_total, C, H, W).
            block_mask: Batched BlockMask object for sparse attention.
            seq_lens: Lengths of the individual sequences packed in x, shape (b,).

        Returns:
            Outputs dict containing graph logits, nuclei logits, and attention weights.
        """
        real_seq_len = int(seq_lens.sum().item())
        x = self._prepare_features(x, bboxes, pos, real_seq_len)

        x = x.unsqueeze(0)  # add batch dim: (1, N_total, dim)
        pos = pos.unsqueeze(0)  # (1, N_total, 2)

        for layer in self.layers:
            x = layer(x, pos, block_mask)

        x = self.final_norm(x)
        x = x.squeeze(0)  # remove batch dim: (N_total, dim)

        nuclei_logits = self.class_head(x)  # (N_total, num_classes)
        attn_scores = self.attn_head(x)  # (N_total, 1)

        graph_logits, attn_weights = self._pool_graph_logits(
            nuclei_logits, attn_scores, seq_lens
        )

        return Outputs(
            graph=graph_logits,
            nuclei=nuclei_logits[:real_seq_len],
            attn_weights=attn_weights,
        )
