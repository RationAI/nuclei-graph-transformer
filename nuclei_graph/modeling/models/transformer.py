import torch
from timm.layers.drop import DropPath
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask
from torch.utils.checkpoint import checkpoint

from nuclei_graph.configuration import Config
from nuclei_graph.modeling.layers import GeGLU, RotarySparseAttention
from nuclei_graph.nuclei_graph_typing import Outputs


EMBEDDING_MODES = ("efd", "bbox", "both")


class CNN(nn.Module):
    def __init__(self, out_dim: int) -> None:
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
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.features(x))


class MLPSpatialEmbedding(nn.Module):
    """(state_dict compatibility with old checkpoints)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

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
        self.mil_mode = config.get("mil_mode", None)
        assert self.mil_mode is None or self.mil_mode in ["embedding", "logit"]

        self.embedding_mode = config.get("embedding_mode", "efd")
        assert self.embedding_mode in EMBEDDING_MODES, (
            f"Invalid embedding_mode: {self.embedding_mode}"
        )

        dpr = [
            x.item()
            for x in torch.linspace(0, config.drop_path_rate, config.num_layers)
        ]
        self.layers = nn.ModuleList(
            Layer(config, drop_path_rate=dpr[i]) for i in range(config.num_layers)
        )

        # if self.embedding_mode in ("efd", "both"):
        #     self.batch_norm = nn.BatchNorm1d(config.norm_dim)
        #     self.input_proj = nn.Linear(config.node_features, config.dim)

        # if self.embedding_mode in ("bbox", "both"):
        #     self.patch_cnn = CNN(out_dim=config.dim)

        # if self.embedding_mode == "both":
        #     self.efd_norm = nn.RMSNorm(config.dim)
        #     self.cnn_norm = nn.RMSNorm(config.dim)

        ###
        self.batch_norm = nn.BatchNorm1d(config.norm_dim)
        self.input_proj = nn.Linear(config.node_features, config.dim)
        self.patch_cnn = CNN(out_dim=config.dim)
        self.efd_norm = nn.RMSNorm(config.dim)
        self.cnn_norm = nn.RMSNorm(config.dim)
        self.pos_scale = nn.Parameter(torch.tensor(1.0))
        self.pos_encoder = MLPSpatialEmbedding(config.dim)
        ###

        self.final_norm = nn.RMSNorm(config.dim)

        self.class_head = nn.Linear(config.dim, config.num_classes)

        self.attn_head = nn.Sequential(
            nn.Linear(config.dim, config.dim // 2),
            nn.Tanh(),
            nn.Linear(config.dim // 2, 1),
        )

    def embed_efd(self, x: Tensor, real_seq_len: int) -> Tensor:
        norm_dim = self.batch_norm.num_features
        not_to_norm = x[..., norm_dim:]  # angles

        norm_full = torch.zeros_like(x[..., :norm_dim])
        norm_full[:real_seq_len] = self.batch_norm(x[:real_seq_len, :norm_dim])

        return self.input_proj(torch.cat([norm_full, not_to_norm], dim=-1))

    def embed_patches(self, bboxes: Tensor, chunk_size: int = 1024) -> Tensor:
        """Embeds nuclei image patches via the CNN, in chunks to bound peak memory."""
        outputs = []
        for i in range(0, bboxes.size(0), chunk_size):
            chunk = bboxes[i : i + chunk_size].float()
            chunk = (chunk / 127.5) - 1.0
            outputs.append(self.patch_cnn(chunk))
        return torch.cat(outputs, dim=0)

    def prepare_features(
        self, x: Tensor, bboxes: Tensor | None, real_seq_len: int
    ) -> Tensor:
        if self.embedding_mode == "efd":
            return self.embed_efd(x, real_seq_len)

        assert bboxes is not None, (
            f"bboxes are required for embedding_mode={self.embedding_mode!r}"
        )
        if self.embedding_mode == "bbox":
            return self.embed_patches(bboxes)

        return self.efd_norm(self.embed_efd(x, real_seq_len)) + self.cnn_norm(
            self.embed_patches(bboxes)
        )

    def pool(
        self, x: Tensor, attn_scores: Tensor, seq_lens_list: list[int]
    ) -> tuple[Tensor, Tensor]:
        """Pools node features/logits into graph features using attention scores as weights.

        Args:
            x: Node features or logits.
            attn_scores: Attention scores.
            seq_lens_list: List of sequence lengths for each graph in the batch.
        """
        x_split = torch.split(x, seq_lens_list)
        attn_scores_split = torch.split(attn_scores, seq_lens_list)

        pooled_list = []
        attn_weights_list = []

        for score, _x in zip(attn_scores_split, x_split, strict=True):
            weights = torch.nan_to_num(torch.softmax(score, dim=0), nan=0.0)
            pooled_list.append(torch.sum(weights * _x, dim=0))
            attn_weights_list.append(weights)

        return torch.stack(pooled_list), torch.cat(attn_weights_list)

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        block_mask: BlockMask,
        seq_lens: Tensor,
        roi_mask: Tensor | None = None,
        bboxes: Tensor | None = None,
    ) -> Outputs:
        real_seq_len = int(seq_lens.sum().item())
        x = self.prepare_features(x, bboxes, real_seq_len)

        x = x.unsqueeze(0)
        pos = pos.unsqueeze(0)

        for layer in self.layers:
            x = checkpoint(layer, x, pos, block_mask, use_reentrant=False)

        x = self.final_norm(x)
        x = x.squeeze(0)

        nuclei_logits = self.class_head(x)
        attn_scores = self.attn_head(x)

        x = x[:real_seq_len]
        nuclei_logits = nuclei_logits[:real_seq_len]
        attn_scores = attn_scores[:real_seq_len]

        if roi_mask is not None:  # pool only nuclei in ROI
            roi_mask = roi_mask[:real_seq_len]
            attn_scores = attn_scores.masked_fill(
                ~roi_mask.bool().unsqueeze(-1), float("-inf")
            )

        seq_lens_list = seq_lens.tolist()
        if self.mil_mode == "embedding":
            pooled_features, attn_weights = self.pool(x, attn_scores, seq_lens_list)
            graph_logits = self.class_head(pooled_features)
        else:  # logit-level pooling
            graph_logits, attn_weights = self.pool(
                nuclei_logits, attn_scores, seq_lens_list
            )

        return Outputs(
            graph=graph_logits,
            nuclei=nuclei_logits[:real_seq_len],
            attn_weights=attn_weights,
        )
