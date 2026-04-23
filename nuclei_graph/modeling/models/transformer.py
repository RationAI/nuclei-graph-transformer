import torch
from timm.layers.drop import DropPath
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask

from nuclei_graph.configuration import Config
from nuclei_graph.modeling.layers import GeGLU, RotarySparseAttention
from nuclei_graph.nuclei_graph_typing import Outputs


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

        self.batch_norm = nn.BatchNorm1d(config.norm_dim)

        self.efd_proj = nn.Linear(config.norm_dim, config.dim)
        geo_dim = config.node_features - config.norm_dim
        self.geo_proj = nn.Sequential(
            nn.Linear(geo_dim, config.dim // 2),
            nn.GELU(),
            nn.Linear(config.dim // 2, config.dim),
        )

        self.final_norm = nn.RMSNorm(config.dim)

        self.class_head = nn.Linear(config.dim, config.num_classes)

        self.attn_head = nn.Sequential(
            nn.Linear(config.dim, config.dim // 2),
            nn.Tanh(),
            nn.Linear(config.dim // 2, 1),
        )

    def forward(
        self, x: Tensor, pos: Tensor, block_mask: BlockMask, seq_lens: Tensor
    ) -> Outputs:
        """Forward pass of the Transformer model handling packed ragged sequences.

        Args:
            x: Target sequence of shape (N_total, d).
            pos: Target positions of shape (N_total, 2).
            block_mask: Batched BlockMask object for sparse attention.
            seq_lens: Lengths of the individual sequences packed in x, shape (b,).

        Returns:
            Outputs dict containing graph logits, nuclei logits, and attention weights.
        """
        norm_dim = self.batch_norm.num_features
        to_norm = x[..., :norm_dim]
        not_to_norm = x[..., norm_dim:]  # scales and angles

        norm = self.batch_norm(to_norm)

        x = self.efd_proj(norm) + self.geo_proj(not_to_norm)
        x = x.unsqueeze(0)  # add batch dim: (1, N_total, dim)
        pos = pos.unsqueeze(0)  # (1, N_total, 2)

        for layer in self.layers:
            x = layer(x, pos, block_mask)

        x = self.final_norm(x)
        x = x.squeeze(0)  # remove batch dim: (N_total, dim)

        nuclei_logits = self.class_head(x)  # (N_total, num_classes)
        attn_scores = self.attn_head(x)  # (N_total, 1)

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
        attn_weights = torch.cat(attn_weights_list)  # (N_total, 1)

        return Outputs(
            graph=graph_logits,
            nuclei=nuclei_logits,
            attn_weights=attn_weights,
        )
