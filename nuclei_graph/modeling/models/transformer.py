import torch
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask

from nuclei_graph.configuration import Config
from nuclei_graph.modeling.layers import GeGLU, RotarySparseAttention


class Layer(nn.Module):
    def __init__(self, config: Config, layer_idx: int, num_layers: int) -> None:
        super().__init__()
        self.pos_dim = config.pos_dim
        self.self_attn = RotarySparseAttention(
            dim=config.dim, num_heads=config.num_heads, pos_dim=self.pos_dim
        )
        self.ffn = GeGLU(dim=config.dim, hidden_dim=config.hidden_dim)

        self.pre_attn_norm = nn.RMSNorm(config.dim)
        self.pre_ffn_norm = nn.RMSNorm(config.dim)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.ffn_dropout = nn.Dropout(config.dropout)

        self.ls1 = nn.Parameter(torch.full((config.dim,), config.layer_scale_init))
        self.ls2 = nn.Parameter(torch.full((config.dim,), config.layer_scale_init))

    def forward(self, x: Tensor, pos: Tensor, block_mask: BlockMask) -> Tensor:
        assert pos.shape[-1] >= self.pos_dim
        pos = pos[:, :, : self.pos_dim]

        y = self.pre_attn_norm(x)
        x = x + self.ls1 * self.attn_dropout(self.self_attn(y, pos, block_mask))

        y = self.pre_ffn_norm(x)
        x = x + self.ls2 * self.ffn_dropout(self.ffn(y))
        return x


class Transformer(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            Layer(config, i, config.num_layers) for i in range(config.num_layers)
        )
        self.input_proj = nn.Linear(config.node_features, config.dim)
        self.final_norm = nn.RMSNorm(config.dim)
        self.class_head = nn.Linear(config.dim, config.num_classes)
        self.num_heads = config.num_heads
        self.pos_dim = config.pos_dim

    def forward(self, x: Tensor, pos: Tensor, block_mask: BlockMask) -> Tensor:
        """Forward pass of the Transformer model.

        Args:
            x: Target sequence of shape (b, n, d).
            pos: Target positions of shape (b, n, pos_dim).
            block_mask: Batched BlockMask object for sparse attention with layouts
                - kv_num_blocks of shape (b, 1, num_blocks), num_blocks = n // block_size
                - kv_indices of shape (b, 1, num_blocks, max_num_blocks)

        Returns:
            Tensor of shape (b, n, 1).
        """
        x = self.input_proj(x)

        for layer in self.layers:
            x = layer(x, pos, block_mask)

        x = self.final_norm(x)

        return self.class_head(x)
