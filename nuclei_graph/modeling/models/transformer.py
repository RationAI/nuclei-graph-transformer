import torch
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask

from nuclei_graph.configuration import Config
from nuclei_graph.modeling.layers import GeGLU, RotarySparseAttention
from nuclei_graph.nuclei_graph_typing import Outputs


class Layer(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.self_attn = RotarySparseAttention(
            dim=config.dim, num_heads=config.num_heads
        )
        self.ffn = GeGLU(dim=config.dim, hidden_dim=config.hidden_dim)

        self.pre_attn_norm = nn.RMSNorm(config.dim)
        self.pre_ffn_norm = nn.RMSNorm(config.dim)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.ffn_dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, pos: Tensor, block_mask: BlockMask) -> Tensor:
        y = self.pre_attn_norm(x)
        x = x + self.attn_dropout(self.self_attn(y, pos, block_mask))

        y = self.pre_ffn_norm(x)
        x = x + self.ffn_dropout(self.ffn(y))
        return x


class Transformer(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self.layers = nn.ModuleList(Layer(config) for _ in range(config.num_layers))
        self.batch_norm = nn.BatchNorm1d(config.norm_dim)
        self.input_proj = nn.Linear(config.node_features, config.dim)
        self.final_norm = nn.RMSNorm(config.dim)

        self.class_head = nn.Linear(config.dim, config.num_classes)
        self.attn_head = nn.Sequential(
            nn.Linear(config.dim, config.dim // 2),
            nn.Tanh(),
            nn.Linear(config.dim // 2, 1),
        )

    def forward(
        self, x: Tensor, pos: Tensor, block_mask: BlockMask, seq_len: Tensor
    ) -> Outputs:
        """Forward pass of the Transformer model.

        Args:
            x: Target sequence of shape (b, n, d).
            pos: Target positions of shape (b, n, 2).
            block_mask: Batched BlockMask object for sparse attention with layouts
                - kv_num_blocks of shape (b, 1, num_blocks), num_blocks = n // block_size
                - kv_indices of shape (b, 1, num_blocks, max_num_blocks)
            seq_len: Length of the sequences before padding, shape (b,).

        Returns:
            Dictionary containing the graph logits, nuclei logits, and attention weights.
        """
        norm_dim = self.batch_norm.num_features
        to_norm = x[..., :norm_dim]
        not_to_norm = x[..., norm_dim:]

        norm = self.batch_norm(to_norm.transpose(1, 2)).transpose(1, 2)
        x = torch.cat([norm, not_to_norm], dim=-1)

        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x, pos, block_mask)

        x = self.final_norm(x)

        nuclei_preds = self.class_head(x)  # (b, n, num_classes)
        attn_scores = self.attn_head(x)  # (b, n, 1)

        # compute mask for valid tokens based on seq_len
        valid_mask = (
            torch.arange(x.shape[1], device=x.device)[None, :] < seq_len[:, None]
        )
        valid_mask = valid_mask.unsqueeze(-1)  # (b, n, 1)

        # mask out padded tokens before softmax
        attn_scores = attn_scores.masked_fill(~valid_mask, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=1)

        graph_pred = torch.sum(attn_weights * nuclei_preds, dim=1)

        return Outputs(
            graph=graph_pred,  # (b, num_classes)
            nuclei=nuclei_preds,  # (b, n, num_classes)
            attn_weights=attn_weights,  # (b, n, 1)
        )
