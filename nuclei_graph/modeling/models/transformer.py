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
        self.mil_mode = config.get("mil_mode", None)
        assert self.mil_mode is None or self.mil_mode in ["embedding", "logit"]

        dpr = [
            x.item()
            for x in torch.linspace(0, config.drop_path_rate, config.num_layers)
        ]
        self.layers = nn.ModuleList(
            Layer(config, drop_path_rate=dpr[i]) for i in range(config.num_layers)
        )

        self.batch_norm = nn.BatchNorm1d(config.norm_dim)
        self.input_proj = nn.Linear(config.node_features, config.dim)
        self.final_norm = nn.RMSNorm(config.dim)

        self.class_head = nn.Linear(config.dim, config.num_classes)

        self.attn_head = nn.Sequential(
            nn.Linear(config.dim, config.dim // 2),
            nn.Tanh(),
            nn.Linear(config.dim // 2, 1),
        )

    def prepare_features(self, x: Tensor, real_seq_len: int) -> Tensor:
        norm_dim = self.batch_norm.num_features
        not_to_norm = x[..., norm_dim:]  # angles

        norm_full = torch.zeros_like(x[..., :norm_dim])
        norm_full[:real_seq_len] = self.batch_norm(x[:real_seq_len, :norm_dim])

        return self.input_proj(torch.cat([norm_full, not_to_norm], dim=-1))

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
    ) -> Outputs:
        real_seq_len = int(seq_lens.sum().item())
        x = self.prepare_features(x, real_seq_len)

        x = x.unsqueeze(0)
        pos = pos.unsqueeze(0)

        for layer in self.layers:
            x = layer(x, pos, block_mask)

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
