import math

import torch
import torch.nn.functional as F
from timm.layers.drop import DropPath
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask
from torch.utils.checkpoint import checkpoint

from nuclei_graph.configuration import Config
from nuclei_graph.modeling.layers import GeGLU, RotarySparseAttention
from nuclei_graph.modeling.layers.attention import RelativePositionValueAttention
from nuclei_graph.nuclei_graph_typing import (
    CONTEXT_MODES,
    EMBEDDING_MODES,
    GRAPH_POOLING_MODES,
    Outputs,
)


class CNN(nn.Module):
    def __init__(self, out_dim: int) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1, bias=False),
            nn.GroupNorm(4, 8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(32, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.features(x))


class Layer(nn.Module):
    def __init__(self, config: Config, drop_path_rate: float = 0.0) -> None:
        super().__init__()
        attention_variant = config.get("attention_variant", "standard")
        context = config.get("context", "knn")
        use_rope = config.get("use_rope", True)
        assert context in CONTEXT_MODES, f"Invalid context: {context}"
        assert context != "none" or attention_variant == "standard", (
            f"context='none' requires attention_variant='standard', got '{attention_variant}'"
        )

        if attention_variant == "standard":
            self.self_attn = RotarySparseAttention(
                dim=config.dim,
                num_heads=config.num_heads,
                context=context,
                use_rope=use_rope,
            )
        elif attention_variant == "vrope":
            self.self_attn = RotarySparseAttention(
                dim=config.dim,
                num_heads=config.num_heads,
                rotate_v=True,
                context=context,
                use_rope=use_rope,
            )
        elif attention_variant == "rel_value":
            self.self_attn = RelativePositionValueAttention(
                dim=config.dim, num_heads=config.num_heads
            )
        else:
            raise ValueError(f"Unknown attention_variant: {attention_variant}")

        self.attention_variant = attention_variant
        self.ffn = GeGLU(dim=config.dim, hidden_dim=config.hidden_dim)
        self.pre_attn_norm = nn.RMSNorm(config.dim)
        self.pre_ffn_norm = nn.RMSNorm(config.dim)
        self.drop_path = (
            DropPath(drop_prob=drop_path_rate)
            if drop_path_rate > 0.0
            else nn.Identity()
        )

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        block_mask: BlockMask | None,
        neighbor_idx: Tensor | None,
        neighbor_mask: Tensor | None,
    ) -> Tensor:
        y = self.pre_attn_norm(x)
        if self.attention_variant == "rel_value":
            attn_out = self.self_attn(y, pos, neighbor_idx, neighbor_mask)
        else:
            attn_out = self.self_attn(y, pos, block_mask)
        x = x + self.drop_path(attn_out)

        y = self.pre_ffn_norm(x)
        x = x + self.drop_path(self.ffn(y))
        return x


class Transformer(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
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

        # Projections based on the selected embedding mode
        if self.embedding_mode == "efd":
            self.batch_norm = nn.BatchNorm1d(config.norm_dim)
            self.input_proj = nn.Linear(config.node_features, config.dim)
        elif self.embedding_mode == "blank":
            self.blank_token = nn.Parameter(torch.empty(config.dim))
            nn.init.normal_(self.blank_token, std=0.02)
        elif self.embedding_mode == "bbox":
            self.patch_cnn = CNN(out_dim=config.dim)
            # in place, so the state dict keys (and checkpoints) are unchanged
            self.patch_cnn.compile()
            self.angle_proj = nn.Linear(2, config.dim)  # (cos 2ψ, sin 2ψ)

        self.final_norm = nn.RMSNorm(config.dim)
        self.class_head = nn.Linear(config.dim, config.num_classes)

        self.pooling = config.get("pooling", "attention")
        assert self.pooling in GRAPH_POOLING_MODES, f"Invalid pooling: {self.pooling}"
        if self.pooling == "attention":
            self.attn_head = nn.Sequential(
                nn.Linear(config.dim, config.dim // 2),
                nn.Tanh(),
                nn.Linear(config.dim // 2, 1),
            )
        elif self.pooling == "topk":
            self.topk_frac = config.get("topk_frac", 0.01)
            self.topk_min = config.get("topk_min", 50)
        elif self.pooling == "lse":
            # r = softplus(rho) > 0, initialized at r = 1
            self.lse_rho = nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))

    def embed_efd(self, x: Tensor, real_seq_len: int) -> Tensor:
        norm_dim = self.batch_norm.num_features
        not_to_norm = x[..., norm_dim:]  # angles

        norm_full = torch.zeros_like(x[..., :norm_dim])
        norm_full[:real_seq_len] = self.batch_norm(x[:real_seq_len, :norm_dim])

        return torch.cat([norm_full, not_to_norm], dim=-1)

    def embed_patches(self, bboxes: Tensor, chunk_size: int = 1024) -> Tensor:
        """Embeds nuclei image patches via CNN."""
        outputs = []

        # process in chunks to avoid memory issues with large batches
        for i in range(0, bboxes.size(0), chunk_size):
            chunk = bboxes[i : i + chunk_size].float()
            chunk = (chunk / 127.5) - 1.0
            outputs.append(self.patch_cnn(chunk))
        return torch.cat(outputs, dim=0)

    def prepare_features(
        self,
        x: Tensor,
        pos: Tensor,
        bboxes: Tensor | None,
        real_seq_len: int,
        seq_lens: Tensor,
    ) -> Tensor:
        if self.embedding_mode == "efd":
            assert x is not None, "Features cannot be None in 'efd' mode."
            feats = self.embed_efd(x, real_seq_len)
            return self.input_proj(feats)
        elif self.embedding_mode == "blank":
            total_len = pos.shape[0]
            feats = self.blank_token.new_zeros(total_len, self.blank_token.shape[0])
            feats[:real_seq_len] = self.blank_token
            return feats

        assert bboxes is not None, "Bounding boxes cannot be None in 'bbox' mode."
        assert x is not None, "Orientations cannot be None in 'bbox' mode."
        # patches are orientation-normalized, so the orientation is added back;
        # padding stays zero (as in 'blank' mode) instead of going through the CNN
        real = self.embed_patches(bboxes[:real_seq_len]) + self.angle_proj(
            x[:real_seq_len]
        )
        feats = real.new_zeros(pos.shape[0], real.shape[-1])
        feats[:real_seq_len] = real
        return feats

    def pool(
        self, x: Tensor, attn_scores: Tensor, seq_lens_list: list[int]
    ) -> tuple[Tensor, Tensor]:
        """Pools node features/logits into graph features using attention scores as weights."""
        x_split = torch.split(x, seq_lens_list)
        attn_scores_split = torch.split(attn_scores, seq_lens_list)

        pooled_list = []
        attn_weights_list = []

        for score, _x in zip(attn_scores_split, x_split, strict=True):
            weights = torch.nan_to_num(torch.softmax(score, dim=0), nan=0.0)
            pooled_list.append(torch.sum(weights * _x, dim=0))
            attn_weights_list.append(weights)

        return torch.stack(pooled_list), torch.cat(attn_weights_list)

    def pool_topk(
        self, logits: Tensor, seq_lens_list: list[int]
    ) -> tuple[Tensor, Tensor]:
        """Pools nuclei logits into the graph logit as the mean of the top-k logits.

        k = max(`topk_min`, ceil(`topk_frac` * n)), capped at n. The returned weights
        are 1/k for the selected nuclei and 0 elsewhere.
        """
        pooled_list = []
        weights_list = []

        for _x in torch.split(logits, seq_lens_list):
            n = _x.shape[0]
            k = min(n, max(self.topk_min, math.ceil(self.topk_frac * n)))
            top_idx = torch.topk(_x, k, dim=0).indices
            weights = torch.zeros_like(_x).scatter_(0, top_idx, 1.0 / k)
            pooled_list.append(torch.sum(weights * _x, dim=0))
            weights_list.append(weights)

        return torch.stack(pooled_list), torch.cat(weights_list)

    @property
    def lse_r(self) -> Tensor:
        return F.softplus(self.lse_rho)

    def pool_lse(
        self, logits: Tensor, seq_lens_list: list[int]
    ) -> tuple[Tensor, Tensor]:
        """Pools nuclei logits into the graph logit by LogSumExp with a learnable r.

        z = 1/r * log(1/n * sum_i exp(r * z_i)), which goes from the mean (r -> 0) to the
        max (r -> inf). The returned weights softmax(r * z_i) are dz/dz_i.
        """
        r = self.lse_r
        pooled_list = []
        weights_list = []

        for _x in torch.split(logits, seq_lens_list):
            n = _x.shape[0]
            pooled_list.append((torch.logsumexp(r * _x, dim=0) - math.log(n)) / r)
            weights_list.append(torch.softmax(r * _x, dim=0))

        return torch.stack(pooled_list), torch.cat(weights_list)

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        block_mask: BlockMask | None,
        seq_lens: Tensor,
        neighbor_idx: Tensor | None = None,
        bboxes: Tensor | None = None,
    ) -> Outputs:
        real_seq_len = int(seq_lens.sum().item())
        x = self.prepare_features(x, pos, bboxes, real_seq_len, seq_lens)

        x = x.unsqueeze(0)
        pos = pos.unsqueeze(0)

        neighbor_mask = neighbor_idx >= 0 if neighbor_idx is not None else None

        for layer in self.layers:
            x = checkpoint(
                layer,
                x,
                pos,
                block_mask,
                neighbor_idx,
                neighbor_mask,
                use_reentrant=False,
            )
        x = self.final_norm(x)
        x = x.squeeze(0)

        nuclei_logits = self.class_head(x)[:real_seq_len]
        x = x[:real_seq_len]

        seq_lens_list = seq_lens.tolist()
        if self.pooling == "attention":
            attn_scores = self.attn_head(x)
            graph_logits, attn_weights = self.pool(
                nuclei_logits, attn_scores, seq_lens_list
            )
        elif self.pooling == "topk":
            graph_logits, attn_weights = self.pool_topk(nuclei_logits, seq_lens_list)
        else:
            graph_logits, attn_weights = self.pool_lse(nuclei_logits, seq_lens_list)

        return Outputs(
            graph=graph_logits,
            nuclei=nuclei_logits[:real_seq_len],
            attn_weights=attn_weights,
        )
