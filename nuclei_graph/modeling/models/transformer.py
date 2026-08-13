import torch
from timm.layers.drop import DropPath
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask
from torch.utils.checkpoint import checkpoint

from nuclei_graph.configuration import Config
from nuclei_graph.modeling.layers import GeGLU, RotarySparseAttention
from nuclei_graph.nuclei_graph_typing import EMBEDDING_MODES, Outputs


class PointNetLocalAggregation(nn.Module):
    """Implicit Geometric Tokenizer: Local Set Abstraction layer over k-NN spatial coordinates."""

    def __init__(
        self,
        k: int = 5,
        in_channels: int = 2,
        hidden_dim: int = 16,
        out_dim: int = 32,
    ) -> None:
        super().__init__()
        self.k = k
        self.out_dim = out_dim

        # 1x1 Convolutions act as a shared MLP applied to each neighbor
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_dim, kernel_size=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, pos: Tensor) -> Tensor:
        B, N, _ = pos.shape
        if N <= 1:
            return torch.zeros(B, N, self.out_dim, device=pos.device, dtype=pos.dtype)

        k_eff = min(self.k, N - 1)

        chunk_size = 2048
        knn_indices_list = []

        for i in range(0, N, chunk_size):
            pos_chunk = pos[:, i : i + chunk_size, :]  # [B, C, 2]

            # Compute distances only for this chunk against all points
            dists_chunk = torch.cdist(pos_chunk, pos)  # [B, C, N]

            # Get top k+1 (index 0 is the self-loop)
            _, knn_idx = torch.topk(dists_chunk, k_eff + 1, dim=-1, largest=False)

            # Drop the self-loop and store
            knn_indices_list.append(knn_idx[:, :, 1:])

        # concatenate chunks back together
        knn_indices = torch.cat(knn_indices_list, dim=1)  # [B, N, k_eff]

        # relative neighbor positions
        batch_idx = torch.arange(B, device=pos.device).view(B, 1, 1).expand(B, N, k_eff)
        knn_pos = pos[batch_idx, knn_indices]  # [B, N, k_eff, 2]
        rel_pos = knn_pos - pos.unsqueeze(2)  # [B, N, k_eff, 2]

        local_dists = torch.norm(rel_pos, dim=-1, keepdim=True)
        max_dists = local_dists.max(dim=2, keepdim=True)[0]
        rel_pos_norm = rel_pos / (max_dists + 1e-6)

        x = rel_pos_norm.permute(0, 3, 1, 2)  # [B, 2, N, k_eff]
        x = self.mlp(x)  # [B, out_dim, N, k_eff]
        x = torch.max(x, dim=3)[0]  # [B, out_dim, N]

        return x.permute(0, 2, 1)  # [B, N, out_dim]


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

        # PointNet Local Aggregation module
        self.pointnet_k = config.get("k_neighbors", 5)
        self.pointnet_dim = config.get("pointnet_dim", 32)
        if self.embedding_mode in ["pointnet", "efd_pointnet"]:
            self.pointnet = PointNetLocalAggregation(
                k=self.pointnet_k, out_dim=self.pointnet_dim
            )

        # Projections based on the selected embedding mode
        if self.embedding_mode in ["efd", "spatial", "efd_spatial"]:
            self.batch_norm = nn.BatchNorm1d(config.norm_dim)
            self.input_proj = nn.Linear(config.node_features, config.dim)
        elif self.embedding_mode == "pointnet":
            self.input_proj = nn.Linear(self.pointnet_dim, config.dim)
        elif self.embedding_mode == "efd_pointnet":
            self.batch_norm = nn.BatchNorm1d(config.norm_dim)
            self.input_proj = nn.Linear(
                config.node_features + self.pointnet_dim, config.dim
            )
        elif self.embedding_mode == "bbox":
            self.patch_cnn = CNN(out_dim=config.dim)

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

        return torch.cat([norm_full, not_to_norm], dim=-1)

    def embed_pointnet(self, pos: Tensor, seq_lens: Tensor) -> Tensor:
        """Processes each graph/crop independently to prevent cross-graph k-NN bleeding."""
        seq_lens_list = seq_lens.tolist()
        pos_splits = torch.split(pos[: sum(seq_lens_list)], seq_lens_list)

        pointnet_outs = []
        for pos_g in pos_splits:
            out_g = self.pointnet(pos_g.unsqueeze(0)).squeeze(0)  # [L, pointnet_dim]
            pointnet_outs.append(out_g)

        out = torch.cat(pointnet_outs, dim=0)  # [real_seq_len, pointnet_dim]

        # Zero-pad if pos was padded beyond real_seq_len
        if out.shape[0] < pos.shape[0]:
            pad = torch.zeros(
                pos.shape[0] - out.shape[0],
                self.pointnet_dim,
                device=pos.device,
                dtype=pos.dtype,
            )
            out = torch.cat([out, pad], dim=0)

        return out

    def embed_patches(self, bboxes: Tensor, chunk_size: int = 1024) -> Tensor:
        """Embeds nuclei image patches via CNN."""
        outputs = []

        # process in chunks to avoid memory issues with large batches
        for i in range(0, bboxes.size(0), chunk_size):
            chunk = bboxes[i : i + chunk_size].float()
            chunk = (chunk / 127.5) - 1.0
            outputs.append(self.patch_cnn(chunk))
        return torch.cat(outputs, dim=0)

    def embed_spatial(self, x: Tensor, real_seq_len: int) -> Tensor:
        """Embeds spatial statistics. All features are normalized."""
        norm_full = torch.zeros_like(x)
        norm_full[:real_seq_len] = self.batch_norm(x[:real_seq_len])
        return self.input_proj(norm_full)

    def prepare_features(
        self,
        x: Tensor,
        pos: Tensor,
        bboxes: Tensor | None,
        real_seq_len: int,
        seq_lens: Tensor,
    ) -> Tensor:
        if self.embedding_mode == "efd":
            assert x is not None, "EFD features cannot be None in 'efd' mode."
            efd_feats = self.embed_efd(x, real_seq_len)
            return self.input_proj(efd_feats)
        elif self.embedding_mode == "pointnet":
            pn_feats = self.embed_pointnet(pos, seq_lens)
            return self.input_proj(pn_feats)
        elif self.embedding_mode == "efd_pointnet":
            assert x is not None, "EFD features cannot be None in 'efd_pointnet' mode."
            efd_feats = self.embed_efd(x, real_seq_len)
            pn_feats = self.embed_pointnet(pos, seq_lens)
            combined = torch.cat([efd_feats, pn_feats], dim=-1)
            return self.input_proj(combined)
        elif self.embedding_mode == "spatial":
            assert x is not None, "Spatial features cannot be None in 'spatial' mode."
            return self.embed_spatial(x, real_seq_len)

        assert bboxes is not None, "Bounding boxes cannot be None in 'bbox' mode."
        return self.embed_patches(bboxes)

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

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        block_mask: BlockMask,
        seq_lens: Tensor,
        bboxes: Tensor | None = None,
    ) -> Outputs:
        real_seq_len = int(seq_lens.sum().item())

        x = self.prepare_features(x, pos, bboxes, real_seq_len, seq_lens)
        x = x.unsqueeze(0)
        pos = pos.unsqueeze(0)

        for layer in self.layers:
            x = checkpoint(layer, x, pos, block_mask, use_reentrant=False)
        x = self.final_norm(x)
        x = x.squeeze(0)

        nuclei_logits = self.class_head(x)
        attn_scores = self.attn_head(x)

        nuclei_logits = nuclei_logits[:real_seq_len]
        attn_scores = attn_scores[:real_seq_len]
        x = x[:real_seq_len]

        seq_lens_list = seq_lens.tolist()
        graph_logits, attn_weights = self.pool(
            nuclei_logits, attn_scores, seq_lens_list
        )

        return Outputs(
            graph=graph_logits,
            nuclei=nuclei_logits[:real_seq_len],
            attn_weights=attn_weights,
        )
