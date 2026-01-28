from typing import Any

import torch
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask

from nuclei_graph.nuclei_graph_typing import WSLMasks


def graph_smoothness_loss_blockwise(
    logits: Tensor,
    wsl_masks: WSLMasks,
    block_mask: BlockMask,
    pos: Tensor,
    sigma: float = 1.0,
) -> Tensor:
    probs = torch.sigmoid(logits.squeeze(-1))

    sup_mask = wsl_masks["sup_mask"]
    ignore_mask = wsl_masks["ignore_mask"]
    uncertain_mask = (~ignore_mask) & (~sup_mask)

    if not uncertain_mask.any():
        return logits.new_tensor(0.0)

    kv_indices = block_mask.kv_indices.squeeze(1)
    kv_num_blocks = block_mask.kv_num_blocks.squeeze(1)

    block_size = block_mask.BLOCK_SIZE[0]
    batch_size, num_blocks = kv_num_blocks.shape
    block_offsets = torch.arange(block_size, device=logits.device)

    total_loss = logits.new_tensor(0.0)
    valid_batches = 0

    for b_idx in range(batch_size):
        probs_b = probs[b_idx]
        pos_b = pos[b_idx, :, :2]
        uncertain_b = uncertain_mask[b_idx]
        ignore_b = ignore_mask[b_idx]

        kv_blocks_b = kv_indices[b_idx]
        kv_num_b = kv_num_blocks[b_idx]

        loss_b = logits.new_tensor(0.0)
        num_q = 0

        for q_block in range(num_blocks):
            q_nodes = block_offsets + q_block * block_size
            q_mask = uncertain_b[q_nodes]
            if not q_mask.any():
                continue

            q_nodes = q_nodes[q_mask]
            q_probs = probs_b[q_nodes]
            q_pos = pos_b[q_nodes]

            num_k = kv_num_b[q_block].item()
            if num_k == 0:
                continue

            neighbor_blocks = kv_blocks_b[q_block, :num_k]
            neighbor_nodes = (
                neighbor_blocks[:, None] * block_size + block_offsets[None, :]
            ).flatten()

            k_mask = ~ignore_b[neighbor_nodes]
            if not k_mask.any():
                continue

            neighbor_nodes = neighbor_nodes[k_mask]
            k_probs = probs_b[neighbor_nodes]
            k_pos = pos_b[neighbor_nodes]

            dists = torch.cdist(q_pos, k_pos)

            # identify self-loops
            is_self = q_nodes[:, None] == neighbor_nodes[None, :]

            weights = torch.exp(-dists / sigma)
            # zero out the weight for self-loops so a node doesn't "cheat" by predicting itself
            weights.masked_fill_(is_self, 0.0)

            weights_sum = weights.sum(dim=-1, keepdim=True)
            weights = weights / (weights_sum + 1e-8)

            # weighted average of neighbor probabilities
            k_probs_mean = (weights * k_probs).sum(dim=-1)

            loss_b += (q_probs - k_probs_mean).pow(2).sum()
            num_q += q_nodes.numel()

        if num_q > 0:
            total_loss += loss_b / num_q
            valid_batches += 1

    return total_loss / max(valid_batches, 1)


class SupervisedBCEWithGraphConsistency(nn.Module):
    def __init__(self, graph_weight: float = 10.0):
        super().__init__()
        self.graph_weight = graph_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        logits: Tensor,
        targets_sup: Tensor,
        wsl_masks: WSLMasks,
        block_mask: BlockMask,
        pos: Tensor,
        **kwargs: Any,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Computes combined loss from the supervised BCE and graph smoothness consistency.

        The loss consists of two parts:
          1. supervised loss: BCE computed only on nuclei marked by `wsl_masks["sup_mask"]`,
          2. graph consistency loss: Encourages logits of uncertain nuclei to be similar to the average of
                their neighbors, as defined by the sparse neighborhood structure in `block_mask`.

        It is assumed that training batches do not contain padding.

        Args:
            logits: Logits from the model.
            targets_sup: Target labels; only for the supervised set of nuclei.
            wsl_masks: Dictionary of masks with keys:
                - "sup_mask" (tensor[bool]): Selects nuclei for supervised loss.
                - "ignore_mask" (tensor[bool]): Selects nuclei to exclude from all losses.
            block_mask: BlockMask object for sparse attention specifying the neighborhood structure.
            pos: Positions of nuclei; used for distance weighting in graph consistency loss.
            kwargs: Additional keyword arguments.

        Returns:
            total_loss: Combined loss tensor.
            logs: Dictionary containing detached loss components and the size of the supervised set.
        """
        logits_sup = logits[wsl_masks["sup_mask"]]
        sup_size = targets_sup.numel()

        loss_sup = (
            self.bce(logits_sup, targets_sup) if sup_size > 0 else logits.sum() * 0.0
        )
        loss_graph = graph_smoothness_loss_blockwise(logits, wsl_masks, block_mask, pos)

        total_loss = loss_sup + self.graph_weight * loss_graph
        logs = {
            "loss_sup": loss_sup.detach().item(),
            "loss_graph": loss_graph.detach().item(),
            "sup_size": sup_size,
        }
        return total_loss, logs
