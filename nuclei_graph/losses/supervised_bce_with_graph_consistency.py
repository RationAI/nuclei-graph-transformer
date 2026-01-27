from typing import Any

import torch
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask

from nuclei_graph.nuclei_graph_typing import WSLMasks


def graph_smoothness_loss_blockwise(
    logits: Tensor,
    wsl_masks: WSLMasks,
    block_mask: BlockMask,
) -> Tensor:
    probs = torch.sigmoid(logits.squeeze(-1))

    sup_mask = wsl_masks["sup_mask"]
    ignore_mask = wsl_masks["ignore_mask"]
    uncertain_mask = (~ignore_mask) & (~sup_mask)

    if not uncertain_mask.any():
        return logits.new_tensor(0.0)

    # squeeze head dimension as we assume 1 head for graph structure
    kv_indices = block_mask.kv_indices.squeeze(1)
    kv_num_blocks = block_mask.kv_num_blocks.squeeze(1)

    block_size = block_mask.BLOCK_SIZE[0]
    batch_size, num_blocks = kv_num_blocks.shape
    block_offsets = torch.arange(block_size, device=logits.device)

    total_loss = logits.new_tensor(0.0)
    valid_batches = 0

    for b_i in range(batch_size):
        probs_b = probs[b_i]  # (Seq,)
        uncertain_b = uncertain_mask[b_i]  # (Seq,)
        ignore_b = ignore_mask[b_i]  # (Seq,)

        kv_ind_b = kv_indices[b_i]  # (num_blocks, max_neighbors)
        kv_num_b = kv_num_blocks[b_i]  # (num_blocks,)

        loss_b = logits.new_tensor(0.0)
        num_uncertain_nodes = 0

        for qb in range(num_blocks):
            # identify uncertain query nodes in the current block (graph)
            q_start = qb * block_size
            q_nodes = block_offsets + q_start
            q_nodes = q_nodes[uncertain_b[q_nodes]]
            if q_nodes.numel() == 0:
                continue

            # get neighbor blocks for the current query block
            num_k = kv_num_b[qb].item()
            if num_k == 0:
                continue

            neighbor_blocks = kv_ind_b[qb, :num_k]
            neighbor_nodes = (
                neighbor_blocks[:, None] * block_size + block_offsets[None, :]
            ).flatten()

            # exclude ignored nodes from the neighbor average
            neighbor_nodes = neighbor_nodes[~ignore_b[neighbor_nodes]]
            if neighbor_nodes.numel() == 0:
                continue
            neighbor_mean = probs_b[neighbor_nodes].mean()

            # MSE: (prediction - average_of_neighbors)^2
            loss_b += (probs_b[q_nodes] - neighbor_mean).pow(2).sum()
            num_uncertain_nodes += q_nodes.numel()

        if num_uncertain_nodes > 0:
            total_loss += loss_b / num_uncertain_nodes
            valid_batches += 1

    return total_loss / max(valid_batches, 1)


class SupervisedBCEWithGraphConsistency(nn.Module):
    def __init__(self, graph_weight: float = 0.3):
        super().__init__()
        self.graph_weight = graph_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        logits: Tensor,
        targets_sup: Tensor,
        wsl_masks: WSLMasks,
        block_mask: BlockMask,
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
        loss_graph = graph_smoothness_loss_blockwise(logits, wsl_masks, block_mask)

        total_loss = loss_sup + self.graph_weight * loss_graph
        logs = {
            "loss_sup": loss_sup.detach().item(),
            "loss_graph": loss_graph.detach().item(),
            "sup_size": sup_size,
        }
        return total_loss, logs
