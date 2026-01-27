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
    device = logits.device

    sup_mask = wsl_masks["sup_mask"]
    ignore_mask = wsl_masks["ignore_mask"]
    uncertain_mask = (~ignore_mask) & (~sup_mask)

    if not uncertain_mask.any():
        return logits.new_tensor(0.0)

    kv_indices = block_mask.kv_indices.squeeze(1)
    kv_num_blocks = block_mask.kv_num_blocks.squeeze(1)  # (batch, num_blocks)

    BLOCK_SIZE = block_mask.BLOCK_SIZE[0]
    batch_size, num_blocks = kv_num_blocks.shape

    total_loss = logits.new_tensor(0.0)
    valid_batches = 0

    for batch_i in range(batch_size):
        probs_b = probs[batch_i]  # (seq_length,)
        uncertain_b = uncertain_mask[batch_i]  # (seq_length,)
        ignore_b = ignore_mask[batch_i]  # (seq_length,)

        kv_ind_b = kv_indices[batch_i]  # (num_blocks, max_neighbors)
        kv_num_b = kv_num_blocks[batch_i]  # (num_blocks,)

        loss_b = logits.new_tensor(0.0)
        count_b = 0

        for qb in range(num_blocks):
            q_start = qb * BLOCK_SIZE
            q_end = (qb + 1) * BLOCK_SIZE
            q_nodes = torch.arange(q_start, q_end, device=device)
            q_nodes = q_nodes[uncertain_b[q_nodes]]

            if q_nodes.numel() == 0:
                continue

            num_k = kv_num_b[qb].item()
            if num_k == 0:
                continue

            k_blocks = kv_ind_b[qb, :num_k]

            neigh_nodes_list = [
                torch.arange(kb * BLOCK_SIZE, (kb + 1) * BLOCK_SIZE, device=device)
                for kb in k_blocks
            ]
            neigh_nodes = torch.cat(neigh_nodes_list)
            neigh_nodes = neigh_nodes[~ignore_b[neigh_nodes]]

            if neigh_nodes.numel() == 0:
                continue
            neigh_mean = probs_b[neigh_nodes].mean()

            loss_b += (probs_b[q_nodes] - neigh_mean).pow(2).sum()
            count_b += q_nodes.numel()

        if count_b > 0:
            total_loss += loss_b / count_b
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
