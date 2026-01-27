from typing import Any

import torch
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask

from nuclei_graph.nuclei_graph_typing import WSLMasks


def graph_smoothness_loss_blockwise(
    logits: Tensor,  # (N, 1)
    masks: WSLMasks,
    block_mask: BlockMask,
) -> Tensor:
    device = logits.device
    p = torch.sigmoid(logits.squeeze(-1))  # (N,)

    sup_mask = masks["sup_mask"]
    ignore_mask = masks["ignore_mask"]
    uncertain_mask = (~ignore_mask) & (~sup_mask)

    if not uncertain_mask.any():
        return logits.new_tensor(0.0)

    BLOCK_SIZE = block_mask.BLOCK_SIZE[0]
    num_blocks = block_mask.kv_num_blocks.shape[-1]

    kv_indices = block_mask.kv_indices[0, 0]  # (num_blocks, max_kv_blocks)
    kv_num_blocks = block_mask.kv_num_blocks[0, 0]  # (num_blocks,)

    loss = logits.new_tensor(0.0)
    count = 0

    for qb in range(num_blocks):
        q_nodes = torch.arange(
            qb * BLOCK_SIZE,
            (qb + 1) * BLOCK_SIZE,
            device=device,
        )

        q_nodes = q_nodes[uncertain_mask[q_nodes]]
        if q_nodes.numel() == 0:
            continue

        kv_blocks = kv_indices[qb, : kv_num_blocks[qb]]

        neigh_nodes = torch.cat(
            [
                torch.arange(
                    kb * BLOCK_SIZE,
                    (kb + 1) * BLOCK_SIZE,
                    device=device,
                )
                for kb in kv_blocks.tolist()
            ]
        )

        neigh_nodes = neigh_nodes[~ignore_mask[neigh_nodes]]
        if neigh_nodes.numel() == 0:
            continue

        neigh_mean = p[neigh_nodes].mean()
        loss += (p[q_nodes] - neigh_mean).pow(2).mean()
        count += 1

    return loss / max(count, 1)


class SupervisedBCEWithGraphConsistency(nn.Module):
    def __init__(self, graph_weight: float = 0.3):
        super().__init__()
        self.graph_weight = graph_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        logits: Tensor,
        targets_sup: Tensor,
        masks: WSLMasks,
        block_mask: BlockMask,
        **kwargs: Any,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Computes combined loss from the supervised BCE and graph smoothness consistency.

        The loss consists of two parts:
          1. supervised loss: BCE computed only on nuclei marked by `masks["sup_mask"]`,
          2. graph consistency loss: Encourages logits of uncertain nuclei to be similar to the average of
                their neighbors, as defined by the sparse neighborhood structure in `block_mask`.

        It is assumed that training batches do not contain padding.

        Args:
            logits: Logits from the model.
            targets_sup: Target labels; only for the supervised set of nuclei.
            masks: Dictionary of masks with keys:
                - "sup_mask" (tensor[bool]): Selects nuclei for supervised loss.
                - "ignore_mask" (tensor[bool]): Selects nuclei to exclude from all losses.
            block_mask: BlockMask object for sparse attention specifying the neighborhood structure.
            kwargs: Additional keyword arguments.

        Returns:
            total_loss: Combined loss tensor.
            logs: Dictionary containing detached loss components and the size of the supervised set.
        """
        logits_sup = logits[masks["sup_mask"]]
        sup_size = targets_sup.numel()

        loss_sup = (
            self.bce(logits_sup, targets_sup) if sup_size > 0 else logits.sum() * 0.0
        )
        loss_graph = graph_smoothness_loss_blockwise(logits, masks, block_mask)

        total_loss = loss_sup + self.graph_weight * loss_graph
        logs = {
            "loss_sup": loss_sup.detach().item(),
            "loss_graph": loss_graph.detach().item(),
            "sup_size": sup_size,
        }
        return total_loss, logs
