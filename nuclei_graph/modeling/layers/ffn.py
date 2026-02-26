"""Source: Nuclei Foundational Model repository."""

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GeGLU(nn.Module):
    """Gated GELU."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gating_proj = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate, up = self.gating_proj(x).chunk(2, dim=-1)
        outputs = F.gelu(gate, approximate="tanh") * up
        return self.down_proj(outputs)
