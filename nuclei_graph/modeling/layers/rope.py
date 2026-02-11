"""Adjusted from the Nuclei Foundational Model repository, by Matěj Pekár."""

from typing import Any, cast

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn.utils.parametrizations import orthogonal


class RoPE(nn.Module):
    """Implements the N dimensional RoPE positional encoding.

    Applies RoPE-Mixed on input under learnable orthogonal transformation P.

    Reference:
        - "Learning the RoPEs: Better 2D and 3D Position Encodings with STRING" (https://arxiv.org/abs/2502.02562)
        - "Rethinking RoPE: A Mathematical Blueprint for N-dimensional Rotary Positional Embedding" (https://arxiv.org/abs/2504.06308)
    """

    def __init__(
        self,
        dim: int,
        pos_dim: int = 2,
        theta: float = 1000.0,
        num_linear_dims: int = 2,
    ) -> None:
        """Initialize RoPE module.

        Args:
            dim: The feature dimension of the input tensor. Must be even.
            pos_dim: The dimensionality of the position vectors (e.g., 1 for 1D, 2 for 2D).
            theta: The base value for the RoPE frequency calculation.
            num_linear_dims: Number of leading dimensions to treat as linear/spatial. The remaining (pos_dim - num_linear_dims)
                             are treated as angular (cyclic). Defaults to 2 (for X, Y spatial + optional Theta).
        """
        super().__init__()
        linear_freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        angular_freqs = torch.arange(1, (dim // 2) + 1).float()

        freqs_list = []
        for i in range(pos_dim):
            if i < num_linear_dims:
                freqs_list.append(linear_freqs)
            else:
                freqs_list.append(angular_freqs)
        self.freqs = nn.Parameter(torch.stack(freqs_list))

        self.P = orthogonal(
            nn.Linear(dim, dim, bias=False), orthogonal_map="householder"
        )

        cast("Any", self.P).parametrizations.weight.original._no_weight_decay = True
        cast("Any", self.freqs)._no_weight_decay = True

    @torch.autocast("cuda", enabled=False)  # type: ignore[untyped-decorator]
    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        """Apply RoPE positional encoding.

        Args:
            x ([b, h, n, d]): Input tensor.
            positions ([b, n, pos_dim]): Positions tensor.
        """
        px = self.P(x.float())

        # apply RoPE-Mixed
        freqs = positions @ self.freqs
        freqs_cis = rearrange(
            torch.polar(torch.ones_like(freqs), freqs), "b n c -> b 1 n c"
        )
        px_ = torch.view_as_complex(rearrange(px, "... (d two) -> ... d two", two=2))
        out = rearrange(torch.view_as_real(px_ * freqs_cis), "... d two -> ... (d two)")

        return out.type_as(x)
