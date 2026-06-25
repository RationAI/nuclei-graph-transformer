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

    rot_harmonics: Tensor  # Type hint

    def __init__(self, dim: int, pos_dim: int = 2, theta: float = 100.0) -> None:
        """Initialize RoPE module.

        Args:
            dim: The feature dimension of the input tensor. Must be even.
            pos_dim: The dimensionality of the position vectors (e.g., 1 for 1D, 2 for 2D).
            theta: The base value for the RoPE frequency calculation.
        """
        super().__init__()
        assert dim % 2 == 0

        self.head_dim = dim // 2

        base = 1.0 / (theta ** (torch.arange(0, self.head_dim).float() / self.head_dim))
        self.base_freqs = nn.Parameter(base)

        self.W_pos = orthogonal(nn.Linear(pos_dim, self.head_dim, bias=False))

        harmonics = torch.arange(1, self.head_dim + 1, dtype=torch.float32)
        self.register_buffer("rot_harmonics", harmonics, persistent=False)

        self.P = orthogonal(
            nn.Linear(dim, dim, bias=False),
            orthogonal_map="householder",
        )

    @torch.autocast("cuda", dtype=torch.float32)
    def forward(self, x: Tensor, positions: Tensor, angles: Tensor) -> Tensor:
        """Apply RoPE positional encoding.

        Args:
            x ([b, h, n, d]): Input tensor.
            positions ([b, n, pos_dim]): Positions tensor.
            angles ([b, n, angle_dim]): Angles tensor (expected as cos, sin).
        """
        x = self.P(x.float())

        pos_phase = self.W_pos(positions) * self.base_freqs  # [b, n, d/2]
        nucleus_angle = torch.atan2(angles[..., 1], angles[..., 0]).unsqueeze(
            -1
        )  # [b, n, 1]

        rot_phase = nucleus_angle * self.rot_harmonics  # [b, n, d/2]

        total_phase = pos_phase + rot_phase

        cis = torch.polar(torch.ones_like(total_phase), total_phase)
        cis = rearrange(cis, "b n d -> b 1 n d")

        x_c = torch.view_as_complex(rearrange(x, "... (d two) -> ... d two", two=2))
        out = x_c * cis

        return torch.view_as_real(out).flatten(-2).to(x.dtype)
