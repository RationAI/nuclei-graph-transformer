"""Source: Nuclei Foundational Model repository."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class GeGLU(nn.Module):
    """Gated GELU."""

    # Tokens per chunk without gradients. The hidden activations of a whole slide
    # (tokens x 2 * hidden_dim, 8 KB per nucleus in fp32) are the largest tensors of the
    # model's evaluation: a slide of 2.3M nuclei would need ~48 GB of GPU memory.
    INFERENCE_CHUNK_SIZE: int = 131072

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gating_proj = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """The FFN of each token of `x`, (..., tokens, dim).

        Without gradients, long sequences go through in chunks of tokens: each token's
        output depends on that token alone. With gradients, chunking would save
        nothing, since the hidden activations are kept for the backward pass.
        """
        n = x.shape[-2]
        if torch.is_grad_enabled() or n <= self.INFERENCE_CHUNK_SIZE:
            return self.ffn(x)

        out: Tensor | None = None
        for start in range(0, n, self.INFERENCE_CHUNK_SIZE):
            end = min(start + self.INFERENCE_CHUNK_SIZE, n)
            y = self.ffn(x[..., start:end, :])
            if out is None:
                out = y.new_empty((*x.shape[:-2], n, y.shape[-1]))
            out[..., start:end, :] = y
        assert out is not None
        return out

    def ffn(self, x: Tensor) -> Tensor:
        gate, up = self.gating_proj(x).chunk(2, dim=-1)
        outputs = F.gelu(gate, approximate="tanh") * up
        return self.down_proj(outputs)
