"""``postprocessing/nucleus_inference_at_commit.py`` with a lower-memory RoPE rotation.

For slides too large for the GPU (iCAIRD validation, 2.3M nuclei on a 22 GB slice: the RoPE
rotation of q / k / the attention output needs ~3 full [b, h, n, d] fp32 temporaries). The rotation
is the same elementwise complex product, computed one head at a time and written back in place, so
the logits are bit-identical (checked against the unpatched script on a 2.2M-nucleus iCAIRD
validation slide, all three shape seeds: max |Δp| = 0). Same arguments as the wrapped script.
"""

import torch
from einops import rearrange

from nuclei_graph.modeling.layers.rope import RoPE
from postprocessing import nucleus_inference_at_commit


@torch.autocast("cuda", dtype=torch.float32)
def rotate_per_head(self: RoPE, x, positions, inverse: bool = False):
    freqs = positions.to(self.freqs) @ self.freqs
    if inverse:
        freqs = -freqs
    freqs_cis = rearrange(torch.polar(torch.ones_like(freqs), freqs), "b n c -> b 1 n c")
    # in place when possible: both callers rotate fresh temporaries (the output of P, the attention
    # output), so no full-size output buffer is needed
    out = x if x.dtype == torch.float32 else torch.empty(x.shape, dtype=torch.float32, device=x.device)
    for h in range(x.shape[1]):
        x_ = torch.view_as_complex(
            rearrange(x[:, h : h + 1].float(), "... (d two) -> ... d two", two=2).contiguous()
        )
        out[:, h : h + 1] = rearrange(
            torch.view_as_real(x_ * freqs_cis), "... d two -> ... (d two)"
        )
        del x_
    return out.to(x)


RoPE.rotate = rotate_per_head

if __name__ == "__main__":
    nucleus_inference_at_commit.main()
