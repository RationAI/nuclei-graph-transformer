import platform

import torch


def setup_precision(matmul_precision: str) -> dict[str, str]:
    """Sets the float32 matmul precision and describes the compute environment.

    The XAI arrays come from the repo's standard inference, which uses
    `torch.set_float32_matmul_precision("high")` (TF32 on Ampere+ GPUs); every
    downstream step uses the same setting.
    """
    torch.set_float32_matmul_precision(matmul_precision)
    return {
        "xai.matmul_precision": matmul_precision,
        "xai.tf32": str(matmul_precision != "highest"),
        "xai.gpu": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "none",
        "xai.torch_version": torch.__version__,
        "xai.cuda_version": str(torch.version.cuda),
        "xai.cudnn_version": str(torch.backends.cudnn.version()),
        "xai.python_version": platform.python_version(),
    }
