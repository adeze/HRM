# utils/device.py
import torch
import torch.distributed as dist


def get_device():
    # Prefer MPS on Apple Silicon, then CUDA, else CPU
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_str(device: torch.device = None):
    d = device or get_device()
    return "mps" if d.type == "mps" else ("cuda" if d.type == "cuda" else "cpu")


def choose_dist_backend():
    try:
        # Use NCCL only when CUDA and NCCL available
        if torch.cuda.is_available() and dist.is_nccl_available():
            return "nccl"
    except Exception:
        pass
    return "gloo"