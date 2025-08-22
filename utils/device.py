# utils/device.py
import torch
import torch.distributed as dist


def get_device():
    """Get the best available device: MPS -> CUDA -> CPU"""
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_str(device: torch.device = None):
    """Get device string representation"""
    d = device or get_device()
    return "mps" if d.type == "mps" else ("cuda" if d.type == "cuda" else "cpu")


def choose_dist_backend():
    """Choose the best distributed backend: NCCL for CUDA, GLOO otherwise"""
    try:
        if torch.cuda.is_available() and dist.is_nccl_available():
            return "nccl"
    except Exception:
        pass
    return "gloo"