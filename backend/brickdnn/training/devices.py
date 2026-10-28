"""Which devices PyTorch can train on (CPU, NVIDIA/AMD GPU via CUDA, Apple GPU via MPS)."""

from __future__ import annotations

import functools
import platform
import sys
from typing import Any

import torch

from .reporting import TrainingError

CPU_LABEL = "CPU"
MPS_LABEL = "Apple GPU (MPS)"


def intel_mac() -> bool:
    """On Intel Macs, PyTorch's MPS backend is no faster than the CPU and has been seen
    computing wrong gradients (e.g. with dual Intel/AMD graphics), so it is only used
    when asked for explicitly."""
    return sys.platform == "darwin" and platform.machine() == "x86_64"


def cuda_label() -> str:
    name = torch.cuda.get_device_name(0)
    vendor = "AMD" if getattr(torch.version, "hip", None) else "NVIDIA"
    return f"{vendor} GPU ({name})"


@functools.lru_cache(maxsize=1)
def mps_works() -> bool:
    """MPS can be reported as available and still fail, so run a tiny operation."""
    mps = getattr(torch.backends, "mps", None)
    if mps is None or not mps.is_available():
        return False
    try:
        return float((torch.ones(4, device="mps") * 2).sum().cpu()) == 8.0
    except Exception:
        return False


def device_label(device: torch.device | str) -> str:
    kind = torch.device(device).type
    if kind == "cuda":
        return cuda_label()
    if kind == "mps":
        return MPS_LABEL
    return CPU_LABEL


def available_devices() -> list[dict[str, Any]]:
    """``[{"id": "cpu", "label": "CPU"}, ...]`` in the order of preference."""
    devices = [{"id": "cpu", "label": CPU_LABEL}]
    if torch.cuda.is_available():
        devices.append({"id": "cuda", "label": cuda_label()})
    if mps_works():
        label = "Apple GPU (MPS, experimental on Intel Macs)" if intel_mac() else MPS_LABEL
        devices.append({"id": "mps", "label": label})
    return devices


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if mps_works() and not intel_mac():
        return "mps"
    return "cpu"


def select_device(requested: str) -> torch.device:
    """Resolve ``auto|cpu|cuda|mps``. Raises :class:`TrainingError` if it is unavailable."""
    if requested == "auto":
        return torch.device(default_device())
    if requested == "cuda" and not torch.cuda.is_available():
        raise TrainingError("CUDA is not available on this machine.")
    if requested == "mps" and not mps_works():
        raise TrainingError("The Apple GPU (MPS) is not available on this machine.")
    if requested not in ("cpu", "cuda", "mps"):
        raise TrainingError(f"Unknown device '{requested}'.")
    return torch.device(requested)
