"""Resource accounting.

The project treats efficiency as a first-class result, not an afterthought: everything must
be trainable inside Kaggle's free tier, so every run reports wall time and peak memory
alongside its metrics. `ResourceMonitor` is a context manager so adding it to a script is
one line and cannot be forgotten halfway through.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


def _peak_rss_mb() -> float:
    """Peak resident set size in MB.

    `resource.getrusage` reports bytes on macOS and kilobytes on Linux, which is a classic
    silent 1024x error; psutil sidesteps the platform difference where available.
    """
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / 1e6
    except ImportError:
        import resource
        import sys

        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return raw / 1e6 if sys.platform == "darwin" else raw / 1e3


def _gpu_peak_mb() -> float | None:
    """Peak GPU allocation in MB, or None when training on CPU/MPS.

    MPS (Apple Silicon) is reported separately from CUDA because Kaggle runs CUDA and this
    machine runs MPS -- the README needs to say which one a number came from.
    """
    try:
        import torch
    except ImportError:
        return None

    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e6
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        try:
            return torch.mps.current_allocated_memory() / 1e6
        except (AttributeError, RuntimeError):
            return None
    return None


def device_label() -> str:
    try:
        import torch
    except ImportError:
        return "cpu (no torch)"
    if torch.cuda.is_available():
        return f"cuda ({torch.cuda.get_device_name(0)})"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps (Apple Silicon)"
    return "cpu"


@dataclass
class ResourceMonitor:
    """Context manager recording wall time and peak memory for one run."""

    name: str
    extra: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0
    peak_rss_mb: float = 0.0
    gpu_peak_mb: float | None = None
    _started: float = 0.0

    def __enter__(self) -> "ResourceMonitor":
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            pass
        self._started = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.seconds = time.perf_counter() - self._started
        self.peak_rss_mb = _peak_rss_mb()
        self.gpu_peak_mb = _gpu_peak_mb()
        log.info("%s: %s", self.name, self.summary())

    def summary(self) -> str:
        parts = [f"{self.seconds:.1f}s wall", f"{self.peak_rss_mb:.0f} MB peak RSS"]
        if self.gpu_peak_mb is not None:
            parts.append(f"{self.gpu_peak_mb:.0f} MB GPU")
        parts.append(f"device={device_label()}")
        return ", ".join(parts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "seconds": round(self.seconds, 2),
            "peak_rss_mb": round(self.peak_rss_mb, 1),
            "gpu_peak_mb": None if self.gpu_peak_mb is None else round(self.gpu_peak_mb, 1),
            "device": device_label(),
            **self.extra,
        }
