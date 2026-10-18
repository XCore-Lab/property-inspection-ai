"""GPU helpers.

Local models (LocateAnything-3B, Gemma 4 E4B) register a `release()` callable with the arbiter.
`arbiter.use("gemma")` serialises local inference and, with residency "swap" (6-8 GB cards),
first frees every other local model; on a 24 GB L4 use PI_GPU_RESIDENCY=coexist.
`slam_lock` serialises 3D reconstructions (Depth Anything 3) on one GPU.
Gemini calls never touch the GPU and are not serialised.
"""
from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from typing import Callable

log = logging.getLogger(__name__)
slam_lock = threading.Lock()


def gpu_info() -> dict:
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return {"cuda": True, "gpu": torch.cuda.get_device_name(0),
                    "vram_used_mb": round((total - free) / 2**20), "vram_total_mb": round(total / 2**20)}
    except Exception:  # noqa: BLE001
        pass
    return {"cuda": False, "gpu": "", "vram_used_mb": None, "vram_total_mb": None}


class VramMonitor:
    """Timing context kept for the pipeline's stage bookkeeping (peak VRAM is not sampled)."""

    def __init__(self, interval: float = 0.05):
        self.peak_mb = 0.0

    def __enter__(self) -> "VramMonitor":
        return self

    def __exit__(self, *exc) -> None:
        pass


class GpuArbiter:
    def __init__(self, residency: str = "swap"):
        self.residency = residency
        self._release: dict[str, Callable[[], None]] = {}
        self._active: set[str] = set()
        self.lock = threading.RLock()     # serialises all LOCAL model inference

    def register(self, name: str, release: Callable[[], None]) -> None:
        self._release[name] = release

    def acquire(self, name: str) -> None:
        if self.residency == "swap":
            for other in list(self._active):
                if other != name and other in self._release:
                    t = time.perf_counter()
                    self._release[other]()
                    self._active.discard(other)
                    log.info("released %s from GPU in %.2fs", other, time.perf_counter() - t)
        self._active.add(name)

    def mark_released(self, name: str) -> None:
        self._active.discard(name)

    def release_all(self) -> None:
        with self.lock:
            for name in list(self._active):
                if name in self._release:
                    self._release[name]()
                self._active.discard(name)

    @contextmanager
    def use(self, name: str):
        with self.lock:
            self.acquire(name)
            yield


_arbiter: GpuArbiter | None = None


def get_arbiter() -> GpuArbiter:
    global _arbiter
    if _arbiter is None:
        from backend.config.settings import get_settings
        _arbiter = GpuArbiter(get_settings().gpu_residency)
    return _arbiter
