from __future__ import annotations

import gc
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator


@dataclass
class RuntimeMetrics:
    seconds: Dict[str, float] = field(default_factory=dict)
    counters: Dict[str, int] = field(default_factory=dict)

    def add_seconds(self, name: str, elapsed: float) -> None:
        self.seconds[name] = round(self.seconds.get(name, 0.0) + elapsed, 4)

    def increment(self, name: str, amount: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + amount

    def to_dict(self) -> Dict[str, Any]:
        return {"seconds": dict(self.seconds), "counters": dict(self.counters)}


@contextmanager
def timed(metrics: RuntimeMetrics, name: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        metrics.add_seconds(name, time.perf_counter() - start)


def is_cuda_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "out of memory" in message and ("cuda" in message or "cublas" in message or "gpu" in message)


def clear_cuda_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        return
