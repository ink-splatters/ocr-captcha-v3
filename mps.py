from __future__ import annotations

import typing as t
from functools import cache

import torch


class Movable(t.Protocol):
    def to(self, device: torch.device) -> t.Self: ...


@cache
def mps() -> torch.device | None:
    return torch.device("mps") if torch.backends.mps.is_available() else None


def to_mps[T: Movable](obj: T) -> T:
    if device := mps():
        return obj.to(device)

    # TODO: log with proper level
    print("Requested moving to MPS but MPS is not available")
    return obj
