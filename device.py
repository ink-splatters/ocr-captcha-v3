from __future__ import annotations

import typing as t
from enum import StrEnum, auto
from functools import cache

import torch


class Movable(t.Protocol):
    def to(self, device: torch.device) -> t.Self: ...


class DeviceType(StrEnum):
    AUTO = auto()
    CPU = auto()
    CUDA = auto()
    MPS = auto()


@cache
def get_supported_devices() -> list[DeviceType]:
    devices: list[DeviceType] = []
    if torch.cuda.is_available():
        devices.append(DeviceType.CUDA)
    if torch.backends.mps.is_available():
        devices.append(DeviceType.MPS)

    devices.append(DeviceType.CPU)

    return devices


def get_device(*, preference: DeviceType = DeviceType.AUTO) -> torch.device:
    devices = get_supported_devices()

    device_choice: DeviceType = (
        devices[0] if preference == DeviceType.AUTO or preference not in devices else preference
    )

    # TODO: log device_choice
    return torch.device(device_choice)


def to_device[T: Movable](obj: T, *, prefer: DeviceType) -> T:
    device: torch.device = get_device(preference=prefer)
    return obj.to(device)


def to_best_available_device[T: Movable](obj: T) -> T:
    return to_device(obj, prefer=DeviceType.AUTO)
