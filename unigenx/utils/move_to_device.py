# -*- coding: utf-8 -*-
from dataclasses import fields, is_dataclass
from typing import Dict, List, Tuple, Union

import torch


class OutputMixIn:
    """
    MixIn to give namedtuple some access capabilities of a dictionary
    """

    def __getitem__(self, k):
        if isinstance(k, str):
            return getattr(self, k)
        else:
            return super().__getitem__(k)

    def get(self, k, default=None):
        return getattr(self, k, default)

    def items(self):
        return zip(self._fields, self)

    def keys(self):
        return self._fields

    def iget(self, idx: Union[int, slice]):
        """Select item(s) row-wise.

        Args:
            idx ([int, slice]): item to select

        Returns:
            Output of single item.
        """
        return self.__class__(*(x[idx] for x in self))


def move_to_device(
    x: Union[
        Dict[str, Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]],
        torch.Tensor,
        List[torch.Tensor],
        Tuple[torch.Tensor],
    ],
    device: Union[str, torch.DeviceObjType],
    non_blocking: bool = False,
) -> Union[
    Dict[str, Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]],
    torch.Tensor,
    List[torch.Tensor],
    Tuple[torch.Tensor],
]:
    """
    Move object to device.

    Args:
        x (dictionary of list of tensors): object (e.g. dictionary) of tensors to move to device
        device (Union[str, torch.DeviceObjType]): device, e.g. "cpu"

    Returns:
        x on targeted device
    """
    if isinstance(device, str):
        device = torch.device(device)

    if isinstance(x, dict):
        for name in x.keys():
            x[name] = move_to_device(x[name], device=device)
    elif is_dataclass(x):
        for f in fields(x):
            setattr(x, f.name, move_to_device(getattr(x, f.name), device=device))
    elif isinstance(x, OutputMixIn):
        for xi in x:
            move_to_device(xi, device=device)
        return x
    elif isinstance(x, torch.Tensor) and x.device != device:
        x = x.to(device, non_blocking=non_blocking)
    elif isinstance(x, (list, tuple)):
        x = [move_to_device(xi, device=device) for xi in x]
    return x
