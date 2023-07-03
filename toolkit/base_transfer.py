from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
import torch.nn as nn


class Transfer(ABC):
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    @torch.no_grad()
    def transfer(self, matched_tensors: List[Tuple[nn.Module, nn.Module]], *args, **kwargs) -> None:
        for matching in matched_tensors:
            if isinstance(matching, list):
                self.transfer(matching)
            elif matching[0] is not None and matching[1] is not None:
                self.transfer_layer(matching[0].weight, matching[1].weight)
                self.transfer_layer(matching[0].bias, matching[1].bias)

    @abstractmethod
    def transfer_layer(self, tensor_from: nn.Module, tensor_to: nn.Module, *args, **kwargs):
        pass

    def __call__(self, matched_tensors: List[Tuple[nn.Module, nn.Module]], *args, **kwargs) -> None:
        self.transfer(matched_tensors, *args, **kwargs)

