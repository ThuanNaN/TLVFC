from typing import List, Union, Optional
import torch.nn as nn

from toolkit.base_standardization import Standardization

class FlattenStandardization(Standardization):
    def standardize(self, 
                    module: nn.Module, 
                    group_filter: Optional[List[nn.Module]] = None,
                    *args, **kwargs) -> List[Union[nn.Module, List[nn.Module]]]:
        classes = [
            nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
            nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
            nn.Linear
        ]
        if any([isinstance(module, clazz) for clazz in classes]):
            return [module]

        layers = []
        for child in module.children():
            layers += self.standardize(child)
        if len(layers) == 0:
            layers = [module]
        
        if group_filter is not None:
            return FlattenStandardization.filter(layers, group_filter)
        return layers

    @staticmethod
    def filter(layers: List[Union[nn.Module, List[nn.Module]]],
               group_filter: Optional[List[nn.Module]]):
        lst_index = []
        for index, layer in enumerate(layers):
            if any([isinstance(layer, group) for group in group_filter]):
                lst_index.append(index)
        return [layers[i] for i in lst_index]

