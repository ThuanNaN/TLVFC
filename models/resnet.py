from typing import List, Type, Union
import numpy as np
import torch
from torch import Tensor
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock, Bottleneck


class CustomResnet(ResNet):
    def __init__(self,
                 block: Type[Union[BasicBlock, Bottleneck]],
                 layers: List[int],
                 num_classes: int = 1000
                 ) -> None:
        super().__init__(block, layers, num_classes)
    
    def _forward_impl(self, x: Tensor, phase: str, x_pretrain: Tensor, p:float) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x) # >> 64, 64, 56, 56
        x = self.layer2(x) # >> 64, 128, 28, 28
        x = self.layer3(x) # >> 64, 256, 14, 14
        x = self.layer4(x)  # >> 64, 512, 7, 7
        x = self.avgpool(x) # >> 64, 512, 1, 1
        x = torch.flatten(x, 1)
        # crossover
        if phase == "train" and x_pretrain is not None:
            CustomResnet.crossover_fc_with_pretrain(x, x_pretrain, p)
        x = self.fc(x)
        return x

    def forward(self, 
                x: Tensor, 
                phase: str = "", 
                x_pretrain: Tensor = None, 
                p:float = 0.1,
                ) -> Tensor:
        return self._forward_impl(x, phase, x_pretrain, p)
    
    @staticmethod
    def crossover_fc_with_pretrain(x: Tensor, 
                                x_pretrain: Tensor, 
                                p: float):
        len_feat = min(x.size(1), x_pretrain.size(1))
        ind = np.random.choice(np.arange(len_feat), size=int(len_feat*p), replace=False)
        x[:,ind] = x_pretrain[:,ind]


    @staticmethod
    def _get_model_custom(model_base: str,  
                         num_classes: int = 1000):
        if model_base == "resnet18":
            model = CustomResnet(BasicBlock, [2,2,2,2], num_classes=num_classes)
        elif model_base == "resnet34":
            model = CustomResnet(Bottleneck, [3, 4, 6, 3], num_classes = num_classes)
        else:
            raise Exception("The model name must be in ()")
        return model
