from typing import List, Type, Union
import torch
from torch import Tensor
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock, Bottleneck
from models.utils import crossover, crossover_simp

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
            crossover(x, x_pretrain, p)
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
    def _get_model_custom(model_base: str,  
                         num_classes: int = 1000):
        if model_base == "resnet18":
            model = CustomResnet(BasicBlock, [2,2,2,2], num_classes=num_classes)
        elif model_base == "resnet34":
            model = CustomResnet(BasicBlock, [3, 4, 6, 3], num_classes = num_classes)
        elif model_base == "resnet50":
            model = CustomResnet(Bottleneck, [3, 4, 6, 3], num_classes = num_classes)
        elif model_base == "resnet101":
            model = CustomResnet(Bottleneck, [3, 4, 23, 3], num_classes = num_classes)
        elif model_base == "resnet152":
            model = CustomResnet(Bottleneck, [3, 8, 36, 3], num_classes = num_classes)
        else:
            raise Exception("The model name must be in [resnet18, resnet34, resnet50, resnet101, resnet152]")
        return model