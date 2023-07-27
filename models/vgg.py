from typing import cast, List, Union
import torch
import torch.nn as nn
from config.vgg_configs import cfgs
from timm.models.layers import trunc_normal_
from models.utils import crossover, crossover_simp

def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False, k_size=3,  pad=1) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=k_size, padding=pad)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class CustomVGG(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 1000,
        init_weights: bool = True, base_init='He',
        dropout: float = 0.5, avgpool: int = 7, last_chanels=512
    ) -> None:
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((avgpool, avgpool))
        self.classifier = nn.Sequential(
            nn.Linear(last_chanels * avgpool * avgpool, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    if base_init == "He":
                        nn.init.kaiming_normal_(
                            m.weight, mode="fan_out", nonlinearity="relu")
                    elif base_init == "Glorot":
                        nn.init.xavier_normal_(m.weight, gain=1)
                    elif base_init == "Trunc":
                        trunc_normal_(m.weight, std=.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, 
                x: torch.Tensor, 
                phase: str = '', 
                x_pretrain: torch.Tensor = None, 
                p: float = 0.1) -> torch.Tensor:
        x = self.features(x) # >> (64, 512, 1, 1) 
        x = self.avgpool(x) # >> (64, 512, 1, 1) 
        x = torch.flatten(x, 1) # >> (64, 512) 
        if phase == "train" and x_pretrain is not None:
            crossover_simp(x, x_pretrain, p)
        x = self.classifier(x)
        return x

    @staticmethod
    def _get_model_custom(model_base: str, num_classes: int, avgpool:int = 7, base_init: str="He"):
        # Origin vgg model
        if model_base == "vgg16":
            model = CustomVGG(
                features=make_layers(cfgs["D"], batch_norm=False,
                                    k_size=3, pad=1),
                num_classes=num_classes,
                init_weights=True,
                base_init=base_init,
                dropout=0.5,
                last_chanels=512,
                avgpool=avgpool
            )

        elif model_base == "vgg19":
            model = CustomVGG(
                features=make_layers(cfgs["E"], batch_norm=False,
                                    k_size=3, pad=1),
                num_classes=num_classes,
                init_weights=True,
                base_init=base_init,
                dropout=0.5,
                last_chanels=512,
                avgpool=avgpool
            )

        #-------------
        elif model_base == "vgg16_5x5_init":
            model = CustomVGG(
                features=make_layers(cfgs["D"], batch_norm=False,
                                    k_size=5, pad="same"),
                num_classes=num_classes,
                init_weights=True,
                base_init=base_init,
                dropout=0.5,
                last_chanels=512,
                avgpool=avgpool
            )

    # ----------------STAGE 2--------------------------------------------
        elif model_base == "vgg16_5x5_Down":
            model = CustomVGG(
                features=make_layers(cfgs["D-Down"], batch_norm=False,
                                    k_size=5, pad="same"),
                num_classes=num_classes,
                init_weights=True,
                base_init=base_init,
                dropout=0.5,
                last_chanels=512,
                avgpool=avgpool
            )

        elif model_base == "vgg16_5x5_Up":
            model = CustomVGG(
                features=make_layers(cfgs["D-Up"], batch_norm=False,
                                    k_size=5, pad="same"),
                num_classes=num_classes,
                init_weights=True,
                base_init=base_init,
                dropout=0.5,
                last_chanels=512,
                avgpool=avgpool
            )

        elif model_base == "vgg16_5x5_DownUp":
            model = CustomVGG(
                features=make_layers(cfgs["D-DownUp"], batch_norm=False,
                                    k_size=5, pad="same"),
                num_classes=num_classes,
                init_weights=True,
                base_init=base_init,
                dropout=0.5,
                last_chanels=512,
                avgpool=avgpool
            )

        elif model_base == "vgg16_5x5_Sort":
            model = CustomVGG(
                features=make_layers(cfgs["D-Sort"], batch_norm=False,
                                    k_size=5, pad="same"),
                num_classes=num_classes,
                init_weights=True,
                base_init=base_init,
                dropout=0.5,
                last_chanels=512,
                avgpool=avgpool
            )

        elif model_base == "vgg16_5x5_Long":
            model = CustomVGG(
                features=make_layers(cfgs["D-Long"], batch_norm=False,
                                    k_size=5, pad="same"),
                num_classes=num_classes,
                init_weights=True,
                base_init=base_init,
                dropout=0.5,
                last_chanels=512,
                avgpool=avgpool
            )

        else:
            raise Exception(
                "model_name must be in ['vgg16_5x5_Down', \
                                        'vgg16_5x5_Up', \
                                        'vgg16_5x5_DownUp', \
                                        'vgg16_5x5_Sort', \
                                        'vgg16_5x5_Long']." )
        return model
