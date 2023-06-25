from typing import cast, List, Union
import torch
import torch.nn as nn
from config.vgg_configs import cfgs
from models.utils import download_ckpt, replace_center
from timm.models.layers import trunc_normal_

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


class VGG(nn.Module):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def get_model(model_type: str, base_init: str, num_classes: int):

    if model_type == "vgg16_5x5_init":
        model = VGG(
            features=make_layers(cfgs["D"], batch_norm=False,
                                 k_size=5, pad="same"),
            num_classes=num_classes,
            init_weights=True,
            base_init=base_init,
            dropout=0.5,
            last_chanels=512
        )

    elif model_type == "vgg16_5x5_padding0":
        model = VGG(
            features=make_layers(cfgs["D"], batch_norm=False,
                                 k_size=5, pad="same"),
            num_classes=num_classes,
            init_weights=True,
            base_init=base_init,
            dropout=0.5,
            last_chanels=512
        )
        vgg16_ckpt = download_ckpt()
        for idx, module in model.features._modules.items():
            if module.__class__.__name__ == 'Conv2d':
                ckpt_w = vgg16_ckpt[f"features.{idx}.weight"]
                new_w = nn.ConstantPad2d((1, 1, 1, 1), 0)(ckpt_w)
                module.weight = nn.parameter.Parameter(new_w)

    elif model_type == "vgg16_5x5_paddingBaseInit":
        model = VGG(
            features=make_layers(cfgs["D"], batch_norm=False,
                                 k_size=5, pad="same"),
            num_classes=num_classes,
            init_weights=True,
            base_init=base_init,
            dropout=0.5,
            last_chanels=512
        )
        vgg16_ckpt = download_ckpt()
        for idx, module in model.features._modules.items():
            if module.__class__.__name__ == 'Conv2d':
                ckpt_w = vgg16_ckpt[f"features.{idx}.weight"]
                new_w = replace_center(ckpt_w, module.weight)
                module.weight = nn.parameter.Parameter(new_w)


# ----------------STAGE 2--------------------------------------------

    elif model_type == "vgg16_5x5_Down":
        model = VGG(
            features=make_layers(cfgs["D-Down"], batch_norm=False,
                                 k_size=5, pad="same"),
            num_classes=num_classes,
            init_weights=True,
            base_init=base_init,
            dropout=0.5,
            last_chanels=512
        )



    elif model_type == "vgg16_5x5_Up":
        model = VGG(
            features=make_layers(cfgs["D-Up"], batch_norm=False,
                                 k_size=5, pad="same"),
            num_classes=num_classes,
            init_weights=True,
            base_init=base_init,
            dropout=0.5,
            last_chanels=512
        )


    elif model_type == "vgg16_5x5_DownUp":
        model = VGG(
            features=make_layers(cfgs["D-DownUp"], batch_norm=False,
                                 k_size=5, pad="same"),
            num_classes=num_classes,
            init_weights=True,
            base_init=base_init,
            dropout=0.5,
            last_chanels=512
        )



    elif model_type == "vgg16_5x5_Sort":
        model = VGG(
            features=make_layers(cfgs["D-Sort"], batch_norm=False,
                                 k_size=5, pad="same"),
            num_classes=num_classes,
            init_weights=True,
            base_init=base_init,
            dropout=0.5,
            last_chanels=512
        )


    elif model_type == "vgg16_5x5_Long":
        model = VGG(
            features=make_layers(cfgs["D-Long"], batch_norm=False,
                                 k_size=5, pad="same"),
            num_classes=num_classes,
            init_weights=True,
            base_init=base_init,
            dropout=0.5,
            last_chanels=512
        )

    else:
        raise Exception(
            "model_name must be in ['vgg16_5x5_Down', \
                                    'vgg16_5x5_Up', \
                                    'vgg16_5x5_DownUp', \
                                    'vgg16_5x5_Sort', \
                                    'vgg16_5x5_Long']."
                        )

    return model
