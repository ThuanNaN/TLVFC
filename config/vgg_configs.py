import torch.nn as nn
from typing import Dict, List, Union

cfgs: Dict[str, List[Union[str, int]]] = {
    #VGG16
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],

    #VGG19
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],

    # Config 1: Down depth
    "D-Down": [64, 64, "M", 128, 128, "M", 128, 128, 256, "M", 256, 256, 512, "M", 512, 512, 512, "M"],

    # Config 2: Up depth
    "D-Up": [128, 128, "M", 128, 128, "M", 256, 256, 512, "M", 512, 512, 512, "M", 512, 512, 512, "M"],

    # Config 3: Down-Up depth
    "D-DownUp": [64, 128, "M", 256, 256, "M", 256, 512, 256, "M", 256, 256, 1024, "M", 256, 512, 512, "M"],

    # Config 4: Down-Up depth + Cut-down Length
    "D-Sort": [64, 128, "M", 256, 256, "M", 512, 256, "M", 256, 1024, "M", 256, 512, "M"],

    # Config 4: Down-Up depth + Expanse Length
    "D-Long": [64, 64, 128, "M", 128, 256, 256, "M", 256, 256, 512, 256, "M", 256, 256, 1024, 256, "M", 256, 512, 512, 512, "M"],

}


#version: ImageNet1K_V1
ckpt_url : Dict = {
    "vgg11": {
        "url": "https://download.pytorch.org/models/vgg11-8a719046.pth",
        "file_name": "vgg11-8a719046.pth"
    },
    "vgg13": {
        "url": "https://download.pytorch.org/models/vgg13-19584684.pth",
        "file_name": "vgg13-19584684.pth"
    },

    "vgg16": {
        "url": "https://download.pytorch.org/models/vgg16-397923af.pth",
        "file_name": "vgg16-397923af.pth"
    },

    "vgg19": {
        "url": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
        "file_name": "vgg19-dcbb9e9d.pth"
    },
}


feature_index : Dict = {
    "vgg16": [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28],

}



