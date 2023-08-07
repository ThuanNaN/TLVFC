# Transfer Learning using Variance-based mapping and Feature crossover
[![python](https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)

This is the official implementation of paper: TLV

Knowledge from the pretrained model is transferred to the target model in the form of weight at convolution layers and in the form of
cumulative statistics to initialize weight for linear layers. Finally, a feature crossover strategy is utilized to improve the performance of target model during training.

![TLV-method](./figures/fig_pipeline.png)

## Installation
```bash
pip install -r requirements.txt
```
## Usage
```python
from torchvision import models as torchmodel
from models import CustomResnet
from toolkit import TLV
from toolkit.standardization import FlattenStandardization
from toolkit.matching import IndexMatching
from toolkit.transfer import VarTransfer

# define the configuration for TLV
var_transfer_config = {
    "type_pad": "zero",
    "type_pool": "avg",
    "choice_method": {
        "keep": "interLeaved",
        "remove": "random"
    }
}

# define modules that will be applied TLV
group_filter = [nn.Conv2d, nn.Conv2d, nn.Conv3d, \
                nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]

# initialize the TLV transfer tool
transfer_tool = TLV(
    standardization=FlattenStandardization(group_filter),
    matching=IndexMatching(),
    transfer=VarTransfer(**var_transfer_config)
)

# define a pre-trained model and load the checkpoint weight from the torchvision hub
pretrained_model:nn.Module = torchmodel.vgg16(weigths = torchmodel.VGG16_Weights.IMAGENET1K_V1)
# compute mean and std to initialize target model linear layers
fc_mean, fc_std = [] ,[]
with torch.no_grad():
    for fc in pretrained_model.classifier:
        if isinstance(fc, nn.Linear):
            fc_mean.append(fc.weight.mean())
            fc_std.append(fc.weight.std())

# define the target model, which is enclosed in a new training mechanism.
target_model = CustomResnet._get_model_custom(model_base='resnet18', num_classes=100)
# initialize the weight of linear layer
num_fc = 1 # normalize for the number of FC layers
nn.init.normal_(
    target_model.fc.weight, 
    torch.Tensor(fc_mean).mean() / num_fc, 
    torch.Tensor(fc_std).mean() / num_fc
)

# start to transfer knowledge
transfer_tool(
    from_module=pretrained_model,
    to_module=target_model
)

# training the target model with knowledge is transferred
train(target_model)
```

## Experiemental results

We present the performance evaluation of our proposed method, TLV (Transfer Learning with Variance-based Regularization), along with comparisons to other methods on four different datasets: CIFAR-10, CIFAR-100, Food-101, and PetImages. The benchmarking is conducted using ResNet18 as the target model and VGG16 pretrained with ImageNet1K weights as the source model for the transfer process.

| Method   | CIFAR10 | CIFAR100 | Food-101 | PetImages |
|----------|---------|----------|----------|-----------|
| He initialization | 0.7635 | 0.4178 | 0.675 | 0.9481 |
| DPIAT     | 0.7695 | 0.4155 | 0.6806 | 0.9498 |
| TLV-base  | 0.7677 | 0.4234 | 0.7038 | 0.9669 |
| TLV-cross | 0.7674 | 0.4293 | **0.7043** | 0.9686 |
| TLV      | **0.7724** | **0.4294** | 0.7025 | **0.9696** |

