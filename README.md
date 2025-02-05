# Heterogeneous Transfer Learning Using Variance-based Mapping and Pre-trained Feature Crossover

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![WAndB](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28-gray.svg)

This is the official implementation of paper: [TLVFC](https://link.springer.com/chapter/10.1007/978-3-031-63929-6_21)

Knowledge from the pretrained model is transferred to the target model in the form of weight at convolution layers and in the form of
cumulative statistics to initialize weight for linear layers. Finally, a feature crossover strategy is utilized to improve the performance of target model during training.

![TLV-method](./figures/fig_pipeline.png)

## 1. Dependencies

- Python 3.9 or above
- Pytorh 1.12.1+cu116 or higher
- Other packages run:

```bash
pip install -r requirements.txt
```

## 2. Quick Start

```python
from torchvision import models as torchmodel
from models import CustomResnet
from toolkit import TLVFC
from toolkit.standardization import FlattenStandardization
from toolkit.matching import IndexMatching
from toolkit.transfer import VarTransfer


# initialize the TLV transfer tool
transfer_tool = TLVFC(
    standardization=FlattenStandardization(),
    matching=IndexMatching(),
    transfer=VarTransfer()
)  

# define a pre-trained model and load the checkpoint weight from the torchvision hub
pretrained_model = torchmodel.vgg16(weights = torchmodel.VGG16_Weights.IMAGENET1K_V1)

# define the target model, which is enclosed in a new training mechanism.
target_model = CustomResnet._get_model_custom(model_base='resnet18', num_classes=100)

# start to transfer knowledge
transfer_tool(
    from_module=pretrained_model,
    to_module=target_model
)

# training the target model with knowledge is transferred
train(target_model)
```

## 3. Experiemental results

We present the performance evaluation of our proposed method, TLVFC (Heterogeneous Transfer Learning Using Variance-based Mapping and Pre-trained Feature Crossover), along with comparisons to other methods on four different datasets: CIFAR-10, CIFAR-100, Food-101, and PetImages. The benchmarking is conducted using ResNet18 as the target model and VGG16 pretrained with ImageNet1K weights as the source model for the transfer process.

| Method   | CIFAR10 | CIFAR100 | Food-101 | PetImages |
|----------|---------|----------|----------|-----------|
| He initialization | 0.7635 | 0.4178 | 0.675 | 0.9481 |
| DPIAT     | 0.7695 | 0.4155 | 0.6806 | 0.9498 |
| TLVFC-base  | 0.7677 | 0.4234 | 0.7038 | 0.9669 |
| TLVFC-cross | 0.7674 | 0.4293 | **0.7043** | 0.9686 |
| TLVFC      | **0.7724** | **0.4294** | 0.7025 | **0.9696** |

## 4. Effectiveness on very deep networks

The performance of TLVFC method when apply it on very deep networks. An experiment with the Resnet series is therefore conducted.

|                   | resnet-34 | resnet-50 | resnet-101 | resnet-152 |
|-------------------|-----------|-----------|------------|------------|
| He Initialization | 0.7524    | 0.705     | 0.6728     | 0.6435     |
| TLVFC             | 0.7696    | 0.7483    | 0.7195     | 0.6979     |

## 5. Citation

```bash
@InProceedings{
    title="Heterogeneous Transfer Learning Using Variance-based Mapping and Pre-trained Feature Crossover",
    author="Thuan Duong Thang Duong and Phuc Nguyen and Vinh Dinh",
    booktitle="Studies in Systems, Decision, and Control - AICI 2024",
    year="2024",
    publisher="Springer",
}
```
