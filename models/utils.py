from collections import OrderedDict
import torch
from torchvision.models.resnet import ResNet18_Weights, ResNet34_Weights
from torchvision.models.vgg import VGG16_Weights, VGG19_Weights


class GetPretrain():
    def __init__(self, model_group:str):
        self.model_group = model_group

        if self.model_group == "vgg":
            pass 
        elif self.model_group == "resnet":
            pass
        else:
            raise Exception("The model group must be in ()")

    def get_info(self):
        print(f"Model group: {self.model_group}")
    
    def get_weight(self):
        pass
    def get_feature_dict(self):
        pass

class GetPretrain_VGG(GetPretrain):
    def __init__(self, model_group:str, model_name: str):
        super().__init__(model_group)
        assert self.model_group == "vgg", "model name should be 'vgg'"
        self.model_name = model_name
        if self.model_name == "vgg16":
            self.pretrain_weight = VGG16_Weights.IMAGENET1K_V1.get_state_dict(progress = True)
        elif self.model_name == "vgg19":
            self.pretrain_weight = VGG19_Weights.IMAGENET1K_V1.get_state_dict(progress = True)
        else:
            raise Exception("The model name mus be in (vgg16, vgg19)")

    def get_feature_dict(self) -> OrderedDict:
        feature_dict = OrderedDict()
        for name in self.pretrain_weight.keys():
            split_named = name.split(".")
            if split_named[0] == "features" and split_named[-1] == "weight":
                feature_dict[name] = self.pretrain_weight[name]
        return feature_dict

class GetPretrain_RESNET(GetPretrain):
    def __init__(self, model_group:str, model_name: str):
        super().__init__(model_group)
        assert self.model_group == "resnet", "model name should be 'resnet'"
        self.model_name = model_name
        if self.model_name == "resnet18":
            self.pretrain_weight = ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress = True)
        elif self.model_name == "resnet34":
            self.pretrain_weight = ResNet34_Weights.IMAGENET1K_V1.get_state_dict(progress = True)
        else:
            raise Exception("The model name mus be in (resnet18, resnet34)")

    def get_feature_dict(self) -> OrderedDict:
        feature_dict = OrderedDict()
        for name in self.pretrain_weight.keys():
            split_named = name.split(".")
            if len(split_named) == 2:
                if split_named[0] in ["conv1", "conv2", "conv3"] and split_named[-1] == "weight":
                    feature_dict[name] = self.pretrain_weight[name]
            elif len(split_named) == 4:
                if split_named[2] in ["conv1", "conv2", "conv3"] and split_named[-1] == "weight":
                    feature_dict[name] = self.pretrain_weight[name]
            else: # len == 5
                if split_named[2] == "downsample" and \
                                    split_named[3] == "0" and \
                                    split_named[-1] == "weight":
                    # feature_dict[name] = pretrain_weight[name]
                    pass
        return feature_dict


@torch.no_grad()
def apply_new_feature(state_dict: OrderedDict, 
                      new_feature_dict: OrderedDict
                      ) -> OrderedDict:
    new_state_dict = state_dict.copy()
    lst_state_dict = list(new_state_dict.keys())
    lst_feature_dict = list(new_feature_dict.keys())

    for name in lst_state_dict:
        if name in lst_feature_dict:
            new_state_dict[name] = torch.nn.Parameter(new_feature_dict[name])

    return new_state_dict

