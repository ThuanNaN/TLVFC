import torchvision
import torch
from torchvision.models.resnet import ResNet18_Weights, ResNet34_Weights
from torchvision.models.vgg import VGG16_Weights, VGG19_Weights
from collections import OrderedDict
from models.vgg import vgg_get_model
from models.resnet import resnet_get_model
from config.resnet_configs import cfgs
from converter.Converter import Converter
from models.utils import GetPretrain_RESNET, GetPretrain_VGG, apply_new_feature

resnet18_model = resnet_get_model(
    model_name='resnet18',
    base_init='He',
    num_classes=1000
)
resnet18_feature_dict = resnet18_model._get_feature_dict()


vgg16_feature_dict = GetPretrain_VGG(model_group='vgg', \
                                model_name='vgg16').get_feature_dict()


new_feature_dict = Converter(dst_feature_dict = resnet18_feature_dict, 
                      src_feature_dict=vgg16_feature_dict, 
                      mapping_type='relative',
                      candidate_method = 'max'
            )._matching(type_pad='zero',
                        type_pool='avg',
                        num_candidate=3,
                        choice_method={
                            "keep": 'maxVar', 
                            "remove": 'minVar'
                        })


new_state_dict = apply_new_feature(state_dict = resnet18_model.state_dict(), 
                                   new_feature_dict = new_feature_dict)
resnet18_model.load_state_dict(new_state_dict)


