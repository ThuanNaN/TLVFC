import os 
import wget
import torch
from torch import Tensor
from config.vgg_configs import ckpt_url

def download_ckpt(model_name: str = "vgg16"):
    os.makedirs("./ckpt", exist_ok = True)
    if model_name not in ["vgg11", "vgg13", "vgg16", "vgg19"]:
        raise Exception("Model name must in {vgg11, vgg13, vgg16, vgg19}")
    
    url = ckpt_url[model_name]["url"]
    file_name = ckpt_url[model_name]["file_name"]
    path_save = os.path.join("./ckpt", file_name)
    
    if not os.path.exists(path_save):
        wget.download(url, out = f"./ckpt/{file_name}")
    ckpt = torch.load(path_save)
    return ckpt

def replace_center(ckpt: Tensor, weight: Tensor):
    weight.requires_grad = False
    ckpt_kernel_sz = ckpt.size(-1)
    pad_size = int(weight.size(-1)/ckpt_kernel_sz)
    for idx_f, filter in enumerate(weight):
        for idx_k, kernel in enumerate(filter):
            kernel[pad_size:pad_size+ckpt_kernel_sz,
                   pad_size:pad_size+ckpt_kernel_sz] = ckpt[idx_f][idx_k]
    return weight


