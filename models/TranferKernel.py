import copy
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from models.init import compute_std


class TransferKernel():
    def __init__(self, model_weight, ckpt_weight):
        self.model_weight = model_weight
        self.ckpt_weight = ckpt_weight

    def __get_model_weight(self):
        return self.model_weight

    def __set_model_weight(self, new_weight):
        self.model_weight = new_weight

    def __get_ckpt_weight(self):
        return self.ckpt_weight

    def __set_ckpt_weight(self, new_weight):
        self.ckpt_weight = new_weight

    def _choose_kernel(self, **kwargs):

        ckpt_out, ckpt_in, ckpt_k = self.ckpt_weight.size(0), self.ckpt_weight.size(1), self.ckpt_weight.size(2)
        weight_out, weight_in, weight_k = self.model_weight.size(0), self.model_weight.size(1), self.model_weight.size(2)

        print(f"\n❗Transfer: {self.ckpt_weight.size()} -> {self.model_weight.size()}")

        resize_k = False if ckpt_k == weight_k else True
        if ckpt_out == weight_out:
            if ckpt_in == weight_in:
                if not resize_k:
                    print("\t ✅ Copy weight from checkpoint", end='')
                    self.model_weight = copy.deepcopy(self.ckpt_weight)

                else:
                    print("\t ✅ Only resize kernel", end='')
                    self.model_weight = change_kernel_size(weight=self.ckpt_weight,
                                                           target_size=self.model_weight.size(),
                                                           **kwargs)
            elif ckpt_in > weight_in:
                print("\t ✅ Down fan_in", end='')
                self._fan_in_down(self.model_weight.size(), resize_k=resize_k, **kwargs)

            elif ckpt_in < weight_in:
                print("\t ✅ Up fan_in", end='')
                self._fan_in_up(resize_k=resize_k, **kwargs)

        elif ckpt_out > weight_out:
            if ckpt_in == weight_in:
                print("\t ✅ Down fan_out", end='')
                self._fan_out_down(self.model_weight.size(), resize_k=resize_k, **kwargs)
                
            elif ckpt_in > weight_in:
                print("\t ✅ Down all", end='')
                self._down_all(self.model_weight.size(), resize_k=resize_k, **kwargs)
                
            elif ckpt_in < weight_in:
                print("\t ✅ Down fan_out and Up fan_in", end='')
                self._fan_out_down_fan_in_up(resize_k=resize_k, **kwargs)

        elif ckpt_out < weight_out:
            if ckpt_in == weight_in:
                print("\t ✅ Up fan_out", end='')
                self._fan_out_up(resize_k=resize_k, **kwargs)
                
            elif ckpt_in > weight_in:
                print("\t ✅ Up fan_out and Down fan_in", end='')
                self._fan_out_up_fan_in_down(resize_k=resize_k, **kwargs)

            elif ckpt_in < weight_in:
                print("\t ✅ Up all", end='')
                self._up_all(resize_k=resize_k, **kwargs)

        return self.model_weight

    # (64,64,3,3) -> (64,32,5,5)
    def _fan_in_down(self, target_size, resize_k, choice_method, shuffe_k=True, **kwargs):
        temp_weight = []
        for fan_out in self.ckpt_weight:
            fan_in_var = np.var(fan_out.detach().numpy(), axis=(1,2))
            fan_in_ind= choose_candidate(fan_in_var, 
                                           choice_type=choice_method['keep'], 
                                           num_candidate=target_size[1])
            temp_weight.append(fan_out[fan_in_ind])
        new_weight = torch.stack(temp_weight)
        if resize_k:
            new_weight = change_kernel_size(weight=new_weight,
                                            target_size=target_size,
                                            **kwargs)
        self.__set_model_weight(new_weight)
        if shuffe_k:
            self._shuffe_kernel()


    # (64,64,3,3) -> (64,128,5,5)
    def _fan_in_up(self, resize_k, choice_method, shuffe_k=True,  **kwargs):
        if resize_k:
            new_ckpt = change_kernel_size(weight=self.ckpt_weight,
                                          target_size=(self.ckpt_weight.size(0),
                                                       self.ckpt_weight.size(1),
                                                       self.model_weight.size(-1),
                                                       self.model_weight.size(-1)),
                                          **kwargs)
        else:
            new_ckpt = copy.deepcopy(self.ckpt_weight)

        self.model_weight.requires_grad = False
        for f_ind, fan_out in enumerate(self.model_weight):
            fan_in_var = np.var(fan_out.detach().numpy(), axis=(1,2))
            fan_in_ind = choose_candidate(fan_in_var, 
                                           choice_type=choice_method['remove'], 
                                           num_candidate=self.ckpt_weight.size(1))
            for i, k_ind in enumerate(fan_in_ind):
                fan_out[k_ind] = new_ckpt[f_ind][i]
        if shuffe_k:
            self._shuffe_kernel()


    # (128,64,3,3) -> (64,64,5,5)
    def _fan_out_down(self, target_size, resize_k, choice_method, shuffe_k=True, **kwargs):

        fan_out_var = np.var(self.ckpt_weight.detach().numpy(), axis=(1,2,3))
        fan_out_ind = choose_candidate(fan_out_var, 
                                      choice_type=choice_method['keep'], 
                                      num_candidate=target_size[0])
        new_weight = self.ckpt_weight[fan_out_ind]
        if resize_k:
            new_weight = change_kernel_size(weight=new_weight,
                                            target_size=target_size,
                                            **kwargs)
        self.__set_model_weight(new_weight)
        if shuffe_k:
            self._shuffe_kernel()


    # (128,64,3,3) -> (64,32,5,5)
    def _down_all(self, target_size, resize_k, choice_method, shuffe_k=True, **kwargs):
        fan_out_var = np.var(self.ckpt_weight.detach().numpy(), axis=(1,2,3))
        fan_out_ind = choose_candidate(fan_out_var, 
                                       choice_type=choice_method['keep'], 
                                       num_candidate=target_size[0])
        temp_weight = self.ckpt_weight[fan_out_ind]
        temp_fan_out = []
        for fan_out in temp_weight:
            fan_in_var = np.var(fan_out.detach().numpy(), axis=(1,2))
            fan_in_ind = choose_candidate(fan_in_var, 
                                           choice_type=choice_method['keep'], 
                                           num_candidate=target_size[1])
            temp_fan_out.append(fan_out[fan_in_ind])
        new_weight = torch.stack(temp_fan_out)  # -> (64,32,3,3)
        if resize_k:
            new_weight = change_kernel_size(weight=new_weight,
                                            target_size=target_size,
                                            **kwargs)
        self.__set_model_weight(new_weight)
        if shuffe_k:
            self._shuffe_kernel()


    def _up_all(self, resize_k, choice_method, shuffe_f=True, **kwargs):
        # (128,64,5,5)
        if resize_k:
            ckpt_weight_new = change_kernel_size(weight=self.ckpt_weight,
                                                 target_size=(self.ckpt_weight.size(0),
                                                              self.ckpt_weight.size(1),
                                                              self.model_weight.size(-1),
                                                              self.model_weight.size(-1)),
                                                 **kwargs)
        else:
            ckpt_weight_new = copy.deepcopy(self.ckpt_weight)

        self.model_weight.requires_grad = False
        fan_out_var = np.var(self.model_weight.detach().numpy(), axis=(1,2,3))
        fan_out_ind = choose_candidate(fan_out_var,
                                       choice_type=choice_method['remove'],
                                       num_candidate=self.ckpt_weight.size(0))

        for i, f_ind in enumerate(fan_out_ind):
            fan_in_var = np.var(self.model_weight[f_ind].detach().numpy(), axis=(1,2))
            fan_in_ind = choose_candidate(fan_in_var,
                                           choice_type=choice_method['remove'],
                                           num_candidate=self.ckpt_weight.size(1))
            for j, k_ind in enumerate(fan_in_ind):
                self.model_weight[f_ind][k_ind] = ckpt_weight_new[i][j]

        if shuffe_f:
            self._shuffe_kernel()


    # (128,64,3,3) -> (256,64,5,5)
    def _fan_out_up(self, resize_k, choice_method, shuffe_f=True, **kwargs):
        fan_out_var = np.var(self.model_weight.detach().numpy(), axis=(1,2,3))
        fan_out_ind = choose_candidate(fan_out_var,
                                       choice_type=choice_method['remove'],
                                       num_candidate=self.ckpt_weight.size(0))
        if resize_k:
            ckpt_weight_new = change_kernel_size(weight=self.ckpt_weight,
                                                 target_size=(self.ckpt_weight.size(0),
                                                              self.ckpt_weight.size(1),
                                                              self.model_weight.size(-1),
                                                              self.model_weight.size(-1)),
                                                 **kwargs)
        else:
            ckpt_weight_new = copy.deepcopy(self.ckpt_weight)

        self.model_weight.requires_grad = False
        for i, f_ind in enumerate(fan_out_ind):
            self.model_weight[f_ind] = ckpt_weight_new[i]
        if shuffe_f:
            self._shuffe_kernel()

    # (512, 512, 3, 3) -> (256, 1024, 5, 5)
    def _fan_out_down_fan_in_up(self, resize_k, choice_method, shuffe_k=True, **kwargs):
        temp_model_weight = copy.deepcopy(self.__get_model_weight())
        # - stage 1: (512, 512, 3, 3) -> (256, 512, 3, 3)
        first_down_target = (self.model_weight.size(0),
                             self.ckpt_weight.size(1),
                             self.ckpt_weight.size(2),
                             self.ckpt_weight.size(3))

        self._fan_out_down(first_down_target,
                           resize_k=False,
                           choice_method=choice_method,
                           shuffe_k=False,
                           **kwargs)
        self.__set_ckpt_weight(self.__get_model_weight())
        self.__set_model_weight(temp_model_weight)

        # - stage 2: (256, 512, 3, 3) -> (256, 1024, 5, 5)
        self._fan_in_up(resize_k=resize_k,
                        choice_method=choice_method,
                        shuffe_k=True,
                        **kwargs)

    # (128,128,3,3) -> (256,64,5,5)
    def _fan_out_up_fan_in_down(self, resize_k, choice_method, shuffe_f=True, **kwargs):
        target_size = self.model_weight.size()
        temp_weight = nn.init.kaiming_normal_(torch.randn(self.model_weight.size(0),
                                                          self.ckpt_weight.size(1),
                                                          self.model_weight.size(-1),
                                                          self.model_weight.size(-1)),
                                                    mode="fan_out", nonlinearity="relu")
        self.__set_model_weight(temp_weight)
        self._fan_out_up(resize_k=resize_k,
                         choice_method=choice_method,
                         shuffe_f=False,
                         **kwargs)

        self.__set_ckpt_weight(self.__get_model_weight())
        self._fan_in_down(target_size,
                          resize_k=False,
                          choice_method=choice_method,
                          shuffe_k=True,
                          **kwargs)

    def _shuffe_kernel(self):
        fan_out_ind = np.arange(self.model_weight.size(0))
        np.random.shuffle(fan_out_ind)
        self.model_weight = self.model_weight[fan_out_ind]
        for i in range(self.model_weight.size(0)):
            fan_in_ind = np.arange(self.model_weight.size(1))
            np.random.shuffle(fan_in_ind)
            self.model_weight[i] = self.model_weight[i][fan_in_ind]


def pad_pool_kernel(input_kernel: Tensor, std, target_ksize: int, type_pad: str, type_pool: str):
    input_size = input_kernel.size()[-1]
    # Up kernel
    if input_size < target_ksize:
        if type_pad == "zero":
            len_pad = int((target_ksize-input_size)/2)
            kernel_target = F.pad(
                input_kernel, (len_pad, len_pad, len_pad, len_pad), "constant", 0)
        elif type_pad == "init":
            kernel_target = torch.zeros(target_ksize, target_ksize)
            n_pad = target_ksize**2 - input_kernel.size(-1)**2
            padding = torch.empty(n_pad)
            padding.normal_(0, std)
            kernel_target[0, :] = padding[:target_ksize]
            kernel_target[1:-1, -1] = padding[target_ksize:target_ksize + 3]
            kernel_target[-1, :] = padding[target_ksize + 3:target_ksize*2 + 3]
            kernel_target[1:-1, 0] = padding[-3:]
            kernel_target[1:-1, 1:-1] = input_kernel
    # Down kernel
    else:
        kernel_size = (input_size - target_ksize) + 1
        if type_pool == 'avg':
            kernel_target = F.avg_pool2d(input_kernel.unsqueeze(0), 
                                         kernel_size=kernel_size, 
                                         stride=1).squeeze(0)
        elif type_pool == 'max':
            kernel_target = F.max_pool2d(input_kernel.unsqueeze(0), 
                                         kernel_size=kernel_size, 
                                         stride=1).squeeze(0)
    return kernel_target


def change_kernel_size(weight: Tensor, target_size, **kwargs):
    print("\t ✅ Resize kernel")
    new_weight = torch.randn(target_size)
    std = compute_std(target_size, mode="fan_out")
    for i in range(new_weight.size(0)):
        for j in range(new_weight.size(1)):
            new_weight[i][j] = pad_pool_kernel(input_kernel=weight[i][j],
                                          std=std,
                                          target_ksize=target_size[-1],
                                          type_pad=kwargs["type_pad"],
                                          type_pool=kwargs["type_pool"])
    return new_weight


def choose_candidate(candidate, choice_type, num_candidate):
    candidate_sorted = np.argsort(candidate)
    if choice_type == "maxVar":
        candidate_ind = candidate_sorted[-num_candidate:]
    elif choice_type == "minVar":
        candidate_ind = candidate_sorted[:num_candidate]
    elif choice_type == "random":
        candidate_ind = np.random.choice(
            np.arange(len(candidate)), num_candidate, replace=False)
    elif choice_type == "twoTailed":
        oneTailed = int(num_candidate/2)
        candidate_ind = np.concatenate(
            (candidate_sorted[:oneTailed], candidate_sorted[-oneTailed:]), axis=0)
    elif choice_type == "interLeaved":
        step = int(len(candidate_sorted) / num_candidate)
        candidate_ind = [candidate_sorted[i]
                         for i in range(1, len(candidate_sorted), step)]
    else:
        raise Exception(
            "choice_type must in [maxVar, minVar, random, twoTailed, interLeaved]")
    return candidate_ind
