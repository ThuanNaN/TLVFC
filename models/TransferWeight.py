import copy
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from models.init import compute_std


class TransferWeight():
    def __init__(self, dst_weight, src_weight, type_pad, type_pool, choice_method):
        self.dst_weight = dst_weight
        self.src_weight = src_weight

        self.dst_size = self.dst_weight.size()
        # self.src_size = self.src_weight.size()

        self.type_pad = type_pad
        self.type_pool = type_pool
        self.choice_method = choice_method


    def __get_src_weight(self):
        return self.src_weight

    def __set_src_weight(self, new_weight):
        self.src_weight = new_weight


    def __get_dst_weight(self):
        return self.dst_weight

    def __set_dst_weight(self, new_weight):
        self.dst_weight = new_weight

    def __get_dst_size(self):
        return self.dst_weight.size()
    

    def _transfer(self):

        src_fout, src_fin, src_k, _ = self.src_weight.size()
        dst_fout, dst_fin, dst_k, _ = self.dst_weight.size()

        print(f"\n❗Transfer: {self.src_weight.size()} -> {self.dst_weight.size()}")

        resize_k = False if src_k == dst_k else True

        if resize_k:
            temp_src_weight = self._change_kernel_size(input_tensor=self.src_weight,
                                                        kernel_size=dst_k)
            self.__set_src_weight(temp_src_weight)

        if src_fout > dst_fout:
            if src_fin > dst_fin:
                self._down_all()
            elif src_fin < dst_fin:
                self._fout_down_fin_up()
            else:
                self._fout_down()

        elif src_fout < dst_fout:
            if src_fin > dst_fin:
                self._fout_up_fin_down()

            elif src_fin < dst_fin:
                self._up_all()
            else:
                self._fout_up()
        else:
            if src_fin > dst_fin:
                self._fin_down()
            elif src_fin < dst_fin:
                self._fin_up()
            else:
                self.__set_dst_weight(copy.deepcopy(self.__get_src_weight()))

        self._shuffe_kernel()
        return self.dst_weight


    @torch.no_grad()
    def _fin_down(self):
        temp_weight = []
        for fan_out in self.src_weight:
            fan_in_var = np.var(fan_out.detach().numpy(), axis=(1,2))
            fan_in_ind= TransferWeight.choose_candidate(fan_in_var, 
                                           choice_type=self.choice_method['keep'], 
                                           num_candidate=self.dst_weight.size(1))
            temp_weight.append(fan_out[fan_in_ind])
        new_weight = torch.stack(temp_weight)

        self.__set_dst_weight(new_weight)


    @torch.no_grad()
    def _fin_up(self):
        for f_ind, fan_out in enumerate(self.dst_weight):
            fan_in_var = np.var(fan_out.detach().numpy(), axis=(1,2))
            fan_in_ind = TransferWeight.choose_candidate(fan_in_var, 
                                           choice_type=self.choice_method['remove'], 
                                           num_candidate=self.src_weight.size(1))
            for i, k_ind in enumerate(fan_in_ind):
                fan_out[k_ind] = self.__get_src_weight()[f_ind][i]


    @torch.no_grad()
    def _fout_down(self):
        fan_out_var = np.var(self.src_weight.detach().numpy(), axis=(1,2,3))
        fan_out_ind = TransferWeight.choose_candidate(fan_out_var, 
                                      choice_type=self.choice_method['keep'], 
                                      num_candidate=self.dst_weight.size(0))
        new_weight = self.src_weight[fan_out_ind]

        self.__set_dst_weight(new_weight)


    @torch.no_grad()
    def _down_all(self):
        fan_out_var = np.var(self.src_weight.detach().numpy(), axis=(1,2,3))
        fan_out_ind = TransferWeight.choose_candidate(fan_out_var, 
                                       choice_type=self.choice_method['keep'], 
                                       num_candidate=self.dst_weight.size(0)) 
        temp_weight = self.src_weight[fan_out_ind]
        temp_fan_out = []

        for fan_out in temp_weight:
            fan_in_var = np.var(fan_out.detach().numpy(), axis=(1,2))
            fan_in_ind = TransferWeight.choose_candidate(fan_in_var, 
                                           choice_type=self.choice_method['keep'], 
                                           num_candidate=self.dst_weight.size(1)) 
            temp_fan_out.append(fan_out[fan_in_ind])
        new_weight = torch.stack(temp_fan_out)
        self.__set_dst_weight(new_weight)


    @torch.no_grad()
    def _up_all(self):
        fan_out_var = np.var(self.dst_weight.detach().numpy(), axis=(1,2,3))
        fan_out_ind = TransferWeight.choose_candidate(fan_out_var,
                                       choice_type=self.choice_method['remove'],
                                       num_candidate=self.src_weight.size(0))

        for i, f_ind in enumerate(fan_out_ind):
            fan_in_var = np.var(self.dst_weight[f_ind].detach().numpy(), axis=(1,2))
            fan_in_ind = TransferWeight.choose_candidate(fan_in_var,
                                           choice_type=self.choice_method['remove'],
                                           num_candidate=self.src_weight.size(1))
            for j, k_ind in enumerate(fan_in_ind):
                self.dst_weight[f_ind][k_ind] = self.__get_src_weight()[i][j]


    @torch.no_grad()
    def _fout_up(self):
        fan_out_var = np.var(self.dst_weight.detach().numpy(), axis=(1,2,3))
        fan_out_ind = TransferWeight.choose_candidate(fan_out_var,
                                       choice_type=self.choice_method['remove'],
                                       num_candidate=self.src_weight.size(0))
        for i, f_ind in enumerate(fan_out_ind):
            self.dst_weight[f_ind] = self.__get_src_weight()[i]



    @torch.no_grad()
    def _fout_down_fin_up(self):
        # - stage 1: (512, 512, 5, 5) -> (256, 512, 5, 5) - fout_down
        temp_dst_weight = copy.deepcopy(self.__get_dst_weight())
        temp_src_size = (self.dst_weight.size(0),
                    self.src_weight.size(1),
                    self.src_weight.size(2),
                    self.src_weight.size(3))
        self.__set_dst_weight(torch.randn(temp_src_size))
        self._fout_down()
        self.__set_src_weight(self.__get_dst_weight())
        self.__set_dst_weight(temp_dst_weight)

        # - stage 2: (256, 512, 5, 5) -> (256, 1024, 5, 5)
        self._fin_up()


    @torch.no_grad()
    def _fout_up_fin_down(self):
        target_dst_size = self.__get_dst_size()
        temp_weight = nn.init.kaiming_normal_(torch.randn(self.dst_weight.size(0),
                                                          self.src_weight.size(1),
                                                          self.dst_weight.size(-1),
                                                          self.dst_weight.size(-1)),
                                                    mode="fan_out", nonlinearity="relu")
        self.__set_dst_weight(temp_weight)
        self._fout_up()
        self.__set_src_weight(self.__get_dst_weight())
        self.__set_dst_weight(torch.randn(target_dst_size))
        self._fin_down()


    def _shuffe_kernel(self):
        fan_out_ind = np.arange(self.dst_weight.size(0))
        np.random.shuffle(fan_out_ind)
        self.dst_weight = self.dst_weight[fan_out_ind]
        for i in range(self.dst_weight.size(0)):
            fan_in_ind = np.arange(self.dst_weight.size(1))
            np.random.shuffle(fan_in_ind)
            self.dst_weight[i] = self.dst_weight[i][fan_in_ind]


    @torch.no_grad()
    def _change_kernel_size(self, input_tensor: Tensor, kernel_size):
        fan_out, fan_in, _, _ = input_tensor.size()
        new_tensor = torch.randn((fan_out, fan_in, kernel_size, kernel_size))
        std = compute_std(new_tensor.size(), mode="fan_out")
        for i in range(fan_out):
            for j in range(fan_in):
                new_tensor[i][j] = self._pad_pool_kernel(
                                        input_kernel=input_tensor[i][j],
                                        std=std,
                                        kernel_size=kernel_size)
        return new_tensor



    def _pad_pool_kernel(self, input_kernel: Tensor, std, kernel_size: int):
        input_size = input_kernel.size()[-1]
        # Up kernel
        if input_size < kernel_size:
            if self.type_pad == "zero":
                len_pad = int((kernel_size-input_size)/2)
                kernel_target = F.pad(
                    input_kernel, (len_pad, len_pad, len_pad, len_pad), "constant", 0)
            elif self.type_pad == "init":
                kernel_target = torch.zeros(kernel_size, kernel_size)
                n_pad = kernel_size**2 - input_kernel.size(-1)**2
                padding = torch.empty(n_pad)
                padding.normal_(0, std)
                kernel_target[0, :] = padding[:kernel_size]
                kernel_target[1:-1, -1] = padding[kernel_size:kernel_size + 3]
                kernel_target[-1, :] = padding[kernel_size + 3:kernel_size*2 + 3]
                kernel_target[1:-1, 0] = padding[-3:]
                kernel_target[1:-1, 1:-1] = input_kernel
        # Down kernel
        else:
            kernel_size = (input_size - kernel_size) + 1
            if self.type_pool == 'avg':
                kernel_target = F.avg_pool2d(input_kernel.unsqueeze(0), 
                                            kernel_size=kernel_size, 
                                            stride=1).squeeze(0)
            elif self.type_pool == 'max':
                kernel_target = F.max_pool2d(input_kernel.unsqueeze(0), 
                                            kernel_size=kernel_size, 
                                            stride=1).squeeze(0)
        return kernel_target



    @staticmethod
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
            candidate_ind = candidate_sorted[slice(1, len(candidate_sorted), step)]
        else:
            raise Exception(
                "choice_type must in [maxVar, minVar, random, twoTailed, interLeaved]")
        
        if len(candidate_ind) > num_candidate:
            candidate_ind = candidate_ind[:num_candidate]
            
        return candidate_ind



