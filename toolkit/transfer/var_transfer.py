import numpy as np
import copy
import torch
import torch.nn as nn
from toolkit.base_transfer import Transfer
import torch.nn.functional as F
from models.init import compute_std


class VarTransfer(Transfer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.var_transfer_kwargs = kwargs

    def transfer_layer(self, tensor_from: torch.Tensor, tensor_to: torch.Tensor, *args, **kwargs) -> None:
        if tensor_from is None or tensor_to is None:
            return
        #Transfer convolution
        if len(tensor_from.size()) == 4:
            self.transfer_conv(tensor_from, tensor_to, **kwargs)

        #Transfer Linear
        elif len(tensor_from.size()) == 2:
            self.transfer_linear_bias(tensor_from, tensor_to)

        #Transfer Bias
        elif len(tensor_from.size()) == 1:
            self.transfer_linear_bias(tensor_from, tensor_to)


    def transfer_linear_bias(self, tensor_from: torch.Tensor, tensor_to: torch.Tensor) -> None:
        if tensor_from is None or tensor_to is None:
            return
        from_slices, to_slices = [], []
        for a, b in zip(tensor_from.shape, tensor_to.shape):
            if a < b:
                ids = torch.randint(a, (b,))
                ids[slice((b - a) // 2, -((b - a + 1) // 2))] = torch.arange(a)
                from_slices.append(ids)
            elif a > b:
                from_slices.append((a - b) // 2 + torch.arange(b))
            else:
                from_slices.append(torch.arange(a))
            to_slices.append(slice(0, b))

        total_unsqueeze = 0
        for i in range(len(from_slices) - 1, -1, -1):
            if isinstance(from_slices[i], torch.Tensor):
                for _ in range(total_unsqueeze):
                    from_slices[i] = from_slices[i].unsqueeze(-1)
                total_unsqueeze += 1

        tensor_to[tuple(to_slices)] = tensor_from[tuple(from_slices)]
 

    def transfer_conv(self,
                      tensor_from: torch.Tensor, 
                      tensor_to: torch.Tensor, 
                      ) -> None:
        kwargs = self.var_transfer_kwargs
        tensor_to_temp = copy.deepcopy(tensor_to)
        src_fout, src_fin, src_k, _ = tensor_from.size()
        dst_fout, dst_fin, dst_k, _ = tensor_to_temp.size()

        resize_k = False if src_k == dst_k else True
        if resize_k:
            tensor_from = VarTransfer._change_kernel_size(input_tensor=tensor_from,
                                                        kernel_size=dst_k,
                                                        **kwargs)
        if src_fout > dst_fout:
            if src_fin > dst_fin:
                new_tensor = VarTransfer._down_all(tensor_from, tensor_to_temp.size(), **kwargs)
            elif src_fin < dst_fin:
                new_tensor = VarTransfer._fout_down_fin_up(tensor_from, tensor_to_temp, **kwargs)
            else:
                new_tensor = VarTransfer._fout_down(tensor_from, tensor_to_temp.size(), **kwargs)
        elif src_fout < dst_fout:
            if src_fin > dst_fin:
                new_tensor = VarTransfer._fout_up_fin_down(tensor_from, tensor_to_temp, **kwargs)
            elif src_fin < dst_fin:
                new_tensor = VarTransfer._up_all(tensor_from, tensor_to_temp, **kwargs)
            else:
                new_tensor = VarTransfer._fout_up(tensor_from, tensor_to_temp, **kwargs)
        else:
            if src_fin > dst_fin:
                new_tensor = VarTransfer._fin_down(tensor_from, tensor_to_temp.size(), **kwargs)
            elif src_fin < dst_fin:
                new_tensor = VarTransfer._fin_up(tensor_from, tensor_to_temp, **kwargs)
            else:
                new_tensor = copy.deepcopy(tensor_from)
        new_tensor = VarTransfer._shuffe_kernel(new_tensor)
        tensor_to[:dst_fout,:dst_fin,:dst_k,:dst_k] = new_tensor[:dst_fout,:dst_fin,:dst_k,:dst_k] 


    @staticmethod
    @torch.no_grad()
    def _fin_down(tensor_from, tensor_to_size, **kwargs):
        temp_weight = []
        for fan_out in tensor_from:
            fan_in_var = np.var(fan_out.detach().numpy(), axis=(1,2))
            fan_in_ind= VarTransfer.choose_candidate(fan_in_var, 
                                           choice_type=kwargs["choice_method"]['keep'], 
                                           num_candidate=tensor_to_size[1])
            temp_weight.append(fan_out[fan_in_ind])
        return torch.stack(temp_weight)

    @staticmethod
    @torch.no_grad()
    def _fin_up(tensor_from, tensor_to, **kwargs):
        for f_ind, fan_out in enumerate(tensor_to):
            fan_in_var = np.var(fan_out.detach().numpy(), axis=(1,2))
            fan_in_ind = VarTransfer.choose_candidate(fan_in_var, 
                                           choice_type=kwargs["choice_method"]['remove'], 
                                           num_candidate=tensor_from.size(1))
            for i, k_ind in enumerate(fan_in_ind):
                fan_out[k_ind] = tensor_from[f_ind][i]
        return tensor_to

    @staticmethod
    @torch.no_grad()
    def _fout_down(tensor_from, tensor_to_size, **kwargs):
        fan_out_var = np.var(tensor_from.detach().numpy(), axis=(1,2,3))
        fan_out_ind = VarTransfer.choose_candidate(fan_out_var, 
                                      choice_type=kwargs["choice_method"]['keep'], 
                                      num_candidate=tensor_to_size[0])
        return tensor_from[fan_out_ind]

    @staticmethod
    @torch.no_grad()
    def _down_all(tensor_from, tensor_to_size, **kwargs):
        fan_out_var = np.var(tensor_from.detach().numpy(), axis=(1,2,3))
        fan_out_ind = VarTransfer.choose_candidate(fan_out_var, 
                                       choice_type=kwargs["choice_method"]['keep'], 
                                       num_candidate=tensor_to_size[0]) 
        temp_weight = tensor_from[fan_out_ind]
        temp_fan_out = []

        for fan_out in temp_weight:
            fan_in_var = np.var(fan_out.detach().numpy(), axis=(1,2))
            fan_in_ind = VarTransfer.choose_candidate(fan_in_var, 
                                           choice_type=kwargs["choice_method"]['keep'], 
                                           num_candidate=tensor_to_size[1]) 
            temp_fan_out.append(fan_out[fan_in_ind])
        return torch.stack(temp_fan_out)


    @staticmethod
    @torch.no_grad()
    def _up_all(tensor_from, tensor_to, **kwargs):
        fan_out_var = np.var(tensor_to.detach().numpy(), axis=(1,2,3))
        fan_out_ind = VarTransfer.choose_candidate(fan_out_var,
                                       choice_type=kwargs["choice_method"]['remove'],
                                       num_candidate=tensor_from.size(0))

        for i, f_ind in enumerate(fan_out_ind):
            fan_in_var = np.var(tensor_to[f_ind].detach().numpy(), axis=(1,2))
            fan_in_ind = VarTransfer.choose_candidate(fan_in_var,
                                           choice_type=kwargs["choice_method"]['remove'],
                                           num_candidate=tensor_from.size(1))
            for j, k_ind in enumerate(fan_in_ind):
                tensor_to[f_ind][k_ind] = tensor_from[i][j]
        return tensor_to

    @staticmethod
    @torch.no_grad()
    def _fout_up(tensor_from, tensor_to, **kwargs):
        fan_out_var = np.var(tensor_to.detach().numpy(), axis=(1,2,3))
        fan_out_ind = VarTransfer.choose_candidate(fan_out_var,
                                       choice_type=kwargs["choice_method"]['remove'],
                                       num_candidate=tensor_from.size(0))
        for i, f_ind in enumerate(fan_out_ind):
            tensor_to[f_ind] = tensor_from[i]
        return tensor_to

    @staticmethod
    @torch.no_grad()
    def _fout_down_fin_up(tensor_from, tensor_to, **kwargs):
        # (512, 512, 5, 5) -> (256, 1024, 5, 5)
        # (256, 512, 5, 5)
        temp_src_size = (tensor_to.size(0),
                        tensor_from.size(1),
                        tensor_from.size(2),
                        tensor_from.size(3))
        # - stage 1: (512, 512, 5, 5) -> (256, 512, 5, 5) - fout_down
        temp_w = VarTransfer._fout_down(tensor_from, temp_src_size, **kwargs)
        # - stage 2: (256, 512, 5, 5) -> (256, 1024, 5, 5)
        return VarTransfer._fin_up(temp_w, tensor_to, **kwargs)


    @staticmethod
    @torch.no_grad()
    def _fout_up_fin_down(tensor_from, tensor_to, **kwargs):
        #(512, 512, 3, 3) -> (1024, 256, 3, 3)
        #(1024, 512, 3, 3)
        temp_weight = nn.init.kaiming_normal_(torch.randn(tensor_to.size(0),
                                                          tensor_from.size(1),
                                                          tensor_to.size(-1),
                                                          tensor_to.size(-1)),
                                                    mode="fan_out", nonlinearity="relu")
        #(512, 512, 3, 3) -> (1024, 512, 3, 3)
        temp_weight = VarTransfer._fout_up(tensor_from, temp_weight, **kwargs)
        return VarTransfer._fin_down(temp_weight, tensor_to.size(), **kwargs)


    @staticmethod
    @torch.no_grad()
    def _shuffe_kernel(weight):
        fan_out_ind = np.arange(weight.size(0))
        np.random.shuffle(fan_out_ind)
        weight = weight[fan_out_ind]
        for i in range(weight.size(0)):
            fan_in_ind = np.arange(weight.size(1))
            np.random.shuffle(fan_in_ind)
            weight[i] = weight[i][fan_in_ind]
        return weight

    @staticmethod
    @torch.no_grad()
    def _change_kernel_size(input_tensor: torch.Tensor, kernel_size, **kwargs):
        fan_out, fan_in, _, _ = input_tensor.size()
        new_tensor = torch.randn((fan_out, fan_in, kernel_size, kernel_size))
        std = compute_std(new_tensor.size(), mode="fan_out")
        for i in range(fan_out):
            for j in range(fan_in):
                new_tensor[i][j] = VarTransfer._pad_pool_kernel(
                                        input_kernel=input_tensor[i][j],
                                        std=std,
                                        kernel_size=kernel_size,
                                        **kwargs)
        return new_tensor


    @staticmethod
    def _pad_pool_kernel(input_kernel: torch.Tensor, std, kernel_size: int, **kwargs):
        input_size = input_kernel.size()[-1]
        # Up kernel
        if input_size < kernel_size:
            if kwargs["type_pad"] == "zero":
                len_pad = int((kernel_size-input_size)/2)
                kernel_target = F.pad(
                    input_kernel, (len_pad, len_pad, len_pad, len_pad), "constant", 0)
            elif kwargs["type_pad"] == "init":
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
            if kwargs["type_pool"] == 'avg':
                kernel_target = F.avg_pool2d(input_kernel.unsqueeze(0), 
                                            kernel_size=kernel_size, 
                                            stride=1).squeeze(0)
            elif kwargs["type_pool"] == 'max':
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



