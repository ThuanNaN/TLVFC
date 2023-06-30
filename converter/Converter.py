from collections import OrderedDict
import numpy as np
import copy
import torch
from torch import Tensor
from converter.TransferWeight import TransferWeight 


class Converter():
    def __init__(self,
                 dst_feature_dict,
                 src_feature_dict,
                 mapping_type,
                 candidate_method
                 
                 ):
        self.dst_feature_dict = dst_feature_dict
        self.dst_feature_list = list(self.dst_feature_dict.items())

        self.src_feature_dict = src_feature_dict
        self.src_feature_list = list(self.src_feature_dict.items())

        self.mapping_type = mapping_type
        self.candidate_method = candidate_method


    def _matching(self, num_candidate, **kwargs):
        new_feature_dict = OrderedDict()
        len_src_feature_dict = len(self.src_feature_dict)
        len_dst_feature_dict = len(self.dst_feature_dict)
        scale = len_src_feature_dict / len_dst_feature_dict

        for layer_idx in range(len_dst_feature_dict):
            if scale != 1.0:
                if self.mapping_type == "relative":
                    lst_candidate = self._relative_mapping(num_src_layer=len_src_feature_dict,
                                                            dst_index=layer_idx,
                                                            scale=scale,
                                                            num_candidate=num_candidate)

                    weight_var = [self.src_feature_list[cdd][1].var() \
                                                for cdd in lst_candidate]

                    if self.candidate_method == "max":
                        can_index = np.argsort(weight_var)[-1:][0]
                    elif self.candidate_method == "min":
                        can_index = np.argsort(weight_var)[:-1][0]
                    else:
                        raise Exception("candidate_method must be in ('max', 'min')")
                    
                    src_index = lst_candidate[can_index]

                elif self.mapping_type == "absolute":
                    src_index = self._absolute_mapping(num_src_layer=len_src_feature_dict,
                                                        dst_index=layer_idx,
                                                        scale=scale)
                    
                src_weight_name, src_weight = self.src_feature_list[src_index]

            else:
                src_weight_name, src_weight = self.src_feature_list[layer_idx]
            
            dst_weight_name, dst_weight = self.dst_feature_list[layer_idx]

            copied_weight = TransferWeight(src_weight=src_weight,
                                            dst_weight=dst_weight,
                                            **kwargs)._transfer()

            # copied_weight = replace_center(
            #     src_tensor=src_weight,
            #     dst_tensor=dst_weight
            # )
                            
            new_feature_dict[dst_weight_name] = copied_weight
        return new_feature_dict


    def _relative_mapping(self, num_src_layer, dst_index, scale, num_candidate):
        w_pos = dst_index*scale
        list_score = np.argsort([(1/((w_pos-i)**2+1))
                                for i in range(num_src_layer)])[-num_candidate:]
        return list_score


    def _absolute_mapping(self, num_src_layer, dst_index, scale, zero_point = 0):
        src_index = round((1/scale)*dst_index + zero_point)
        if src_index < 0:
            return 0
        elif src_index >= num_src_layer:
            return num_src_layer - 1
        else:
            return src_index


@torch.no_grad()
def replace_center(src_tensor: Tensor, dst_tensor: Tensor) -> Tensor:
    new_weight = copy.deepcopy(dst_tensor)
    src_slices, dst_slices = [], []
    for src_shape, dst_shape in zip(src_tensor.shape, dst_tensor.shape):
        if src_shape < dst_shape:
            src_slices.append(slice(0,src_shape))
            dst_slices.append(slice((dst_shape - src_shape) // 2, \
                                    -((dst_shape - src_shape + 1) // 2)))
        elif src_shape > dst_shape:
            src_slices.append(slice((src_shape - dst_shape) // 2, \
                                    -((src_shape - dst_shape + 1) // 2)))
            dst_slices.append(slice(0, dst_shape))
        else:
            src_slices.append(slice(0, src_shape))
            dst_slices.append(slice(0, dst_shape))
    new_weight[tuple(dst_slices)] = src_tensor[tuple(src_slices)]
    return new_weight