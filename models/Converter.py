import numpy as np
import torch.nn as nn
from models.TranferKernel import TransferKernel


class Converter():
    def __init__(self,
                 model,
                 ckpt,
                 feature_index,
                 candidate_method,
                 mapping_type
                 ):
        self.model = model
        self.ckpt = ckpt
        self.feature_index = feature_index
        self.candidate_method = candidate_method
        self.mapping_type = mapping_type

    def _load_weight(self, **kwargs):
        print("\n🚅 Starting tranfer weight from pre-trained weight")
        model_dict = self._get_model_dict()
        ckpt_dict = self._get_ckpt_dict()
        new_model_dict = self._matching_weight(model_dict,
                                               ckpt_dict,
                                               **kwargs)
        self._load_weigth_from_dict(new_model_dict)
        return self.model

    def _matching_weight(self, model_dict, ckpt_dict, num_candidate, **kwargs):
        new_model_dict = {}
        scale = (len(ckpt_dict)) / (len(model_dict))
        for layer_idx in range(len(model_dict)):
            if scale != 1.0:
                if self.mapping_type == "relative":
                    candidate_idx = self._relative_mapping(num_layer_ckpt=len(ckpt_dict),
                                                            w_index=layer_idx,
                                                            scale=scale,
                                                            k_candidate=num_candidate)

                    weight_var = [ckpt_dict[f"conv_{cdd}"].var() for cdd in candidate_idx]
                    if self.candidate_method == "max":
                        max_var_idx = np.argsort(weight_var)[-1:][0]
                    else:
                        max_var_idx = np.argsort(weight_var)[:-1][0]
                    conv_index = candidate_idx[max_var_idx]

                elif self.mapping_type == "absolute":
                    conv_index = self._absolute_mapping(num_layer_ckpt=len(ckpt_dict),
                                                        w_index=layer_idx,
                                                        scale=scale)

                new_w = TransferKernel(ckpt_weight=ckpt_dict[f"conv_{conv_index}"],
                                       model_weight=model_dict[f"conv_{layer_idx}"],
                                       )._choose_kernel(**kwargs)
            else:
                new_w = TransferKernel(ckpt_weight=ckpt_dict[f"conv_{layer_idx}"],
                                       model_weight=model_dict[f"conv_{layer_idx}"],
                                       )._choose_kernel(**kwargs)

            new_model_dict[layer_idx] = new_w

        return new_model_dict

    def _get_model_dict(self):
        model_dict = {}
        index = 0
        for _, module in self.model.features._modules.items():
            if module.__class__.__name__ == 'Conv2d':
                model_dict[f"conv_{index}"] = module.weight
                index += 1
        return model_dict

    def _get_ckpt_dict(self):
        ckpt_dict = {}
        for index, idx in enumerate(self.feature_index):
            ckpt_dict[f"conv_{index}"] = self.ckpt[f"features.{idx}.weight"]
        return ckpt_dict

    def _relative_mapping(self, num_layer_ckpt, w_index, scale, k_candidate):
        w_pos = w_index*scale
        list_score = np.argsort([(1/((w_pos-i)**2+1))
                                for i in range(num_layer_ckpt)])[-k_candidate:]
        return list_score

    def _absolute_mapping(self, num_layer_ckpt, w_index, scale, zero_point = 0):
        xq = round((1/scale)*w_index + zero_point)
        if xq < 0:
            return 0
        elif xq >= num_layer_ckpt:
            return num_layer_ckpt - 1
        else:
            return xq

    def _load_weigth_from_dict(self, model_dict):
        index = 0
        for _, module in self.model.features._modules.items():
            if module.__class__.__name__ == 'Conv2d':
                new_w = model_dict[index]
                module.weight = nn.parameter.Parameter(new_w)
                index += 1
