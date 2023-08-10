from typing import List, Tuple, Any, Optional
import torch.nn as nn
from toolkit.base_matching import Matching



class IndexMatching(Matching):
    def match(self, 
              from_module: List[nn.Module],
              to_module: List[nn.Module],
              *args, **kwargs) \
                -> List[Tuple[nn.Module, nn.Module]]:
        
        matched, matched_indices = self._match_models(from_module, to_module)
        return matched

    def _match_models(self, 
                    from_module: List[nn.Module], 
                    to_module: List[nn.Module]) \
                        -> List[Tuple[Optional[nn.Module]]]:

        from_features, from_bn, from_fc = IndexMatching.classify_module(from_module)
        to_features, to_bn, to_fc = IndexMatching.classify_module(to_module)

        if len(from_features) != 0 and len(to_features) != 0:
            feature_match, feature_match_ind = self._index_match(from_features, to_features)
        else:
            feature_match, feature_match_ind = [], []
            
        if len(from_fc) != 0 and len(to_fc) != 0:
            fc_match, fc_match_ind = self._index_match(from_fc, to_fc, invert=True)
        else:
            fc_match, fc_match_ind = [], []

        if len(from_bn) != 0 and len(to_bn) != 0:
            bn_match, bn_match_ind = self._index_match(from_bn, to_bn)
        else:
            bn_match, bn_match_ind = [], []

        matched = feature_match + bn_match + fc_match
        matched_indices = feature_match_ind + bn_match_ind + fc_match_ind

        return matched, matched_indices

    def _index_match(self, 
                    from_module: List[nn.Module], 
                    to_module: List[nn.Module],
                    invert: bool = False
                    ) -> List[Tuple[Optional[nn.Module]]]:
        if invert:
            from_module = from_module[::-1]
        len_to_module = len(to_module)
        len_from_module = len(from_module)
        scale = len_to_module / len_from_module
        matched, matched_indices = [], []
        for layer_idx in range(len_to_module):
            src_index = self._absolute_mapping(len_from_module=len_from_module,
                                                dst_index=layer_idx,
                                                scale=scale)
            matched.append((from_module[src_index], to_module[layer_idx]))
            matched_indices.append((src_index, layer_idx))
        return matched, matched_indices

    def _absolute_mapping(self, 
                        len_from_module: int,
                        dst_index:int,
                        scale:float,
                        zero_point:float = 0.0):
        src_index = round((1/scale)*dst_index + zero_point)
        if src_index < 0:
            return 0
        elif src_index >= len_from_module:
            return len_from_module - 1
        else:
            return src_index

    @staticmethod
    def classify_module(modules: List[nn.Module]):
        #Convolution
        features = []
        feature_class = [nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d]
        #Normalize
        batch_norm = []
        bn_class = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
        #Fully-connected
        fc = []
        fc_class = [nn.Linear]

        for module in modules:
            if any([isinstance(module, clazz) for clazz in feature_class]):
                features.append(module)
            elif any([isinstance(module, clazz) for clazz in bn_class]):
                batch_norm.append(module)
            else:
                fc.append(module)
        return (features, batch_norm, fc)





