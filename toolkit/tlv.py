from typing import Any, List, Set, Dict, Union, Tuple

import torch.nn as nn
from inflection import camelize

from toolkit.base_standardization import Standardization
from toolkit.base_matching import Matching
from toolkit.base_transfer import Transfer

from toolkit.transfer.transfer_stats import TransferStats

from toolkit.utils.flatten import flatten_modules
from toolkit.utils.dot_dict import DotDict
from toolkit.utils.subclass_utils import get_subclasses


class TLV:
    standardization: Standardization = None
    matching: Matching = None
    transfer: Transfer = None

    def run(self, from_module: nn.Module, to_module: nn.Module, *args, **kwargs):
        context = {'from_module': from_module, 'to_module': to_module}

        flat_module_from = set(flatten_modules(from_module))
        flat_module_to = set(flatten_modules(to_module))
        from_paths, to_paths = self.standardization(from_module), self.standardization(to_module)

        # self.score.precompute_scores(from_module, to_module)
        # self.matching.set_score(self.score)
        matched_tensors = self.matching(from_paths, to_paths, context=context)
        self.transfer(matched_tensors, context=context)

        all_from = len(flat_module_from)
        all_to = len(flat_module_to)
        self._flat_remove(matched_tensors, flat_module_from, flat_module_to)

        return TransferStats(
            all_from=all_from,
            all_to=all_to,
            left_from=len(flat_module_from),
            left_to=len(flat_module_to),
            matched_from=all_from - len(flat_module_from),
            matched_to=all_to - len(flat_module_to)
        )



    def _flat_remove(self, matched: List, flat_from: Set[nn.Module], flat_to: Set[nn.Module]):
        for x in matched:
            if isinstance(x, list):
                self._flat_remove(x, flat_from, flat_to)
            else:
                tensor_from, tensor_to = x
                if tensor_to and tensor_from:
                    if tensor_from in flat_from:
                        flat_from.remove(tensor_from)
                    if tensor_to in flat_to:
                        flat_to.remove(tensor_to)

    def __call__(self, from_module: nn.Module, to_module: nn.Module, 
                 *args, **kwargs) -> TransferStats:
        return self.run(from_module, to_module, *args, **kwargs)


    def __init__(self, 
                 standardization,
                 matching,
                 transfer,
                 *args, **kwargs
                ):
        ctx = DotDict()
        ctx._standardization_classes = get_subclasses(Standardization)
        ctx._matching_classes = get_subclasses(Matching)
        ctx._transfer_classes = get_subclasses(Transfer)
        # ctx._score_classes = get_subclasses(Score)

        self._try_setting(ctx, 'standardization', standardization, Standardization)
        self._try_setting(ctx, 'matching', matching, Matching)
        self._try_setting(ctx, 'transfer', transfer, Transfer)
        # self._try_setting(ctx, 'score', score, Score)


    @staticmethod
    def make(standardization: Union[Standardization, str, Tuple[Standardization, Dict], Tuple[str, Dict]]='blocks',
             matching: Union[Matching, str, Tuple[Matching, Dict], Tuple[str, Dict]] = 'dp',
             transfer: Union[Transfer, str, Tuple[Transfer, Dict], Tuple[str, Dict]] = 'clip',
            #  score: Union[Score, str, Tuple[Score, Dict], Tuple[str, Dict]] = 'ShapeScore',
             *args,
             **kwargs):
        """Method factory.
        """
        # return TLV(standardization, matching, transfer, score, *args, **kwargs)
        return TLV(standardization, matching, transfer, *args, **kwargs)
    

    def _try_setting(self, ctx: Dict, key: str, value: Any, clazz: Any) -> None:
        kwargs = {}
        if isinstance(value, tuple):
            value, kwargs = value[0], value[1]
        if isinstance(value, str):
            setattr(self, key, self._find_subclass(ctx, key, value)(**kwargs))
        elif isinstance(value, clazz):
            setattr(self, key, value)
        else:
            raise TypeError()


    def _find_subclass(self, ctx: Dict, key: str, value: str) -> Any:
        classes = getattr(ctx, f'_{key}_classes')
        trials = [lambda x: x[1],
                  lambda x: camelize(f'{x[1]}_{x[0]}'),
                  lambda x: camelize(f'{camelize(x[1])}{camelize(x[0])}'),
                  lambda x: camelize(f'{x[1].upper()}{camelize(x[0])}'),
                  lambda x: f'{x[1].split("_")[0].upper()}{camelize("_".join(x[1].split("_")[1:]))}{camelize(x[0])}'
                  ]
        for trial in trials:
            res = classes.get(trial((key, value)))
            if res is not None:
                return res
        raise ValueError()