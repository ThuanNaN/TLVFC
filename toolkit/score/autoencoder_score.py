from typing import Dict, List, Tuple
from torch import nn
from toolkit.base_score import Score
from toolkit.tli import score_autoencoder, get_graph

def get_model_graph_and_ids_mapping(model: nn.Module):
    graph = get_graph(model)
    names_to_layers_mapping = {}
    def dfs(model: nn.Module, name_prefix: List[str]):
        for child_name, child in model.named_children():
            dfs(child, name_prefix + [child_name])
        names_to_layers_mapping[".".join(name_prefix)] = model
    dfs(model, [])
    
    layers_to_ids_mapping = {}
    for node in graph.nodes.values():
        if node.name.endswith(".weight"):
            layer = names_to_layers_mapping[node.name.replace(".weight", "")]
            layers_to_ids_mapping[layer] = (node.idx, layers_to_ids_mapping.get(layer, (None, None))[1])
        elif node.name.endswith(".bias"):
            layer = names_to_layers_mapping[node.name.replace(".bias", "")]
            layers_to_ids_mapping[layer] = (layers_to_ids_mapping.get(layer, (None, None))[0], node.idx)
    
    return graph, layers_to_ids_mapping


class AutoEncoderScore(Score):
    def precompute_scores(self, from_model: nn.Module, to_model: nn.Module, *args, **kwargs) \
            -> float:
        teacher_graph, teacher_mapping = get_model_graph_and_ids_mapping(from_model)
        student_graph, student_mapping = get_model_graph_and_ids_mapping(to_model)
        self.scores, teacher_arr, student_arr = score_autoencoder(teacher_graph, student_graph)
        self.teacher_mapping = self._join_dicts(teacher_mapping, dict([(x, i) for i, x in enumerate(teacher_arr)]))
        self.student_mapping = self._join_dicts(student_mapping, dict([(x, i) for i, x in enumerate(student_arr)]))

    def _join_dicts(self, a: Dict, b: Dict) -> Dict:
        res = {}
        for key, value in a.items():
            res[key] = b[value[0]]
        return res

    def score(self, from_module: nn.Module, to_module: nn.Module, *args, **kwargs) \
            -> float:
        try:
            return self.scores[self.teacher_mapping[from_module]][self.student_mapping[to_module]]
        except KeyError:
            return 0