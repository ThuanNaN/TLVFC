from typing import Dict, List, Tuple

from torch import nn

from toolkit.tli import get_graph


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