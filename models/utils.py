import torch
from torch import Tensor

def uniform_crossover(x: Tensor, 
              x_pretrain: Tensor, 
              p_crossover: float):
    len_feat = min(x.size(1), x_pretrain.size(1))
    p = torch.rand(len_feat)
    indies = torch.arange(len_feat)[p < p_crossover]
    x[:,indies] = x_pretrain[:,indies]



def uniform_crossover_simp(x: Tensor, 
                    x_pretrain: Tensor, 
                    p: float):
    len_feat = min(x.size(1), x_pretrain.size(1))
    indies = torch.randint(0, len_feat-1, size=(int(len_feat*p),))
    x[:,indies] = x_pretrain[:,indies]
