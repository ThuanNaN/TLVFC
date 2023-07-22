from torch import Tensor
import numpy as np

def crossover(x: Tensor, 
              x_pretrain: Tensor, 
              p_crossover: float):
    len_feat = min(x.size(1), x_pretrain.size(1))
    p = np.random.rand(len_feat)
    indies = np.arange(len_feat)[p < p_crossover]
    x[:,indies] = x_pretrain[:,indies]



def crossover_simp(x: Tensor, 
                    x_pretrain: Tensor, 
                    p: float):
    len_feat = min(x.size(1), x_pretrain.size(1))
    indies = np.random.choice(np.arange(len_feat), size=int(len_feat*p), replace=False)
    # indies = np.random.randint(0, len_feat-1, size=int(len_feat*p))
    x[:,indies] = x_pretrain[:,indies]