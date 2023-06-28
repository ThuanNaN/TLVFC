import math
import warnings
from torch import Tensor
import torch



def _calculate_correct_fan(tensor_size, mode):
    num_input_fmaps = tensor_size[1]
    num_output_fmaps = tensor_size[0]
    receptive_field_size = 1
    for s in tensor_size[2:]:
            receptive_field_size *= s
    if mode == "fan_in":
        fan_in = num_input_fmaps * receptive_field_size
        return fan_in
    elif mode == "fan_out":
        fan_out = num_output_fmaps * receptive_field_size
        return fan_out

def calculate_gain(nonlinearity: str = 'relu', a: float = 0):
    if nonlinearity == 'relu':
        return math.sqrt(2.0)



def compute_std(target_size, mode="fan_out"):
    fan = _calculate_correct_fan(target_size, mode)
    gain = calculate_gain()
    std = gain / math.sqrt(fan)

    return std