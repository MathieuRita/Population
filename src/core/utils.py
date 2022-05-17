from typing import Any
from collections import defaultdict
import torch as th
import numpy as np

def find_lengths(messages: th.Tensor) -> th.Tensor:
    """
    :param messages: A tensor of term ids, encoded as Long values, of size (batch size, max sequence length).
    :returns A tensor with lengths of the sequences, including the end-of-sequence symbol <eos> (in EGG, it is 0).
    If no <eos> is found, the full length is returned (i.e. messages.size(1)).
    >>> messages = th.tensor([[1, 1, 0, 0, 0, 1], [1, 1, 1, 10, 100500, 5]])
    >>> lengths = find_lengths(messages)
    >>> lengths
    tensor([3, 6])
    """
    max_k = messages.size(1)
    zero_mask = messages == 0

    lengths = max_k - (zero_mask.cumsum(dim=1) > 0).sum(dim=1)
    lengths.add_(1).clamp_(max=max_k)

    return lengths


def _add_dicts(a, b):
    result = dict(a)
    for k, v in b.items():
        result[k] = result.get(k, 0) + v
    return result

def _div_dict(d, n):
    result = dict(d)
    if type(n)==int:
        for k in result:
            result[k] /= n
    elif type(n)==dict:
        for k in result:
            result[k] /= n[k]
    else:
        raise "Error with n type"
    return result

def move_to(x: Any, device: th.device)-> Any:
    """
    Simple utility function that moves a tensor or a dict/list/tuple of (dict/list/tuples of ...) tensors to a specified device, recursively.
    :param x: tensor, list, tuple, or dict with values that are lists, tuples or dicts with values of ...
    :param device: device to be moved to
    :return: Same as input, but with all tensors placed on device. Non-tensors are not affected. For dicts, the changes are done in-place!
    """
    if hasattr(x, 'to'):
        return x.to(device)
    if isinstance(x, list) or isinstance(x, tuple):
        return [move_to(i, device) for i in x]
    if isinstance(x, dict) or isinstance(x, defaultdict):
        for k, v in x.items():
            x[k] = move_to(v, device)
        return x
    return x

def from_att_to_one_hot_celeba(attribute_dict):
    return 1*np.array([v for k,v in attribute_dict.items()])
