import logging
import os

import numpy as np
import torch

from .hf_utils import pull_from_hf_model_hub

logger = logging.getLogger(__name__)

SFT_FILE_NAME = 'pytorch_diff.bin'


def decode_sparse_tensor(encoding):
    size = encoding['size']
    index_steps = encoding['index_steps']
    values = encoding['values']

    indices = np.cumsum(index_steps, dtype=np.int32)
    modulos = np.cumprod(list(size), dtype=np.int32)
    divisors = np.concatenate([[1], modulos[:-1]], dtype=np.int32)
    
    coordinates = np.expand_dims(indices, 0) // np.expand_dims(divisors, 1)
    coordinates = coordinates % np.expand_dims(modulos, 1)

    return torch.sparse_coo_tensor(coordinates, values, size=size).coalesce()


def encode_sparse_tensor(tensor):
    multipliers = np.cumprod([1] + list(tensor.size())[:-1], dtype=np.int64)
    coordinates = np.array(tensor.indices().to('cpu'), dtype=np.int64)
    indices = np.matmul(multipliers, coordinates)
    perm = list(range(len(indices)))
    perm.sort(key=lambda x: indices[x])
    indices = indices[perm]
    index_steps = indices[1:] - indices[:-1]
    index_steps = index_steps.tolist()
    if len(indices) > 0:
        index_steps = [indices[0]] + index_steps
    values = tensor.values().to('cpu')[perm]
    return {
        'size': tensor.size(),
        'index_steps': index_steps,
        'values': values,
    }


class SFT:

    def __init__(self,
        name_or_path=None,
        version=None,
        cache_dir=None,
    ):
        if name_or_path is not None:
            if os.path.isdir(name_or_path):
                sft_dir = name_or_path
            else:
                sft_dir = pull_from_hf_model_hub(
                    name_or_path,
                    version=version,
                    cache_dir=cache_dir
                )

            sft_file = os.path.join(sft_dir, SFT_FILE_NAME)
            tensors = torch.load(sft_file)
            
            if 'diffs' in tensors:
                self.diffs = {
                    p: decode_sparse_tensor(d)
                    for p, d in tensors['diffs'].items()
                }
            else:
                self.diffs = {}

            if 'abs' in tensors:
                self.abs = tensors['abs']
            else:
                self.abs = {}

            if not self.diffs and not self.abs:
                logger.warn(f'Empty SFT {name_or_path}')
        else:
            self.diffs = {}
            self.abs = {}

    def add_param(self, name, tensor, diff=True):
        if diff:
            self.diffs[name] = tensor.to_sparse().coalesce()
        else:
            self.abs[name] = tensor.to('cpu')

    def save(self, save_dir):
        encoded_diffs = {
            n: encode_sparse_tensor(p)
            for n, p in self.diffs.items()
        }
        tensors = {
            'diffs': encoded_diffs,
            'abs': self.abs,
        }
        save_path = os.path.join(save_dir, SFT_FILE_NAME)
        torch.save(tensors, save_path)

    def apply(self, model, with_abs=True):
        with torch.no_grad():
            for name in self.diffs.keys():
                diff = self.diffs[name]
                tensor = model.get_parameter(name)
                if diff.device != tensor.device:
                    diff = diff.to(tensor.device)
                    self.diffs[name] = diff
                tensor += diff

            if with_abs:
                for name, value in self.abs.items():
                    tensor = model.get_parameter(name)
                    tensor.copy_(value)

    def revert(self, model):
        with torch.no_grad():
            for name in self.diffs.keys():
                diff = self.diffs[name]
                tensor = model.get_parameter(name)
                if diff.device != tensor.device:
                    diff = diff.to(tensor.device)
                    self.diffs[name] = diff
                tensor -= diff

