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
            components = torch.load(sft_file)
            
            if 'body' in components:
                self.diffs = {
                    p: decode_sparse_tensor(d)
                    for p, d in components['body'].items()
                }
            else:
                raise ValueError(f'SFT {name_or_path} is missing a body')

            if 'head' in components:
                self.head = components['head']
            else:
                self.head = None
        else:
            self.diffs = {}
            self.head = None

    def add_tensor(self, param_name, dense_tensor):
        self.diffs[param_name] = dense_tensor.to_sparse().coalesce()

    def save(self, save_dir):
        components = {
            'body': self.diffs,
        }
        if self.head is not None:
            components['head'] = self.head

        save_path = os.path.join(save_dir, SFT_FILE_NAME)
        torch.save(components, save_path)

    def apply(self, model, with_head=None):
        with torch.no_grad():
            for param_name, diff in self.diffs.items():
                logger.info(param_name)
                param_tensor = model.get_parameter(param_name)
                param_tensor += diff.to(param_tensor.device)

            if with_head or (with_head is None and self.head is not None):
                if self.head is None:
                    raise ValueError('Received with_head=True but no head present.')

                for param_name, value in self.head.items():
                    logger.info(param_name)
                    param_tensor = model.get_parameter(param_name)
                    param_tensor.copy_(value)

    def revert(self, model):
        with torch.no_grad():
            for param_name, diff in self.diffs.items():
                param_tensor = model.get_parameter(param_name)
                param_tensor -= diff.to(param_tensor.device)

