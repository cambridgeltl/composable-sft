import logging
import os

import numpy as np
import torch

from .hf_utils import pull_from_hf_model_hub

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
        map_fn=None,
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

            expected_total_params = 0
            if 'diffs' in tensors:
                self.diffs = {
                    n if map_fn is None else map_fn(n): decode_sparse_tensor(d)
                    for n, d in tensors['diffs'].items()
                }
                expected_total_params += len(tensors['diffs'])
            else:
                self.diffs = {}

            if 'abs' in tensors:
                self.abs = {
                    n if map_fn is None else map_fn(n): p
                    for n, p in tensors['abs'].items()
                }
                expected_total_params += len(tensors['abs'])
            else:
                self.abs = {}

            total_params = len(set(self.diffs.keys()) | set(self.abs.keys()))
            if total_params < expected_total_params:
                if map_fn is None:
                    shared_params = set(self.diffs.keys()) & set(self.abs.keys())
                    raise RuntimeError(
                        f'SFT {name_or_path} contained both differences and dense values '
                        f'for the following parameters: {sorted(list(shared_params))}.'
                    )
                else:
                    seen = set()
                    duplicates = []
                    for n in (
                        list(tensors.get('diffs', {}).keys()) +
                        list(tensors.get('abs', {}).keys())
                    ):
                        mapped_n = map_fn(n)
                        if mapped_n in seen:
                            duplicates.append(f'{mapped_n} <- {n}')
                        else:
                            seen.add(mapped_n)
                    duplicates = '\n'.join(sorted(duplicates))
                    raise RuntimeError(
                        f'The following duplicate mappings arose while loading SFT '
                        f'{name_or_path}:\n{duplicates}.'
                    )

            if total_params == 0:
                logger.error(f'Empty SFT {name_or_path}')
            
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

    def apply(
        self,
        model,
        with_abs=True,
        allow_unused=False,
        warn=False,
    ):
        with torch.no_grad():
            if warn:
                unused = []
                all_params = set(n for n, _ in model.named_parameters())

            for name in self.diffs.keys():
                #logger.info(name)
                diff = self.diffs[name]
                try:
                    tensor = model.get_parameter(name)
                except AttributeError:
                    if allow_unused:
                        if warn:
                            unused.append(name)
                        continue
                    else:
                        raise

                if diff.device != tensor.device:
                    diff = diff.to(tensor.device)
                    self.diffs[name] = diff
                tensor += diff

                if warn:
                    all_params.remove(name)

            if with_abs:
                for name, value in self.abs.items():
                    try:
                        tensor = model.get_parameter(name)
                    except AttributeError:
                        if allow_unused:
                            if warn:
                                unused.append(name)
                            continue
                        else:
                            raise
                    
                    tensor.copy_(value)

                    if warn:
                        all_params.remove(name)
            
            if warn:
                if unused:
                    logger.info('The following SFT parameters were not present in the base model:')
                    for n in sorted(unused):
                        logger.info(n)

                if all_params:
                    logger.info('The following base model parameters were not present in the SFT:')
                    for n in sorted(list(all_params)):
                        logger.info(n)

    def revert(self, model):
        with torch.no_grad():
            for name in self.diffs.keys():
                diff = self.diffs[name]
                tensor = model.get_parameter(name)
                if diff.device != tensor.device:
                    diff = diff.to(tensor.device)
                    self.diffs[name] = diff
                tensor -= diff

