import os

import torch

from .hf_utils import pull_from_hf_model_hub

SFT_FILE_NAME = 'pytorch_diff.bin'


class SparseTensorDifference:

    def __init__(self, from_dict=None, dense_tensor=None, size=None):
        if from_dict is not None:
            self.size = from_dict['size']
            self.indices = from_dict['indices']
            self.values = from_dict['values']
        elif dense_tensor is None:
            self.size = size
            self.indices = [[] for i in size]
            self.values = []
        else:
            sparse_tensor = dense_tensor.to_sparse().coalesce()
            self.size = sparse_tensor.size()
            self.indices = sparse_tensor.indices().tolist()
            self.values = sparse_tensor.values().tolist()

    def add(self, index, value):
        for d, i in enumerate(index):
            self.indices[d].append(i)
        self.values.append(value)

    def to_tensor(self):
        return torch.sparse_coo_tensor(self.indices, self.values, self.size)

    def get_indices(self):
        return [
            tuple(self.indices[j][i] for j in range(len(self.size)))
            for i in range(len(self.values))
        ]


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
            diffs = torch.load(sft_file)
            self.diffs = {
                p: SparseTensorDifference(from_dict=d)
                for p, d in diffs.items()
            }
        else:
            self.diffs = {}

    def add_tensor(self, param_name, diff):
        self.diffs[param_name] = SparseTensorDifference(dense_tensor=diff)

    def save(self, save_dir):
        save_path = os.path.join(save_dir, SFT_FILE_NAME)
        torch.save(self.diffs, save_path)

    def apply(self, model):
        with torch.no_grad():
            for param_name, diff in self.diffs.items():
                param_tensor = model.get_parameter(param_name)
                param_tensor += diff.to_tensor().to(param_tensor.device)

    def revert(self, model):
        with torch.no_grad():
            for param_name, diff in self.diffs.items():
                param_tensor = model.get_parameter(param_name)
                param_tensor -= diff.to_tensor().to(param_tensor.device)

