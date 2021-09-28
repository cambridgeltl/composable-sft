import torch


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

    def __init__(self, from_file=None):
        if from_file is not None:
            diffs = torch.load(from_file)
            self.diffs = {
                p: SparseTensorDifference(from_dict=d)
                for p, d in diffs.items()
            }
        else:
            self.diffs = {}

    def add_tensor(self, param_name, diff):
        self.diffs[param_name] = SparseTensorDifference(dense_tensor=diff)

    def save(self, path):
        torch.save(self.diffs, path)

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

