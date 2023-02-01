import json
import logging
import math
import random

from typing import Dict, Optional

import numpy as np
import torch

import datasets
from torch.utils.data import (
    Dataset,
    IterableDataset,
    DataLoader,
    Sampler,
)

from .sft import SFT

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

BATCH_SOURCE_KEY = '_source'


class MultiSourceDataset(Dataset):

    def __init__(self, datasets: Dict[str, Dataset], upsampling={}):
        logger.info('Initialising multi-source dataset with subsets:')
        for source, dataset in sorted(list(datasets.items())):
            upsampling_info = f' (x{upsampling[source]} upsampling)' if source in upsampling else ''
            logger.info(f'{source}: {len(dataset)} examples{upsampling_info}')
        self.datasets = datasets
        self.upsampling = dict(upsampling)

        self.sequential_order = [
            (source, index)
            for source, dataset in sorted(self.datasets.items())
            for index in range(len(dataset))
            for repetition in range(upsampling.get(source, 1))
        ]

    def map(self, *args, **kwargs):
        return MultiSourceDataset(
            {
                    dataset_name: dataset.map(*args, **kwargs)
                for dataset_name, dataset in self.datasets.items()
            },
            upsampling=self.upsampling
        )

    def filter(self, *args, **kwargs):
        return MultiSourceDataset(
            {
                dataset_name: dataset.filter(*args, **kwargs)
                for dataset_name, dataset in self.datasets.items()
            },
            upsampling=self.upsampling
        )

    @property
    def column_names(self):
        names = None
        for dataset in self.datasets.values():
            if names is None:
                names = dataset.column_names
            elif dataset.column_names != names:
                raise ValueError(
                    f'Inconsistent column names in sub-datasets: '
                    f'{names} and {dataset.column_names}'
                )
        return names

    @property
    def features(self):
        shared_feats = None
        feats_source = None
        unshared_feats = set()
        for source, dataset in self.datasets.items():
            if shared_feats is None:
                shared_feats = dataset.features
                feats_source = source
            else:
                for feat_name, feat_def in dataset.features.items():
                    if feat_name in shared_feats:
                        if feat_def != shared_feats[feat_name]:
                            raise ValueError(
                                f'Inconsistent feature definitions in sub-datasets:\n'
                                f'({feats_source}): {shared_feats[feat_name]}\n'
                                f'and\n'
                                f'({source}): {feat_def}'
                            )
                    else:
                        unshared_feats.add(feat_name)

        if unshared_feats:
            unshared_feats = sorted(list(unshared_feats))
            logger.warn(
                f'Some features were not shared across all sub-datasets:'
                f'{unshared_feats}'
            )

        return shared_feats

    def __getitem__(self, index):
        source, subindex = self.sequential_order[index]
        return self.datasets[source][subindex]

    def __iter__(self):
        # Implement as well as __getitem__ in case sub-datasets are IterableDatasets
        for source, dataset in sorted(self.datasets.items()):
            for item in dataset:
                yield item

    def __len__(self):
        return len(self.sequential_order)


class MultiSourceDataLoader(object):

    def __init__(self, dataset, loaders, sampling_policy='random'):
        self.dataset = dataset
        self.loaders = loaders
        self.sampling_policy = sampling_policy
        self.length = sum(len(loader) for loader in self.loaders.values())
        self.batch_size = None
        for loader in loaders.values():
            if self.batch_size is None:
                self.batch_size = loader.batch_size
            elif loader.batch_size != self.batch_size:
                raise ValueError('Loaders have inconsistent batch sizes')

    def __iter__(self):
        order = [
            source
            for source, dataset in sorted(self.loaders.items())
            for _ in range(len(dataset))
        ]
        if self.sampling_policy == 'random':
            random.shuffle(order)
        elif self.sampling_policy != 'sequential':
            raise ValueError(
                f'Unrecognised sampling policy "{self.sampling_policy}". '
                f'Must be one of "random" or "sequential"'
            )
        iterators = {
            source: iter(dataset)
            for source, dataset in self.loaders.items()
        }
        for source in order:
            batch = next(iterators[source])
            batch[BATCH_SOURCE_KEY] = source
            yield batch

    def __len__(self):
        return self.length


def load_single_dataset(
    dataset_json,
    training_args,
    source=None,
    split=None,
    preprocessor=None,
    provide_source_to_preprocessor=None,
    cache_dir=None,
    max_seq_length=None,
    preprocessing_num_workers=None,
    overwrite_cache=False,
    remove_original_columns=False,
):
    split_map = {
        s: dataset_json.get(f'{s}_split', s)
        for s in ['train', 'validation', 'test']
    }

    load_kwargs = dataset_json.get('load_kwargs', {})
    if 'name' in dataset_json:
        kwargs = {}
        if 'config_name' in dataset_json:
            load_kwargs['name'] = dataset_json['config_name']

        raw_datasets = datasets.load_dataset(
            dataset_json['name'],
            cache_dir=cache_dir,
            **load_kwargs,
        )
    else:
        data_files = {}
        if 'train_file' in dataset_json:
            data_files['train'] = dataset_json['train_file']
        if 'validation_file' in dataset_json:
            data_files['validation'] = dataset_json['validation_file']
        if 'test_file' in dataset_json:
            data_files['test'] = dataset_json['test_file']
        if 'file_type' in dataset_json:
            file_type = dataset_json['file_type']
        else:
            file_name = list(data_files.keys())[0]
            file_type = file_name.split('.')[-1]
        raw_datasets = datasets.load_dataset(
            file_type,
            data_files=data_files,
            cache_dir=cache_dir,
            **load_kwargs
        )

    canonical_datasets = {}
    for canonical_split_name, names_in_dataset in split_map.items():
        if names_in_dataset is None:
            continue

        if not isinstance(names_in_dataset, list):
            names_in_dataset = [names_in_dataset]

        component_datasets = []
        for name in names_in_dataset:
            if name in raw_datasets:
                component_datasets.append(raw_datasets[name])
            elif name != canonical_split_name:
                raise ValueError(f'Dataset contains no split "{name}"')
        if len(component_datasets) == 0:
            continue

        canonical_datasets[canonical_split_name] = datasets.concatenate_datasets(
            component_datasets
        )

    if split is not None:
        if split not in canonical_datasets:
            return {}

        canonical_datasets = {split: canonical_datasets[split]}

    max_samples_by_split = {}
    if 'max_train_samples' in dataset_json:
        max_samples_by_split['train'] = int(dataset_json['max_train_samples'])
    if 'max_eval_samples' in dataset_json:
        max_samples_by_split['validation'] = int(dataset_json['max_eval_samples'])

    for key, dataset in canonical_datasets.items():
        max_samples = max_samples_by_split.get(key, None)
        if max_samples is not None:
            dataset = dataset.select(range(max_samples))

        if preprocessor is not None:
            if remove_original_columns:
                remove_columns = dataset.column_names
            else:
                remove_columns = []

            fn_kwargs = {}
            if provide_source_to_preprocessor:
                fn_kwargs['source'] = source

            with training_args.main_process_first(desc='dataset map pre-processing'):
                dataset = dataset.map(
                    preprocessor,
                    batched=True,
                    num_proc=preprocessing_num_workers,
                    remove_columns=remove_columns,
                    load_from_cache_file=not overwrite_cache,
                    desc='Preprocessing dataset',
                    fn_kwargs=fn_kwargs,
                )

            if max_seq_length is not None:
                dataset = dataset.filter(
                    lambda example: len(example['input_ids']) <= max_seq_length
                )

        canonical_datasets[key] = dataset

    return canonical_datasets


def load_source_data(
    source_json,
    training_args,
    **kwargs,
):
    if not isinstance(source_json, list):
        source_json = [source_json]

    split_components = {}
    for dataset_descriptor in source_json:
        dataset_splits = load_single_dataset(
            dataset_descriptor,
            training_args,
            **kwargs,
        )
        for split_name, dataset in dataset_splits.items():
            split_components.setdefault(split_name, []).append(dataset)

    for split_name, component_list in split_components.items():
        if len(component_list) == 1:
            split_components[split_name] = component_list[0]
        else:
            split_components[split_name] = datasets.concatenate_datasets(
                component_list
            )

    return split_components


def load_multisource_dataset(
    multisource_json,
    training_args,
    **kwargs,
):
    multisource_splits = {}
    sfts = {}
    upsampling = {split: {} for split in ['train', 'validation', 'test']}
    for source, source_data in multisource_json.items():
        if 'data' not in source_data:
            raise ValueError(f'Missing "data" field for source "{source}"')
        source_splits = load_source_data(
            source_data['data'],
            training_args,
            source=source,
            **kwargs,
        )
        for split_name, split_data in source_splits.items():
            multisource_splits.setdefault(split_name, {})[source] = split_data
        
        if 'sft' in source_data:
            sfts[source] = SFT(source_data['sft'])

        for split_name, split_upsampling in upsampling.items():
            upsampling_key = f'{split_name}_upsampling'
            if upsampling_key in source_data:
                logger.info(f'Encountered {split_name} upsampling for {source}')
                split_upsampling[source] = int(source_data[upsampling_key])

    for split_name, dataset in multisource_splits.items():
        multisource_splits[split_name] = MultiSourceDataset(
            dataset,
            upsampling=upsampling[split_name]
        )

    return multisource_splits, sfts


def MultiSourcePlugin(_Trainer):

    class _MultiSourceTrainer(_Trainer):
        
        def __init__(
            self,
            *args,
            train_dataset=None,
            eval_dataset=None,
            #data_collator=None,
            source_sfts=None,
            source_sft_apply_abs=False,
            **kwargs
        ):
            self._multisource = (
                (train_dataset is not None and isinstance(train_dataset, MultiSourceDataset)) or
                (eval_dataset is not None and isinstance(eval_dataset, MultiSourceDataset))
            )

            if self._multisource and source_sfts is None:
                raise ValueError('Multi-source datasets provided, but no source SFTs')
            self._source_sfts = source_sfts
            self._source_sft_apply_abs = source_sft_apply_abs
            self._activated_sft = None

            super().__init__(
                *args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                #data_collator=data_collator,
                **kwargs,
            )

        def activate_sft(self, source):
            if source == self._activated_sft:
                return

            if self._activated_sft is not None:
                activated_sft = self._source_sfts[self._activated_sft]
                activated_sft.revert(self.model)

            if source is not None:
                sft_to_activate = self._source_sfts[source]
                sft_to_activate.apply(self.model, with_abs=self._source_sft_apply_abs)
            
            self._activated_sft = source

        def training_step(self, model, inputs):
            if self._source_sfts is not None:
                source = inputs.pop(BATCH_SOURCE_KEY, None)
                if source is None:
                    raise ValueError(f'Batch contained no key "{BATCH_SOURCE_KEY}"')
                if source in self._source_sfts:
                    self.activate_sft(source)

            output = super().training_step(model, inputs)
            self.activate_sft(None)
            return output

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            if self._source_sfts is not None:
                source = inputs.pop(BATCH_SOURCE_KEY, None)
                if source is None:
                    raise ValueError(f'Batch contained no key "{BATCH_SOURCE_KEY}"')
                if source in self._source_sfts:
                    self.activate_sft(source)

            output = super().prediction_step(
                model,
                inputs,
                prediction_loss_only,
                ignore_keys=ignore_keys,
            )
            self.activate_sft(None)
            return output

        def get_train_dataloader(self) -> DataLoader:
            if self.train_dataset is None:
                raise ValueError("Trainer: training requires a train_dataset.")

            train_dataset = self.train_dataset
            if isinstance(train_dataset, MultiSourceDataset):
                loaders = {}
                for source, dataset in train_dataset.datasets.items():
                    self.train_dataset = dataset
                    loaders[source] = super().get_train_dataloader()
                self.train_dataset = train_dataset
                return MultiSourceDataLoader(
                    train_dataset,
                    loaders,
                    sampling_policy='random',
                )
            else:
                return super().get_train_dataloader()

        def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
            if eval_dataset is None and self.eval_dataset is None:
                raise ValueError("Trainer: evaluation requires an eval_dataset.")
            eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

            if isinstance(eval_dataset, MultiSourceDataset):
                return MultiSourceDataLoader(
                    eval_dataset,
                    {
                        source: super(_Trainer, self).get_eval_dataloader(eval_dataset=dataset)
                        for source, dataset in eval_dataset.datasets.items()
                    },
                    sampling_policy='sequential',
                )
            else:
                return super().get_eval_dataloader(eval_dataset)

        def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
            if isinstance(test_dataset, MultiSourceDataset):
                return MultiSourceDataLoader(
                    test_dataset,
                    {
                        source: super(_Trainer, self).get_test_dataloader(dataset)
                        for source, dataset in test_dataset.datasets.items()
                    },
                    sampling_policy='sequential',
                )
            else:
                return super().get_test_dataloader(test_dataset)

    return _MultiSourceTrainer
