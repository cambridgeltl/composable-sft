import collections
import logging
import math
import os

import numpy as np
import torch
from tqdm import tqdm

from transformers import get_scheduler

from .trainer import SparseFineTuner

logger = logging.getLogger(__name__)


def EliminationSparseFineTuner(_Trainer):

    _SparseFineTuner = SparseFineTuner(_Trainer)

    class _EliminationSparseFineTuner(_SparseFineTuner):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            logger.setLevel(self.args.get_process_log_level())
            if self.sft_args.ft_params_num is None:
                self.n_tunable_params = int(
                    self.sft_args.ft_params_proportion * self._num_maskable_params
                )
            else:
                self.n_tunable_params = self.sft_args.ft_params_num

        def reset_least_changed_params(self, k, reset_all=True):
            with torch.no_grad():
                diffs = []
                for n, p in tqdm(
                    list(self.model.named_parameters()),
                    desc='Finding masking threshold',
                    disable=self.args.local_rank > 0 or self.args.disable_tqdm,
                ):
                    if n in self.maskable_params:
                        delta = p - self._original_params[n].to(p.device)
                        delta = delta.view(-1).tolist()
                        for d in delta:
                            diffs.append(abs(d))
                
                if k > len(diffs):
                    raise ValueError(
                        'Was requested to freeze all but {k} params, but only '
                        '{len(diffs)} are frozen.'
                    )
                diffs = np.partition(diffs, k - 1)
                thresh = diffs[k - 1]
                logger.info(f'Masking threshold = {thresh}')
                
                n_reset = 0
                for n, p in tqdm(
                    list(self.model.named_parameters()),
                    desc='Updating masks',
                    disable=self.args.local_rank > 0 or self.args.disable_tqdm,
                ):
                    if n in self.maskable_params:
                        abs_delta = (p - self._original_params[n].to(p.device)).abs()
                        #self._mask[n].copy_(abs_delta > thresh)
                        self._mask[n] = abs_delta > thresh
                        n_reset += int((~self._mask[n]).sum())

                n_trainable = 0
                for n, p in tqdm(
                    list(self.model.named_parameters()),
                    desc='Resetting parameters',
                    disable=self.args.local_rank > 0 or self.args.disable_tqdm,
                ):
                    if not p.requires_grad:
                        continue
                    if n in self.maskable_params:
                        p.data = (
                            (~self._mask[n]) * self._original_params[n].to(p.device) +
                            self._mask[n] * p
                        )
                        n_trainable += int(torch.sum(self._mask[n]))
                    
                logger.info(
                    f'Reset {n_reset} params; '
                    f'{self._num_maskable_params - n_reset} not reset; '
                    f'{n_trainable} trainable.'
                )

                return n_reset

        def get_train_len(self, train_dataloader):
            train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)
            if train_dataset_is_sized:
                if self.args.max_steps > 0:
                    max_steps = self.args.max_steps
                else:
                    num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
                    num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
                    max_steps = math.ceil(self.args.num_train_epochs * num_update_steps_per_epoch)
            else:
                max_steps = self.args.max_steps
            return max_steps

        def train(self, *args, **kwargs):
            self.disable_masking()
            param_count = 0
            
            train_dataloader = self.get_train_dataloader()
            self.set_training_len(
                train_dataloader,
                self.sft_args.full_ft_min_steps_per_iteration,
                self.sft_args.full_ft_max_steps_per_iteration,
                self.sft_args.full_ft_max_epochs_per_iteration,
            )
            full_steps = self.sft_args.n_ft_iterations * self.get_train_len(train_dataloader)
            self.set_training_len(
                train_dataloader,
                self.sft_args.sparse_ft_min_steps_per_iteration,
                self.sft_args.sparse_ft_max_steps_per_iteration,
                self.sft_args.sparse_ft_max_epochs_per_iteration,
            )
            sparse_steps = self.get_train_len(train_dataloader)
            total_steps = full_steps + sparse_steps
            logger.info(
                f'Performing {full_steps} full FT steps, {sparse_steps} '
                f'sparse FT steps, {total_steps} total steps'
            )
            warmup_steps = self.args.warmup_steps
            if warmup_steps == 0:
                warmup_steps = int(self.args.warmup_ratio * (full_steps + sparse_steps))
            self.create_optimizer()
            self.lr_scheduler = get_scheduler(
                'linear',
                optimizer=self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=2 * full_steps,
            )
            #self.optimizer = None
            #self.lr_scheduler = None
            #self.args.lr_scheduler_type = 'constant'

            for it in range(self.sft_args.n_ft_iterations):
                logger.info(f'Fine-tuning iteration {it+1}')
                #with torch.no_grad():
                #    previous_params = {
                #        n: torch.zeros_like(p, device='cpu').copy_(p)
                #        for n, p in self.model.named_parameters()
                #    }

                self.set_training_len(
                    train_dataloader,
                    self.sft_args.full_ft_min_steps_per_iteration,
                    self.sft_args.full_ft_max_steps_per_iteration,
                    self.sft_args.full_ft_max_epochs_per_iteration,
                )
                super().train(*args, **kwargs)

                params_remaining = self._num_maskable_params - param_count
                keep_rate = math.pow(
                    self.n_tunable_params / params_remaining,
                    1.0 / (self.sft_args.n_ft_iterations - it)
                )
                logger.info(
                    f'Resetting {100.0 * (1.0 - keep_rate) :.3f}% '
                    f'of remaining params'
                )
                to_freeze = int((1.0 - keep_rate) * params_remaining)
                param_count = self.reset_least_changed_params(
                    param_count + to_freeze
                )
                
                #with torch.no_grad():
                #    for n, p in self.model.named_parameters():
                #        p.copy_(previous_params[n])

            self.optimizer = None
            self.lr_scheduler = None
            self.args.lr_scheduler_type = 'linear'
            self.args.learning_rate /= 2.0
            self.args.warmup_steps = 0
            self.args.warmup_ratio = 0.0
            self.enable_masking()
            self.set_training_len(
                train_dataloader,
                self.sft_args.sparse_ft_min_steps_per_iteration,
                self.sft_args.sparse_ft_max_steps_per_iteration,
                self.sft_args.sparse_ft_max_epochs_per_iteration,
            )
            result = super().train(*args, **kwargs)
            
            return result

    return _EliminationSparseFineTuner
