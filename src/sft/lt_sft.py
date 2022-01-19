import logging
import os

import numpy as np
import torch
from tqdm import tqdm

from .trainer import SparseFineTuner

logger = logging.getLogger(__name__)


def LotteryTicketSparseFineTuner(_Trainer):

    _SparseFineTuner = SparseFineTuner(_Trainer)

    class _LotteryTicketSparseFineTuner(_SparseFineTuner):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.sft_args.ft_params_num is None:
                self.n_tunable_params = int(
                    self.sft_args.ft_params_proportion * self._num_maskable_params
                )
            else:
                self.n_tunable_params = self.sft_args.ft_params_num

        def unfreeze_k_most_changed_params(self, k):
            with torch.no_grad():
                diffs = []
                for n, p in tqdm(
                    list(self.model.named_parameters()),
                    desc='Finding masking threshold'
                ):
                    if n in self.maskable_params:
                        delta = p - self._original_params[n].to(p.device)
                        delta = delta.view(-1).tolist()
                        mask = self._mask[n].view(-1).tolist()
                        for d, m in zip(delta, mask):
                            if not m:
                                diffs.append(abs(d))
                
                if k > len(diffs):
                    raise ValueError(
                        'Was requested to unfreeze {k} params, but only '
                        '{len(diffs)} are frozen.'
                    )
                diffs = np.partition(diffs, len(diffs) - k)
                thresh = diffs[len(diffs) - k]
                logger.info(f'Masking threshold = {thresh}')
                
                n_masked = 0
                for n, p in tqdm(
                    list(self.model.named_parameters()),
                    desc='Updating masks'
                ):
                    if n in self.maskable_params:
                        abs_delta = (p - self._original_params[n].to(p.device)).abs()
                        to_mask = (abs_delta >= thresh) & (~self._mask[n])
                        self._mask[n] = to_mask | self._mask[n]
                        n_masked += to_mask.sum()

                logger.info(f'Masked {n_masked} params')

        def train(self, *args, **kwargs):
            self.freeze()
            result = None
            train_dataloader = self.get_train_dataloader()
            
            for it in range(self.sft_args.n_ft_iterations):
                logger.info(f'Fine-tuning iteration {it+1}')
                with torch.no_grad():
                    previous_params = {
                        n: torch.zeros_like(p, device='cpu').copy_(p)
                        for n, p in self.model.named_parameters()
                    }

                self.disable_masking()
                self.optimizer = None
                self.lr_scheduler = None
                self.set_training_len(
                    train_dataloader,
                    self.sft_args.full_ft_min_steps_per_iteration,
                    self.sft_args.full_ft_max_steps_per_iteration,
                    self.sft_args.full_ft_max_epochs_per_iteration,
                )
                super().train(*args, **kwargs)

                self.unfreeze_k_most_changed_params(
                    self.n_tunable_params // self.sft_args.n_ft_iterations
                )
                
                with torch.no_grad():
                    for n, p in self.model.named_parameters():
                        p.copy_(previous_params[n])

                self.enable_masking()
                self.optimizer = None
                self.lr_scheduler = None
                self.set_training_len(
                    train_dataloader,
                    self.sft_args.sparse_ft_min_steps_per_iteration,
                    self.sft_args.sparse_ft_max_steps_per_iteration,
                    self.sft_args.sparse_ft_max_epochs_per_iteration,
                )
                result = super().train(*args, **kwargs)
            
            return result

    return _LotteryTicketSparseFineTuner
