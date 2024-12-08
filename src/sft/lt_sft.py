import logging
import os
import gc

import numpy as np
import torch
from tqdm import tqdm

from .trainer import SparseFineTuner

logger = logging.getLogger(__name__)


def LotteryTicketSparseFineTuner(_Trainer):

    _SparseFineTuner = SparseFineTuner(_Trainer)

    class _LotteryTicketSparseFineTuner(_SparseFineTuner):

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            logger.setLevel(self.args.get_process_log_level())
            if self.sft_args.ft_params_num is None:
                self.n_tunable_params = int(
                    self.sft_args.ft_params_proportion * self._num_maskable_params
                )
            else:
                self.n_tunable_params = self.sft_args.ft_params_num

        def unfreeze_k_most_changed_params(self, k):
            print(f"{k} params will be unfrozen")

            with torch.no_grad():
                diffs = []
                for n, p in tqdm(
                        list(self.model.named_parameters()),
                        desc='Finding masking threshold',
                        disable=self.args.local_rank > 0 or self.args.disable_tqdm,
                ):
                    p.grad = None  # save some memory to use for the diff calculation
                    if n in self.maskable_params:
                        delta = p - self._original_params[n].to(p.device)
                        delta = delta.view(-1)
                        self._mask[n] = self._mask[n].to(p.device)
                        valid_indices = (~self._mask[n]).view(-1)
                        valid_deltas = delta[valid_indices]
                        abs_deltas = torch.abs(valid_deltas)
                        # Do not convert tensor to numpy or list.
                        # The tensors are already allocated in GPU and conversion increases RAM usage.
                        # We can take top-K most changed numbers via tensor calculation.
                        diffs.append(abs_deltas) 

                # Concatenating all the tensors to one device simultaneously requires numerous memories and operations.
                # Instead inspired by map-reduce, continuously looping the diffs: the list (size of N) of tensors,
                # We concatenate the tensors, size of L, into slightly larger than `k` and reduce the size of the list, `diffs`, to less than N.
                # Continue this process until the list size, `diff` becomes 1.
                print("reducing diffs: ", end="")
                while sum([len(diff) for diff in diffs]) > k:
                    print(".", end="")
                    new_diffs = []
                    tmp_diffs = []

                    # Concatenate L tensors to one tensor, that the single tensor has a size slightly larger than `K`.
                    # Then the concatenated tensor will be truncated by `torch.topk` operation. 
                    for diff in diffs:
                        if sum([d.shape[0] for d in tmp_diffs]) > k:
                            lowest_rank = min([
                                diff.device.index for diff in tmp_diffs
                            ])
                            tmp_diffs = [
                                diff.to(lowest_rank) for diff in tmp_diffs
                            ]

                            tmp_diffs_topk = torch.topk(
                                torch.cat(tmp_diffs),
                                k,
                                largest=True,
                            )
                            new_diffs.append(tmp_diffs_topk.values)
                            tmp_diffs = []

                        tmp_diffs.append(diff)

                    # If there is a remaining process for `tmp_diffs`, it is processed under this if block.
                    if len(tmp_diffs) > 0:
                        if sum([d.shape[0] for d in tmp_diffs]) > k:
                            lowest_rank = min([
                                diff.device.index for diff in tmp_diffs
                            ])
                            tmp_diffs = [
                                diff.to(lowest_rank) for diff in tmp_diffs
                            ]
                            tmp_diffs_topk = torch.topk(
                                torch.cat(tmp_diffs),
                                k,
                                largest=True,
                            )
                        else:
                            tmp_diffs_topk = torch.cat(tmp_diffs)
                        new_diffs.append(tmp_diffs_topk.values)
                        tmp_diffs = []

                    diffs = new_diffs

                # Now, diffs has a much smaller size of tensors.
                # Still the numbers are allocated on GPU tensors, we use `torch.topk` instead of `np.partition`.
                diffs = [
                    diff.to(lowest_rank) for diff in diffs
                ]
                diffs = torch.cat(diffs)
                thresh = torch.topk(diffs, k, largest=True).values[-1].item()

                print(f'Masking threshold = {thresh}')

                n_masked = 0
                for n, p in tqdm(
                    list(self.model.named_parameters()),
                    desc='Updating masks',
                    disable=self.args.local_rank > 0 or self.args.disable_tqdm,
                ):
                    if n in self.maskable_params:
                        abs_delta = (p - self._original_params[n].to(p.device)).abs()
                        to_mask = (abs_delta >= thresh) & (~self._mask[n])
                        self._mask[n] = to_mask | self._mask[n]
                        n_masked += to_mask.sum()

                logger.info(f'Masked {n_masked} params')
            
            del diffs, valid_indices, valid_deltas, abs_deltas, delta, to_mask, abs_delta
            gc.collect()
            torch.cuda.empty_cache()

        def train(self, **kwargs):
            self.freeze()
            result = None
            
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
                    self.sft_args.full_ft_min_steps_per_iteration,
                    self.sft_args.full_ft_max_steps_per_iteration,
                    self.sft_args.full_ft_max_epochs_per_iteration,
                )
                super().train(**kwargs)

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
                    self.sft_args.sparse_ft_min_steps_per_iteration,
                    self.sft_args.sparse_ft_max_steps_per_iteration,
                    self.sft_args.sparse_ft_max_epochs_per_iteration,
                )
                result = super().train(**kwargs)
            
            return result

    return _LotteryTicketSparseFineTuner
