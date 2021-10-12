import itertools
import os
import random

import numpy as np
import torch
from tqdm import tqdm

from transformers import logging, Trainer, TrainerCallback

from sft import SFT, SftArguments

logger = logging.get_logger(__name__)


class _RegLossCalculationCallback(TrainerCallback):

    def __init__(self, sft):
        self._sft = sft

    def on_step_begin(self, args, state, control, **kwargs):
        self._sft.calculate_reg_loss = True


class SparseFineTuner(Trainer):
    """ Superclass for Trainers that learn sparse fine-tunings. Keeps track
    of original model parameters so that difference vectors can be calculated
    at the end of training, and which parameters are masked so that gradients
    of fixed parameters can be zeroed.

    Args:
        sft_args: an SftArguments object containing SFT training options.
        maskable_params: a list of parameter names; the model parameters which
            are to be sparsely fine-tuned. Parameters not included in
            maskable_params but have requires_grad=True will be fully
            fine-tuned (this is typically preferable for model heads, for
            instance). If None, all parameters will be sparsely fine-tuned.
        **kwargs: arguments to pass to Trainer constructor.
    """

    def __init__(
        self,
        sft_args=None,
        maskable_params=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        if sft_args is None:
            self.sft_args = SftArguments()
        else:
            self.sft_args = sft_args
        if maskable_params is None:
            self.maskable_params = set(
                n for n, _ in self.model.named_parameters()
            )
        else:
            self.maskable_params = set(maskable_params)

        self._num_params = sum(
            p.data.numel()
            for n, p in self.model.named_parameters()
        )
        self._num_maskable_params = sum(
            p.data.numel()
            for n, p in self.model.named_parameters()
            if n in self.maskable_params
        )

        self._regularized = (
            self.sft_args.full_l1_reg != 0.0 or
            self.sft_args.sparse_l1_reg != 0.0
        )
        # Since the regularization loss is dependent only on the parameter
        # values, we can get away with calculating it only once per full step
        # rather than at every gradient accumulation step. This flag gets set
        # by a _RegLossCalculationCallback at the start of each full step to
        # tell us to do so.
        self.calculate_reg_loss = False
        self._reg_loss = 0.0 # Keeps track of the reg loss for logging purposes.
        if self._regularized:
            # If regularization is in use, the original parameters should be
            # kept on the same device as the tuned parameters for efficiency.
            device = None
            self.add_callback(_RegLossCalculationCallback(self))
        else:
            # Otherwise we can save some GPU RAM by keeping them on the CPU.
            device = 'cpu'
        self._original_params = {
            n: torch.zeros_like(p, device=device).copy_(p)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        self._mask = {
            n: torch.ones_like(p, dtype=torch.bool)
            for n, p in self.model.named_parameters()
            if n in self.maskable_params
        }
        # Whether to apply masking during training.
        self._masking_enabled = True

    def enable_masking(self):
        self._masking_enabled = True

    def disable_masking(self):
        self._masking_enabled = False

    def reset(self):
        for n, p in self.model.named_parameters():
            p.data.copy_(self._original_params[n])

    def freeze(self):
        for _, p in self._mask.items():
            p.data.zero_()

    def sft(self):
        """ Calculates the sparse difference vector between the current
        parameter values and the pre-trained values.

        Returns:
            An SFT containing the differences.
        """
        with torch.no_grad():
            diffs = SFT()
            for n, p in self.model.named_parameters():
                if n in self.maskable_params:
                    delta = p - self._original_params[n].to(p.device)
                    diffs.add_param(n, delta, diff=True)
                elif p.requires_grad:
                    # p is to be stored in full rather than as a difference.
                    # Typically this happens when p belongs to the model head.
                    diffs.add_param(n, p, diff=False)
            return diffs

    def set_training_len(self, min_steps, max_steps, max_epochs):
        if max_steps is None and max_epochs is None:
            raise ValueError('Length of sft training not specified.')
        if min_steps is not None and max_steps is not None and min_steps > max_steps:
            raise ValueError('min_steps cannot be > max_steps')

        if max_epochs is None:
            self.args.max_steps = max_steps
        else:
            n_steps = max_epochs * len(self.train_dataset) // (
                self.args.per_device_train_batch_size *
                self.args.gradient_accumulation_steps
            )
            logger.info(f'{max_epochs} epochs = {n_steps} steps')
        
            if max_steps is None or n_steps < max_steps:
                if min_steps is not None and n_steps < min_steps:
                    self.args.max_steps = min_steps
                else:
                    self.args.num_train_epochs = max_epochs
                    self.args.max_steps = -1
            else:
                self.args.max_steps = max_steps

    def training_step(self, model, inputs):
        loss = super().training_step(model, inputs)

        l1_reg = (
            self.sft_args.sparse_l1_reg
            if self._masking_enabled
            else self.sft_args.full_l1_reg
        )
        if l1_reg != 0.0 and self.calculate_reg_loss:
            # Since we only calculate reg loss once per full step.
            l1_reg *= self.args.gradient_accumulation_steps
            l1_dists = []
            for n, p in self.model.named_parameters():
                if (
                    p.requires_grad and
                    (n in self.maskable_params or
                        not self.sft_args.apply_reg_to_sparse_only)
                ):
                    l1_dists.append(
                        torch.sum(torch.abs(p - self._original_params[n]))
                    )
            reg_loss = l1_reg * torch.sum(torch.stack(l1_dists)) / self._num_params
            reg_loss.backward()
            self._reg_loss += float(reg_loss)
            self.calculate_reg_loss = False

        if self._masking_enabled:
            for n, p in model.named_parameters():
                if n in self.maskable_params and p.grad is not None:
                    p.grad *= self._mask[n]

        return loss

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            logs: Dict[str, float] = {}
            tr_loss_scalar = tr_loss.item()
            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            if self._reg_loss != 0.0:
                logs['l1_reg_loss'] = round(self._reg_loss / (self.state.global_step - self._globalstep_last_logged), 4)
                self._reg_loss = 0.0

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, epoch, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

