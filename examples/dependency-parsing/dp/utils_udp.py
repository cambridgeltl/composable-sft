import copy
import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
#from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import (
    DataCollator,
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.file_utils import PaddingStrategy
from transformers.trainer_utils import EvalLoopOutput

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

UD_HEAD_LABELS = [
    "_",
    "acl",
    "advcl",
    "advmod",
    "amod",
    "appos",
    "aux",
    "case",
    "cc",
    "ccomp",
    "clf",
    "compound",
    "conj",
    "cop",
    "csubj",
    "dep",
    "det",
    "discourse",
    "dislocated",
    "expl",
    "fixed",
    "flat",
    "goeswith",
    "iobj",
    "list",
    "mark",
    "nmod",
    "nsubj",
    "nummod",
    "obj",
    "obl",
    "orphan",
    "parataxis",
    "punct",
    "reparandum",
    "root",
    "vocative",
    "xcomp",
]


@dataclass
class UDTrainingArguments(TrainingArguments):
    """
    Extends TrainingArguments for Universal Dependencies (UD) dependency parsing.
    TrainingArguments is the subset of the arguments we use in our example scripts
    **which relate to the training loop itself**.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    decode_mode: str = field(default="greedy", metadata={"help": "Whether to use mst decoding or greedy decoding"})
    metric_score: Optional[str] = field(
        default=None, metadata={"help": "Metric used to determine best model during training."}
    )


def dataset_preprocessor(tokenizer, label2id, padding):

    def preprocess(examples):
        features = {}
        for idx in range(len(examples['tokens'])):
            invalid_indices = set(i for i, head in enumerate(examples['head'][idx]) if head in ['_', 'None'])
            for col in ['tokens', 'head', 'deprel']:
                examples[col][idx] = [v for i, v in enumerate(examples[col][idx]) if i not in invalid_indices]

            tokens = [tokenizer.tokenize(w) for w in examples['tokens'][idx]]
            word_lengths = [len(w) for w in tokens]

            tokenized_inputs = tokenizer(
                examples['tokens'][idx],
                padding=padding,
                truncation=True,
                is_split_into_words=True,
            )

            tokenized_inputs['labels_arcs'] = [int(x) for x in examples['head'][idx]]
            tokenized_inputs['labels_rels'] = [label2id[x.split(':')[0]] for x in examples['deprel'][idx]]

            # determine start indices of words
            tokenized_inputs['word_starts'] = np.cumsum([1] + word_lengths).tolist()

            if idx < 5:
                logger.info("*** Example ***")
                logger.info(f"tokens: {tokens}")
                for k, v in sorted(list(tokenized_inputs.items())):
                    logger.info(f'{k}: {v}')

            for k, v in tokenized_inputs.items():
                features.setdefault(k, []).append(v)

        return features
    
    return preprocess

@dataclass
class DataCollatorForDependencyParsing:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        seq_len = len(batch['input_ids'][0])
        for k in ['labels_arcs', 'labels_rels', 'word_starts']:
            if k in batch:
                for i, example in enumerate(batch[k]):
                    if self.tokenizer.padding_side == 'right':
                        example += (seq_len - len(example)) * [self.tokenizer.pad_token_id]
                    else:
                        batch[k][i] = example + (seq_len - len(example)) * [self.tokenizer.pad_token_id]

        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        return batch


class Metric(object):
    def add(self, gold, prediction):
        raise NotImplementedError

    def get_metric(self) -> Dict[str, float]:
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    @staticmethod
    def unpack(*tensors: torch.Tensor):
        return (x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in tensors)


class ParsingMetric(Metric):
    """
    based on allennlp.training.metrics.AttachmentScores
    Computes labeled and unlabeled attachment scores for a dependency parse. Note that the input
    to this metric is the sampled predictions, not the distribution itself.
    """

    def __init__(self):
        self._labeled_correct = 0.0
        self._unlabeled_correct = 0.0
        self._total_words = 0.0

    def add(
        self,
        gold_indices: torch.Tensor,
        gold_labels: torch.Tensor,
        predicted_indices: torch.Tensor,
        predicted_labels: torch.Tensor,
    ):
        """
        Parameters
        ----------
        predicted_indices : ``torch.Tensor``, required.
            A tensor of head index predictions of shape (batch_size, timesteps).
        predicted_labels : ``torch.Tensor``, required.
            A tensor of arc label predictions of shape (batch_size, timesteps).
        gold_indices : ``torch.Tensor``, required.
            A tensor of the same shape as ``predicted_indices``.
        gold_labels : ``torch.Tensor``, required.
            A tensor of the same shape as ``predicted_labels``.
        """
        unwrapped = self.unpack(predicted_indices, predicted_labels, gold_indices, gold_labels)
        predicted_indices, predicted_labels, gold_indices, gold_labels = unwrapped

        predicted_indices = predicted_indices.long()
        predicted_labels = predicted_labels.long()
        gold_indices = gold_indices.long()
        gold_labels = gold_labels.long()

        correct_indices = predicted_indices.eq(gold_indices).long()
        correct_labels = predicted_labels.eq(gold_labels).long()
        correct_labels_and_indices = correct_indices * correct_labels

        self._unlabeled_correct += correct_indices.sum().item()
        self._labeled_correct += correct_labels_and_indices.sum().item()
        self._total_words += correct_indices.numel()

    def get_metric(self):
        unlabeled_attachment_score = 0.0
        labeled_attachment_score = 0.0
        if self._total_words > 0.0:
            unlabeled_attachment_score = self._unlabeled_correct / self._total_words
            labeled_attachment_score = self._labeled_correct / self._total_words
        return {
            "uas": unlabeled_attachment_score * 100,
            "las": labeled_attachment_score * 100,
        }

    def reset(self):
        self._labeled_correct = 0.0
        self._unlabeled_correct = 0.0
        self._total_words = 0.0


def DependencyParsingTrainer(_Trainer):
    
    class _DependencyParsingTrainer(_Trainer):
        args: UDTrainingArguments

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def prediction_loop(
            self,
            dataloader: DataLoader,
            description: str,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys=None,
            metric_key_prefix=None,
        ) -> EvalLoopOutput:
            """
                    Prediction/evaluation loop, shared by `evaluate()` and `predict()`.

                    Works both with or without labels.
                    """

            prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only

            model = self.model
            # multi-gpu eval
            if self.args.n_gpu > 1:
                model = torch.nn.DataParallel(model)
            else:
                model = self.model
            # Note: in torch.distributed mode, there's no point in wrapping the model
            # inside a DistributedDataParallel as we'll be under `no_grad` anyways.

            batch_size = dataloader.batch_size
            logger.info("***** Running %s *****", description)
            logger.info("  Num examples = %d", self.num_examples(dataloader))
            logger.info("  Batch size = %s", str(batch_size)) # may be None
            logger.info("  Decode mode = %s", self.args.decode_mode)
            eval_losses: List[float] = []
            model.eval()

            metric = ParsingMetric()

            for inputs in tqdm(dataloader, desc=description):

                #for k, v in inputs.items():
                #    if isinstance(v, torch.Tensor):
                #        inputs[k] = v.to(self.args.device)

                #with torch.no_grad():
                step_eval_loss, (rel_preds, arc_preds), _ = self.prediction_step(model, inputs, False)

                eval_losses += [step_eval_loss.mean().item()]

                mask = inputs["labels_arcs"].ne(self.model.config.pad_token_id)
                predictions_arcs = torch.argmax(arc_preds, dim=-1)[mask]

                labels_arcs = inputs["labels_arcs"][mask]

                predictions_rels, labels_rels = rel_preds[mask], inputs["labels_rels"][mask]
                predictions_rels = predictions_rels[torch.arange(len(labels_arcs)), labels_arcs]
                predictions_rels = torch.argmax(predictions_rels, dim=-1)

                metric.add(labels_arcs, labels_rels, predictions_arcs, predictions_rels)

            results = metric.get_metric()
            results = {
                k if k.startswith(metric_key_prefix) else f'{metric_key_prefix}_{k}': v
                for k, v in results.items()
            }
            results[f"{metric_key_prefix}_loss"] = np.mean(eval_losses)

            # Add predictions_rels to output, even though we are only interested in the metrics
            return EvalLoopOutput(
                predictions=predictions_rels,
                label_ids=None,
                metrics=results,
                num_samples=self.num_examples(dataloader),
            )

    return _DependencyParsingTrainer

