This is a library for training and applying sparse fine-tunings with `torch` and `transformers`. Please refer to our paper [Composable Sparse Fine-Tuning for Cross Lingual Transfer](https://arxiv.org/abs/2110.07560) for background.


## News

### 2021/11/30
New utilities have been added for [multi-source training](#multi-source-training) (see [Ansell et al. (2021)](https://aclanthology.org/2021.findings-emnlp.410/)) and multi-source task SFTs have been released for some tasks. SFTs and examples scripts are now also available for question answering, and we have released language SFTs for many new languages (see [MODELS](MODELS.md)). We recommend the use of multi-source task SFTs where available, as they are substantially better than the single-source SFTs for most languages.


## Installation

First, install Python 3.9 and PyTorch >= 1.9 (earlier versions may work but haven't been tested), e.g. using conda:
```
conda create -n sft python=3.9
conda activate sft
conda install pytorch cudatoolkit=11.1 -c pytorch -c conda-forge
```

Then download and install composable-sft:
```
git clone https://github.com/cambridgeltl/composable-sft.git
cd composable-sft
pip install -e .
```


## Using pre-trained SFTs

Pre-trained SFTs can be downloaded directly and applied to models as follows:
```
from transformers import AutoConfig, AutoModelForTokenClassification
from sft import SFT

config = AutoConfig.from_pretrained(
    'bert-base-multilingual-cased',
    num_labels=17,
)

model = AutoModelForTokenClassification.from_pretrained(
    'bert-base-multilingual-cased',
    config=config,
)

language_sft = SFT('cambridgeltl/mbert-lang-sft-bxr-small') # SFT for Buryat
task_sft = SFT('cambridgeltl/mbert-task-sft-pos') # SFT for POS tagging

# Apply SFTs to pre-trained mBERT TokenClassification model
language_sft.apply(model)
task_sft.apply(model)
```

For a full list of pre-trained SFTs available, see [MODELS](MODELS.md)


## Training SFTs

`LotteryTicketSparseFineTuner` is a sub-class of the `transformers Trainer` class which performs Lottery Ticket Sparse Fine-Tuning. Its constructor takes the following arguments in addition to those of `Trainer`:
* `sft_args`: an `SftArguments` object which holds hyperparameters relating to SFT training (c.f. `transformers TrainingArguments`).
* `maskable_params`: a list of model parameter tensors which are eligible for sparse fine-tuning. Parameters of the classification head should be excluded from this list because these should typically be fully fine-tuned. E.g.
```
maskable_params = [
    n for n, p in model.named_parameters()
    if n.startswith(model.base_model_prefix) and p.requires_grad
]
```

The following command-line params processed by `SftArguments` may be useful:
* `ft_params_num`/`ft_params_proportion` - controls the number/proportion of the maskable params that will be fine-tuned.
* `full_ft_max_steps_per_iteration`/`full_ft_max_epochs_per_iteration` - controls the maximum number of steps/epochs in the first phase of LT-SFT. Both can be set.
* `sparse_ft_max_steps_per_iteration`/`sparse_ft_max_epochs_per_iteration` - controls the maximum number of steps/epochs in the second phase of LT-SFT. Both can be set.
* `full_ft_min_steps_per_iteration`/`sparse_ft_min_steps_per_iteration` - controls the minimum number of steps in the first/second phase of LT-SFT. Takes effect if a max number of epochs is set which amounts to a lesser number of steps.


## Example Scripts
Examples of SFT training and evaluation are provided in [`examples/`](examples).


## Multi-source Training
Multi-source training is where a task SFT is trained on data from several languages. We provide support for multi-source training with `MultiSourceDataset` and `MultiSourcePlugin`.

`MultiSourceDataset` combines data from several sources (e.g. different languages) into a single Dataset. Its constructor takes a dict mapping source names to Datasets, e.g.:
```
from sft import MultiSourceDataset

train_dataset = MultiSourceDataset({
    'en': english_train_dataset,
    'ja': japanese_train_dataset,
})

eval_dataset = MultiSourceDataset({
    'en': english_eval_dataset,
    'ja': japanese_eval_dataset,
})
```
where `english_dataset` and `japanese_dataset` are `torch.utils.data.Dataset`s.

`MultiSourcePlugin` can be applied to a `transformers` `Trainer` (or subclass thereof, such as `LotteryTicketSparseFineTuner`) to allow it to be used in conjunction with `MultiSourceDataset`s:
```
from sft import SFT, LotteryTicketSparseFineTuner, MultiSourcePlugin

english_sft = SFT('cambridgeltl/mbert-lang-sft-en-small')
japanese_sft = SFT('cambridgeltl/mbert-lang-sft-ja-small')
source_sfts = {
    'en': english_sft,
    'ja': japanese_sft,
}

trainer_cls = LotteryTicketSparseFineTuner
trainer_cls = MultiSourcePlugin(trainer_cls)
trainer = trainer_cls(
    ..., # standard LotteryTicketSparseFineTuner parameters
    train_dataset=train_dataset, # instance of MultiSourceDataset
    eval_dataset=eval_dataset, # instance of MultiSourceDataset
    source_sfts=source_sfts,
)
```
Note the use of the optional argument `source_sfts`, a dict of source names to SFTs. If provided, the trainer will apply the SFT corresponding to the source language for each batch (note that each batch will consist of examples from only one source).

See the examples for further demonstration of multi-source training.


## Citation
If you use this software, please cite the following paper:
```
@misc{ansell2021composable,
      title={Composable Sparse Fine-Tuning for Cross-Lingual Transfer},
      author={Alan Ansell and Edoardo Maria Ponti and Anna Korhonen and Ivan Vuli\'{c}},
      year={2021},
      eprint={2110.07560},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
