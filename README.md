This is a library for training and applying sparse fine-tunings with `torch` and `transformers`. Please refer to our paper [Composable Sparse Fine-Tuning for Cross Lingual Transfer](https://arxiv.org/abs/2110.07560) for background.

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
Examples of SFT training and evaluation are provided in `examples/`.


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
