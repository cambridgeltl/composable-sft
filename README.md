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

## Example Scripts
Example scripts are provided in `examples/` to show how to train SFTs using LT-SFT and evaluate them.


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
