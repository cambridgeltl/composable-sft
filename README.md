This is a library for training and applying sparse fine-tunings with PyTorch and `transformers`.

# Installation

First, install Python 3.9 and PyTorch >= 1.9, e.g. using conda:
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


# Using pre-trained SFTs

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


# Example Scripts
Example scripts are provided in `examples/` to show how to train SFTs using LT-SFT and evaluate them.
