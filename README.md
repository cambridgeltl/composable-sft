# Installation

First, install Python 3.9 and PyTorch >= 1.9, e.g. using conda:
`conda create -n sft python=3.9
conda activate sft
conda install pytorch cudatoolkit=11.1 -c pytorch -c conda-forge`

Then download and install composable-sft:
`git clone https://github.com/cambridgeltl/composable-sft.git
cd composable-sft
pip install -e .`


# Using pre-trained SFTs

Pre-trained SFTs can be downloaded directly and applied to models as follows:
`from sft import SFT

language_sft = SFT('cambridgeltl/mbert-lang-sft-bxr-small') # SFT for Buryat
task_sft = SFT('cambridgeltl/mbert-task-sft-pos') # SFT for POS tagging

# Apply SFTs to pre-trained mBERT TokenClassification model
language_sft.apply(model)
task_sft.apply(model)`

