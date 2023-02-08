### [AmericasNLI](https://aclanthology.org/2022.acl-long.435/)

Before using the example scripts, you should download the AmericasNLI test data:
```
cd data
git clone https://github.com/nala-cub/AmericasNLI.git
```

Then you can use the `train_nli.sh` and `train_nli_ms.sh` scripts to train single- and multi-source task SFTs for NLI, and `eval_nli.sh` to evaluate on the AmericasNLI test data.


### [NusaX sentiment analysis](https://arxiv.org/abs/2205.15960)

To create a version of the [SMSA dataset](https://arxiv.org/abs/2009.05720) with examples from the test set of NusaX-senti removed, run
```
# first run "pip install editdistance" if you don't already have it installed
python map_nusa.py
```

Then you can use `train_sa.sh` and `train_sa_ms.sh` to train single- and multi-source task SFTs for NusaX, and `eval_sa.sh` to evaluate. Single-source training uses SMSA (trimmed as above to avoid information leaks) as source, multi-source training additionally uses the NusaX training data for languages other than Indonesian.
