There are a number of pre-trained SFTs which can be loaded directly as follows:

```
from sft import SFT

pretrained_sft = SFT(model_identifier)
```

## Language SFTs
Identifiers for language SFTs are of the form `cambridgeltl/{base_model}-lang-sft-{lang_code}-small`, e.g. `cambridgeltl/mbert-lang-sft-en-small`. "Small" SFTs have ~7.6M parameters - we may release larger models in the future. Language SFTs are currently available for the following languages/models:

| Language | Code | bert-base-multilingual-cased (mbert) | xlm-roberta-base (xlmr) |
|----------|------|--------------------------------------|--------------------------|
| Arabic | ar | &check; | &check; |
| Ashaninka | cni | &cross; | &check; |
| Bambara | bm | &check; | &cross; |
| Basque | eu | &check; | &cross; |
| Bribri | bzd | &cross; | &check; |
| Bulgarian | bg | &cross; | &check; |
| Buryat | bxr | &check; | &cross; |
| Cantonese | yue | &check; | &cross; |
| Chinese | zh | &check; | &check; |
| Czech | cs | &check; | &cross; |
| English | en | &check; | &check; |
| Erzya | myv | &check; | &cross; |
| Estonian | et | &check; | &cross; |
| Faroese | fo | &check; | &cross; |
| French | fr | &check; | &check; |
| German | de | &check; | &check; |
| Greek | el | &check; | &check; |
| Guarani | gn | &cross; | &check; |
| Hausa | hau | &check; | &cross; |
| Hindi | hi | &check; | &check; |
| Igbo | ibo | &check; | &cross; |
| Japanese | ja | &check; | &cross; |
| Komi Zyrian | kpv | &check; | &cross; |
| Korean | ko | &check; | &check; |
| Livvi | olo | &check; | &cross; |
| Luganda | lug | &check; | &cross; |
| Luo | luo | &check; | &cross; |
| Maltese | mt | &check; | &cross; |
| Manx | gv | &check; | &cross; |
| Nahuatl | nah | &cross; | &check; |
| Nigerian-Pidgin | pcm | &check; | &cross; |
| Otomi | oto | &cross; | &check; |
| Persian | fa | &check; | &cross; |
| Quechua | quy | &cross; | &check; |
| Raramuri | tar | &cross; | &check; |
| Romanian | ro | &check; | &check; |
| Russian | ru | &check; | &check; |
| Sanskrit | sa | &check; | &cross; |
| Shipibo-Konibo | shp | &cross; | &check; |
| Spanish | es | &check; | &check; |
| Swahili | swa | &check; | &cross; |
| Thai | th | &check; | &check; |
| Turkish | tr | &check; | &check; |
| Upper Sorbian | hsb | &check; | &cross; |
| Urdu | ur | &cross; | &check; |
| Uyghur | ug | &check; | &cross; |
| Vietnamese | vi | &check; | &check; |
| Wixarika | hch | &cross; | &check; |
| Wolof | wol | &check; | &cross; |
| Yoruba | yor | &check; | &cross; |


## Task SFTs
The following single-source (English) task SFTs are available:

* `cambridgeltl/mbert-task-sft-pos`: Universal Dependencies part-of-speech tagging.
* `cambridgeltl/mbert-task-sft-dp`: Universal Dependency parsing.
* `cambridgeltl/mbert-task-sft-masakhaner`: Named Entity Recognition on the restricted tagset `O`, `(B/I)-PER`, `(B/I)-ORG`, `(B/I)-LOC`.
* `cambridgeltl/xlmr-task-sft-nli`: Natural Language Inference.

The following multi-source task SFTs are available:

* `cambridgeltl/mbert-task-sft-pos-ms`: Universal Dependencies part-of-speech tagging (15 diverse source languages).
* `cambridgeltl/mbert-task-sft-dp-ms`: Universal Dependency parsing (15 diverse source languages).
* `cambridgeltl/xlmr-task-sft-nli-ms`: Natural Language Inference, trained on the concatenation of MultiNLI (English) and the test data for all languages in XNLI.
* `cambridgeltl/xlmr-task-sft-squadv1-ms`: SQuADv1-style question answering, trained on SQuADv1 (English) plus the test data from MLQA and XQuAD for all languages in MLQA. The data from XQuAD for languages NOT in MLQA was used for evaluation, achieving the following results (compared to DeepMind's full fine-tuning baselines):

|Base model|Fine-tuning method|Source data|el|ro|ru|th|tr|
|----------|------------------|-----------|--|--|--|--|--|
|mBERT|Full|SQuADv1|62.6/44.9|72.7/59.9|71.3/53.3|42.7/33.5|55.4/40.1|
|XLM-R Large|Full|SQuADv1|79.8/61.7|83.6/69.7|80.1/64.3|74.2/62.8|75.9/59.3|
|XLM-R Base|LT-SFT|SQuADv1 + MLQA + XQuAD(subset)|81.9/65.5|86.3/73.3|81.4/64.6|82.4/75.2|75.2/58.6|
