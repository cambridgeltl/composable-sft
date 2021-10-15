There are a number of pre-trained SFTs which can be loaded directly as follows:

```
from sft import SFT

pretrained_sft = SFT(model_identifier)
```

## Language SFTs
Identifiers for language SFTs are of the form `cambridgeltl/{base_model}-lang-sft-{lang_code}-small`, e.g. `cambridgeltl/mbert-lang-sft-en-small`. "Small" SFTs have ~7.6M parameters - we may release larger models in the future. Language SFTs are currently available for the following languages/models:

`bert-base-multilingual-cased (mbert)`:
| Language | Code |
|----------|------|
| Arabic | ar |
| Bambara | bm |
| Buryat | bxr |
| Cantonese | yue |
| Chinese | zh |
| English | en |
| Erzya | myv |
| Faroese | fo |
| Hausa | hau |
| Igbo | ibo |
| Livvi | olo |
| Luganda | lug |
| Luo | luo |
| Komi Zyrian | kpv |
| Maltese | mt |
| Manx | gv |
| Nigerian-Pidgin | pcm |
| Sanskrit | sa |
| Swahili | swa |
| Upper Sorbian | hsb |
| Uyghur | ug |
| Wolof | wol |
| Yoruba | yor |


`xlm-roberta-base (xlmr)`
| Language | Code |
|----------|------|
| Ashaninka | cni |
| Aymara | aym |
| Bribri | bzd |
| Guarani | gn |
| Nahuatl | nah |
| Otomi | oto |
| Quechua | quy |
| Raramuri | tar |
| Shipibo-Konibo | shp |
| Wixarika | hch |


## Task SFTs
The following task SFTs are available:

* `cambridgeltl/mbert-task-sft-pos`: Universal Dependencies part-of-speech tagging.
* `cambridgeltl/mbert-task-sft-dp`: Universal Dependency parsing.
* `cambridgeltl/mbert-task-sft-masakhaner`: Named Entity Recognition on the restricted tagset `O`, `(B/I)-PER`, `(B/I)-ORG`, `(B/I)-LOC`.
* `cambridgeltl/xlmr-task-sft-nli`: Natural Language Inference.

