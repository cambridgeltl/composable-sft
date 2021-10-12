#!/bin/bash
LANG=hau # Hausa
LANG_FT=cambridgeltl/mbert-lang-sft-${LANG}-small
TASK_FT=cambridgeltl/mbert-task-sft-masakhaner

python run_token_classification.py \
  --model_name_or_path bert-base-multilingual-cased \
  --dataset_name ner_dataset.py \
  --dataset_config_name $LANG \
  --output_dir results/ner/${LANG} \
  --lang_ft $LANG_FT \
  --task_ft $TASK_FT \
  --do_eval \
  --label_column_name ner_tags \
  --per_device_eval_batch_size 8 \
  --task_name ner \
  --overwrite_output_dir \
  --eval_split test
