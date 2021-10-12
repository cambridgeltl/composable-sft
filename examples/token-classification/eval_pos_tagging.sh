#!/bin/bash
LANG=bxr # Buryat
LANG_FT=cambridgeltl/mbert-lang-sft-${LANG}-small
TASK_FT=cambridgeltl/mbert-task-sft-pos

python run_token_classification.py \
  --model_name_or_path bert-base-multilingual-cased \
  --dataset_name universal_dependencies \
  --dataset_config_name bxr_bdt \
  --output_dir results/pos-tagging/${LANG} \
  --lang_ft $LANG_FT \
  --task_ft $TASK_FT \
  --do_eval \
  --label_column_name upos \
  --per_device_eval_batch_size 8 \
  --task_name pos \
  --overwrite_output_dir \
  --eval_split test
