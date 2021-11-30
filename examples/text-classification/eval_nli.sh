#!/bin/bash
LANG=gn # Guarani
LANG_FT=cambridgeltl/xlmr-lang-sft-${LANG}-small
#TASK_FT=cambridgeltl/xlmr-task-sft-nli  # Single-source task SFT
TASK_FT=cambridgeltl/xlmr-task-sft-nli-ms

python run_nli.py \
  --model_name_or_path xlm-roberta-base \
  --validation_file data/AmericasNLI/test/${LANG}.tsv \
  --label_file data/labels.txt \
  --output_dir results/AmericasNLI/${LANG} \
  --lang_ft $LANG_FT \
  --task_ft $TASK_FT \
  --do_eval \
  --per_device_eval_batch_size 8 \
  --overwrite_output_dir
