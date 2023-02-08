#!/bin/bash
LANG=gn # Guarani
LANG_FT=cambridgeltl/xlmr-lang-sft-${LANG}-small
#TASK_FT=cambridgeltl/xlmr-task-sft-nli  # Single-source task SFT
TASK_FT=cambridgeltl/xlmr-task-sft-nli-ms

RESULTS_DIR=results/AmericasNLI/${LANG}
mkdir -p $RESULTS_DIR

python run_text_classification.py \
  --model_name_or_path xlm-roberta-base \
  --test_file data/AmericasNLI/test/${LANG}.tsv \
  --input_columns premise hypothesis \
  --label_file data/anli_labels.json \
  --output_dir $RESULTS_DIR \
  --lang_ft $LANG_FT \
  --task_ft $TASK_FT \
  --do_eval \
  --eval_split test \
  --eval_metric xnli \
  --per_device_eval_batch_size 8 \
  --overwrite_output_dir
