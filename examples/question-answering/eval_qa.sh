#!/bin/bash
LANG=ro
TASK_FT=cambridgeltl/xlmr-task-sft-squadv1-ms
LANG_FT=cambridgeltl/xlmr-lang-sft-${LANG}-small

python run_qa.py \
  --model_name_or_path xlm-roberta-base \
  --dataset_name xquad \
  --dataset_config_name xquad.${LANG} \
  --output_dir results/$LANG \
  --lang_ft $LANG_FT \
  --task_ft $TASK_FT \
  --do_eval \
  --per_device_eval_batch_size 8 \
  --overwrite_output_dir
