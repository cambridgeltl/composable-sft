#!/bin/bash
SOURCE_LANG=id
LANG_FT=cambridgeltl/xlmr-lang-sft-${SOURCE_LANG}-small

MODEL_DIR=models/sa/smsa-trimmed
mkdir -p $MODEL_DIR

python run_text_classification.py \
  --model_name_or_path xlm-roberta-base \
  --lang_ft $LANG_FT \
  --train_file data/smsa-trimmed/train.tsv \
  --validation_file data/smsa-trimmed/validation.tsv \
  --output_dir $MODEL_DIR \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --overwrite_output_dir \
  --full_ft_max_epochs_per_iteration 5 \
  --sparse_ft_max_epochs_per_iteration 5 \
  --ft_params_num 14155776 \
  --freeze_layer_norm \
  --learning_rate 2e-5 \
  --save_strategy epoch \
  --evaluation_strategy epoch \
  --eval_metric f1 \
  --metric_for_best_model eval_f1 \
  --load_best_model_at_end \
  --eval_split validation \
  --save_total_limit 2
