#!/bin/bash
SOURCE_LANG=en
LANG_FT=cambridgeltl/xlmr-lang-sft-${SOURCE_LANG}-small

MODEL_DIR=models/nli/${SOURCE_LANG}
mkdir -p $MODEL_DIR

python run_text_classification.py \
  --model_name_or_path xlm-roberta-base \
  --dataset_name multi_nli \
  --dataset_config_name $SOURCE_LANG \
  --input_columns premise hypothesis \
  --output_dir $MODEL_DIR \
  --lang_ft $LANG_FT \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --overwrite_output_dir \
  --full_ft_max_epochs_per_iteration 5 \
  --sparse_ft_max_epochs_per_iteration 5 \
  --save_steps 1000000 \
  --ft_params_num 14155776 \
  --freeze_layer_norm \
  --evaluation_strategy steps \
  --eval_steps 625 \
  --learning_rate 2e-5 \
  --eval_split validation \
  --eval_metric xnli \
  --metric_for_best_model eval_accuracy \
  --load_best_model_at_end \
  --validation_split validation_matched \
  --save_total_limit 2
