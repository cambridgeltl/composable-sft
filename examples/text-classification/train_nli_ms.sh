#!/bin/bash

MODEL_DIR=models/xlmr_xnli
mkdir -p $MODEL_DIR

python run_nli.py \
  --model_name_or_path xlm-roberta-base \
  --multisource_data nli_multisource.json \
  --output_dir $MODEL_DIR \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --overwrite_output_dir \
  --full_ft_max_epochs_per_iteration 5 \
  --sparse_ft_max_epochs_per_iteration 5 \
  --save_steps 1000000 \
  --ft_params_num 14155776 \
  --evaluation_strategy steps \
  --eval_steps 2000 \
  --freeze_layer_norm \
  --learning_rate 2e-5 \
  --metric_for_best_model eval_accuracy \
  --load_best_model_at_end \
  --eval_split validation \
  --save_total_limit 2
