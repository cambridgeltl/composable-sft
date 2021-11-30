#!/bin/bash

MODEL_DIR=models/multisource
mkdir -p $MODEL_DIR

python run_qa.py \
  --model_name_or_path xlm-roberta-base \
  --multisource_data qa_multisource_train_xlmr.json \
  --output_dir $MODEL_DIR \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 12 \
  --per_device_eval_batch_size 12 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy steps \
  --eval_steps 4000 \
  --overwrite_output_dir \
  --full_ft_max_epochs_per_iteration 5 \
  --sparse_ft_max_epochs_per_iteration 5 \
  --save_steps 1000000 \
  --ft_params_num 14155776 \
  --freeze_layer_norm \
  --learning_rate 3e-5 \
  --save_total_limit 2
