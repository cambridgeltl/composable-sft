#!/bin/bash

MODEL_DIR=models/multisource

python run_dp.py \
  --model_name_or_path bert-base-multilingual-cased \
  --multisource_data dp_multisource.json \
  --output_dir $MODEL_DIR \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --max_seq_length 256 \
  --overwrite_output_dir \
  --full_ft_max_epochs_per_iteration 3 \
  --sparse_ft_max_epochs_per_iteration 10 \
  --save_steps 1000000 \
  --ft_params_num 14155776 \
  --evaluation_strategy steps \
  --eval_steps 1000 \
  --freeze_layer_norm \
  --learning_rate 5e-5 \
  --metric_for_best_model eval_las \
  --load_best_model_at_end \
  --eval_split validation \
  --save_total_limit 2
