#!/bin/bash
LANG=bm

python run_mlm.py \
  --model_name_or_path bert-base-multilingual-cased \
  --train_file corpora/${LANG}.txt \
  --output_dir models/${LANG} \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --max_seq_length 256 \
  --save_steps 10000000 \
  --overwrite_output_dir \
  --ft_params_num 7667712 \
  --freeze_layer_norm \
  --freeze_decoder \
  --full_l1_reg 0.1 \
  --sparse_l1_reg 0.1 \
  --learning_rate 5e-5 \
  --full_ft_min_steps_per_iteration 10000 \
  --sparse_ft_min_steps_per_iteration 10000 \
  --full_ft_max_steps_per_iteration 100000 \
  --sparse_ft_max_steps_per_iteration 100000 \
  --full_ft_max_epochs_per_iteration 100 \
  --sparse_ft_max_epochs_per_iteration 100 \
  --evaluation_strategy steps \
  --eval_steps 1000 \
  --validation_split_percentage 5 \
  --load_best_model_at_end \
  --save_total_limit 2
