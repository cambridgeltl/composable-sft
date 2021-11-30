#!/bin/bash
LANG_FT=cambridgeltl/xlmr-lang-sft-en-small
MODEL_DIR=models/xlmr_squad
mkdir -p $MODEL_DIR

python run_qa.py \
  --model_name_or_path xlm-roberta-base \
  --dataset_name squad \
  --output_dir $MODEL_DIR \
  --lang_ft $LANG_FT \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 12\
  --per_device_eval_batch_size 12 \
  --gradient_accumulation_steps 1 \
  --overwrite_output_dir \
  --full_ft_max_epochs_per_iteration 2 \
  --sparse_ft_max_epochs_per_iteration 2 \
  --save_steps 1000000 \
  --ft_params_num 14155776 \
  --evaluation_strategy steps \
  --eval_steps 2000 \
  --freeze_layer_norm \
  --learning_rate 3e-5 \
  --metric_for_best_model eval_f1 \
  --load_best_model_at_end \
  --save_total_limit 2
