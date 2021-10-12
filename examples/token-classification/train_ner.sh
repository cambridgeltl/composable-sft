#!/bin/bash
SOURCE_LANG=en
LANG_FT=cambridgeltl/mbert-lang-sft-${SOURCE_LANG}-small

python run_token_classification.py \
  --model_name_or_path bert-base-multilingual-cased \
  --dataset_name ner_dataset.py \
  --dataset_config_name $SOURCE_LANG \
  --output_dir models/ner/${SOURCE_LANG} \
  --lang_ft $LANG_FT \
  --do_train \
  --do_eval \
  --label_column_name ner_tags \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --task_name ner \
  --overwrite_output_dir \
  --full_ft_max_epochs_per_iteration 3 \
  --sparse_ft_max_epochs_per_iteration 10 \
  --save_steps 1000000 \
  --ft_params_num 14155776 \
  --evaluation_strategy steps \
  --eval_steps 250 \
  --freeze_layer_norm \
  --learning_rate 5e-5 \
  --metric_for_best_model eval_f1 \
  --load_best_model_at_end \
  --eval_split validation \
  --save_total_limit 2
