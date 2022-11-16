#!/bin/bash

SOURCE_LANG=es
TARGET_LANG=gn
DATA_ROOT=$HOME/datasets/americasnlp2021/data/guarani-spanish

python pretrain.py \
    --do_train \
    --do_eval \
    --output_dir models/${SOURCE_LANG}-${TARGET_LANG}-test \
    --mt_train_file "$DATA_ROOT/train.json" \
    --mt_validation_file "$DATA_ROOT/dev.json" \
    --dn_train_file $HOME/projects/composable-sft/examples/language-modeling/corpora/${TARGET_LANG}.txt \
    --source_lang "${SOURCE_LANG}_XX" \
    --target_lang "${TARGET_LANG}_XX" \
    --model_name_or_path facebook/mbart-large-50-many-to-many-mmt \
    --max_source_length 128 \
    --max_target_length 128 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --save_strategy epoch \
    --save_total_limit 1 \
    --optim adafactor \
    --learning_rate 5e-5 \
    --fp16 \
    --num_train_epochs 3 \
    --preprocessing_num_workers 15 \
    --overwrite_output_dir

