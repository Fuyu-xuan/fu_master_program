#!/bin/bash

python train_internvl.py \
  --model_name_or_path OpenGVLab/InternVL2-8B \
  --data_path ./data/train.json \
  --train_type use_lora \
  --output_dir ./output_internvl2_8b_lora \
  --num_train_epochs 80 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --bf16 True \
  --logging_steps 10 \
  --save_steps 500 \
  --remove_unused_columns False \
  --report_to none