# --nnodes 1 --nproc_per_node 4 --master_port 25641
#分布式训练脚本代码
deepspeed --include localhost:0,1,2,3 ../llava_train.py \
    --deepspeed /home/yuxuan/yuanxuan-new/ds_zero2_no_offload.json \
    --base_model_name_or_path /home/yuxuan/yuanxuan-new/show_model/model001 \
    --lora_weights  /home/yuxuan/yuanxuan-new/output_model_user_lora_0705/checkpoint-27906 \
    --train_type use_lora \
    --data_path /home/yuxuan/yuanxuan-new/dataset-temp \
    --remove_unused_columns false \
    --bf16 true \
    --fp16 false \
    --dataloader_pin_memory True \
    --dataloader_num_workers 10 \
    --dataloader_persistent_workers True \
    --output_dir /home/yuxuan/yuanxuan-new/result/juice-bottle/train2new-output/output_model_user_lora_0514 \
    --num_train_epochs 80 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --logging_dir /home/yuxuan/yuanxuan-new/result/juice-bottle/train2new-output/logs \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --report_to "tensorboard" \
    --learning_rate 4e-4 \
    --logging_steps 10
# --model_max_length 2048

# --save_strategy "steps" \
# --save_steps 10 \
# --save_steps 1000 \
# --save_strategy "epoch" \