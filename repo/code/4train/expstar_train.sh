#!/bin/bash
export NPROC_PER_NODE=4
export MASTER_PORT=29506 
export VIDEO_MAX_PIXELS=$((768 * 16 * 16)) 
export FPS_MAX_FRAMES=64
export FPS=1.0
export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL="/path/to/Qwen2.5-VL-7B-Instruct"
DATASET="/path/to/expstar_data"

swift sft \
    --model ${MODEL}\
    --dataset ${DATASET} \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --save_strategy "epoch" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --gradient_accumulation_steps 1 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir /path/to/outputs/expstar \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 1 \
    --deepspeed zero2 \
    --split_dataset_ratio 0 \
    --save_only_model true
    
