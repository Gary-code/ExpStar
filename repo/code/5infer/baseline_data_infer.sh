#!/bin/bash
export MASTER_PORT=29543
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
export CUDA_VISIBLE_DEVICES=0
export VIDEO_MAX_PIXELS=$((768 * 16 * 16)) 
export FPS_MAX_FRAMES=24     
export FPS=1.0
export CUDA_VISIBLE_DEVICES=0,1,2,3

CHECKPOINT_PATH="/path/to/checkpoint"
MAX_MODEL_LEN=32000
MAX_NEW_TOKENS=1024
MAX_SEQUENCE_LEN=16
MAX_IMAGE_NUMS='{"image":4,"video":2}'
TENSOR_PARALLEL_SIZE=1

echo "Starting inference..."
echo "Using checkpoint: $CHECKPOINT_PATH"
echo "Maximum model length: $MAX_MODEL_LEN"
echo "Maximum new tokens: $MAX_NEW_TOKENS"
echo "Temperature: $TEMPERATURE"
echo "Video FPS: $FPS"

swift infer \
    --adapters $CHECKPOINT_PATH \
    --stream true \
    --merge_lora true \
    --infer_backend vllm \
    --dataset  /path/to/merge_test.jsonl \
    --split_dataset_ratio 1.0 \
    --max_model_len $MAX_MODEL_LEN \
    --max_new_tokens $MAX_NEW_TOKENS \
    --result_path /path/to/baseline.jsonl \
    --gpu_memory_utilization 0.95 

echo "Inference completed"