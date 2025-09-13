#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate swift_env

export MASTER_PORT=29543
export VIDEO_MAX_PIXELS=$((768 * 16 * 16))  
export FPS_MAX_FRAMES=24                 
export FPS=1.0

LORA_CHECKPOINT_PATH="checkpoints/expstar/checkpoint"
MODEL_PATH="models/Qwen2.5-VL-7B-Instruct"
MAX_MODEL_LEN=32000
MAX_NEW_TOKENS=1024
MAX_SEQUENCE_LEN=16
MAX_IMAGE_NUMS='{"image":4,"video":2}'
TENSOR_PARALLEL_SIZE=1
CUDA_VISIBLE_DEVICES=0,1,2,3

group_size=1  
gpu_list=(${CUDA_VISIBLE_DEVICES//,/ })
group_count=$((${#gpu_list[@]} / group_size))

echo "Starting deployment..."
echo "Using Lora checkpoint: $LORA_CHECKPOINT_PATH"
echo "Maximum model length: $MAX_MODEL_LEN"
echo "Maximum new tokens: $MAX_NEW_TOKENS"
echo "Video FPS: $FPS"

for group in $(seq 0 $((group_count - 1))); do
    start_idx=$((group * group_size))
    gpu_subset=(${gpu_list[@]:start_idx:group_size})
    gpu_subset_str=$(IFS=','; echo "${gpu_subset[*]}")

    export CUDA_VISIBLE_DEVICES=$gpu_subset_str
    port=$((9020 + group))

    echo "Launching on GPUs: $CUDA_VISIBLE_DEVICES with port $port"
    swift deploy \
        --ckpt_dir $LORA_CHECKPOINT_PATH \
        --merge_lora true \
        --stream true \
        --model $MODEL_PATH \
        --infer_backend vllm \
        --max_model_len $MAX_MODEL_LEN \
        --max_num_seqs $MAX_SEQUENCE_LEN \
        --max_new_tokens $MAX_NEW_TOKENS \
        --gpu_memory_utilization 0.95 \
        --limit_mm_per_prompt $MAX_IMAGE_NUMS \
        --port $port \
        --tensor_parallel_size $TENSOR_PARALLEL_SIZE &
done

wait

echo "Inference completed"