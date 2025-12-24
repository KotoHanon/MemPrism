#!/bin/bash

# Set environment variables
# export MODEL_PATH="/raid/shared/mem1/models/Qwen2.5-7B-search-sft-v2/v0-20250511-083818/checkpoint-1485"  # Replace with your actual model path
export MODEL_PATH=".cache/Qwen3-4B"  # use the official MEM1 model


# vLLM server configuration
HOST="0.0.0.0"
PORT="8014"
TENSOR_PARALLEL_SIZE=1  # Using all 4 GPUs for tensor parallelism
MAX_MODEL_LEN=32768
GPU_MEMORY_UTILIZATION=0.6
# Start vLLM server
CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
  --model $MODEL_PATH \
  --served-model-name qwen3-4b \
  --host $HOST \
  --port $PORT \
  --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
  --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
  --max-model-len $MAX_MODEL_LEN

