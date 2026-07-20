#!/usr/bin/env bash

set -euo pipefail

# Start an OpenAI-compatible vLLM server for DeepEval judge calls.
#
# Required/typical:
#   JUDGE_MODEL=Qwen/Qwen3.6-27B
#
# Useful overrides:
#   JUDGE_HOST=0.0.0.0
#   JUDGE_PORT=8000
#   JUDGE_TP=4
#   JUDGE_DP=2
#   JUDGE_DP_LOCAL=$JUDGE_DP
#   JUDGE_MAX_MODEL_LEN=32768
#   JUDGE_MAX_NUM_SEQS=64
#   JUDGE_GPU_MEMORY_UTILIZATION=0.90
#   JUDGE_DTYPE=bfloat16
#   JUDGE_SERVED_MODEL_NAME=$JUDGE_MODEL
#
# Extra vLLM args may be appended directly, or after "--".

MODEL="${JUDGE_MODEL:-${MODEL:-Qwen/Qwen3.6-27B}}"
SERVED_MODEL_NAME="${JUDGE_SERVED_MODEL_NAME:-$MODEL}"
HOST="${JUDGE_HOST:-0.0.0.0}"
PORT="${JUDGE_PORT:-8000}"
TP="${JUDGE_TP:-${TENSOR_PARALLEL_SIZE:-1}}"
DP="${JUDGE_DP:-${DATA_PARALLEL_SIZE:-}}"
DP_LOCAL="${JUDGE_DP_LOCAL:-${DATA_PARALLEL_SIZE_LOCAL:-${DP:-}}}"
MAX_MODEL_LEN="${JUDGE_MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${JUDGE_MAX_NUM_SEQS:-64}"
GPU_MEMORY_UTILIZATION="${JUDGE_GPU_MEMORY_UTILIZATION:-0.90}"
DTYPE="${JUDGE_DTYPE:-bfloat16}"

if [[ "${JUDGE_DISABLE_FLASHINFER_SAMPLER:-0}" == "1" ]]; then
  export VLLM_USE_FLASHINFER_SAMPLER=0
fi

ARGS=(
  serve "$MODEL"
  --host "$HOST"
  --port "$PORT"
  --served-model-name "$SERVED_MODEL_NAME"
  --tensor-parallel-size "$TP"
  --dtype "$DTYPE"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
)

if [[ -n "$DP" ]]; then
  ARGS+=(--data-parallel-size "$DP")
  if [[ -n "$DP_LOCAL" ]]; then
    ARGS+=(--data-parallel-size-local "$DP_LOCAL")
  fi
fi

if [[ "${JUDGE_TRUST_REMOTE_CODE:-0}" == "1" ]]; then
  ARGS+=(--trust-remote-code)
fi
if [[ -n "${JUDGE_TOKENIZER:-}" ]]; then
  ARGS+=(--tokenizer "$JUDGE_TOKENIZER")
fi
if [[ -n "${JUDGE_CHAT_TEMPLATE:-}" ]]; then
  ARGS+=(--chat-template "$JUDGE_CHAT_TEMPLATE")
fi

echo "[vllm_judge] model = $MODEL"
echo "[vllm_judge] served_model_name = $SERVED_MODEL_NAME"
echo "[vllm_judge] base_url = http://$HOST:$PORT/v1"
echo "[vllm_judge] tensor_parallel_size = $TP"
echo "[vllm_judge] data_parallel_size = ${DP:-<vLLM default>}"
echo "[vllm_judge] data_parallel_size_local = ${DP_LOCAL:-<vLLM default>}"
echo "[vllm_judge] max_model_len = $MAX_MODEL_LEN"
echo "[vllm_judge] max_num_seqs = $MAX_NUM_SEQS"
echo "[vllm_judge] gpu_memory_utilization = $GPU_MEMORY_UTILIZATION"
echo "[vllm_judge] dtype = $DTYPE"

if [[ "${1:-}" == "--" ]]; then
  shift
fi

exec vllm "${ARGS[@]}" "$@"
