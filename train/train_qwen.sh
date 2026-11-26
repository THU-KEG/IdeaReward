#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: bash train/train_qwen.sh DATA.jsonl MODEL_PATH OUTPUT_DIR" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_PATH="$(realpath "$1")"
MODEL_PATH="$(realpath "$2")"
OUTPUT_DIR="$(realpath -m "$3")"
WANDB_ARGS=()
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  WANDB_ARGS=(--logger.wandb.key "$WANDB_API_KEY")
fi

cd "$ROOT/train/OpenRLHF"
deepspeed --num_gpus "${NUM_GPUS:-4}" --master_port "${MASTER_PORT:-29500}" \
  --module openrlhf.cli.train_rm \
  --model.model_name_or_path "$MODEL_PATH" \
  --data.dataset "$DATA_PATH" \
  --ckpt.output_dir "$OUTPUT_DIR" \
  --data.chosen_key chosen \
  --data.rejected_key rejected \
  --data.apply_chat_template \
  --train.max_epochs 1 \
  --train.batch_size 256 \
  --train.micro_batch_size 1 \
  --data.max_len 16384 \
  --adam.lr 2e-5 \
  --ds.param_dtype bf16 \
  --ds.zero_stage 3 \
  --ds.attn_implementation flash_attention_2 \
  --ds.packing_samples \
  --model.gradient_checkpointing_enable \
  --ckpt.save_steps -1 \
  --eval.steps -1 \
  "${WANDB_ARGS[@]}"
