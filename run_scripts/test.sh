#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# conda activate ursyn
# Use the same entry point and parameters as the training command for this checkpoint.
entry="main_multibench.py"
dataset="mosi"
seed=42
ckpt_path="/path/to/your/best_checkpoint_on_validation.ckpt"

python "$entry" \
    seed="${seed}" \
    model=ursyn \
    trainer.devices=1 \
    data.data_module.dataset="${dataset}" \
    mode=test \
    "ckpt_path='${ckpt_path}'"
