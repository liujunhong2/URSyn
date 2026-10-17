#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# conda activate ursyn
# HYDRA_FULL_ERROR=1 bash run_scripts/MM_IMDb_run.sh

export TOKENIZERS_PARALLELISM=true

# ================================== mmimdb ==================================
for seed in {42..46}; do
    python main_mmimdb.py \
        seed=${seed} \
        model=ursyn \
        trainer.devices=1 \
        trainer.max_epochs=100 \
        model.model.encoder.embed_dim=768 \
        optim.lr=0.001 \
        optim.weight_decay=0.01 \
        model.model.loss_kwargs.entropy_weight=0.01 \
        model.model.loss_kwargs.align_weight=1.0 \
        model.model.loss_kwargs.lambda_repel=2 \
        mode=train \
        exp_name=mmimdb
done
