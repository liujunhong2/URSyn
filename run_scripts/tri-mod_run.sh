#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# conda activate ursyn
# HYDRA_FULL_ERROR=1 bash run_scripts/tri-mod_run.sh

# Regularization coefficients (entropy_weight / align_weight / lambda_repel) and
# learning rates follow the paper.

# ================================== humor ==================================
for seed in {42..46}; do
    python main_multibench_tri-mod.py \
        seed=${seed} \
        model=ursyn \
        trainer.devices=1 \
        trainer.max_epochs=100 \
        data.data_module.dataset=humor \
        model.model.encoder.embed_dim=40 \
        optim.lr=0.001 \
        optim.weight_decay=0.01 \
        model.model.loss_kwargs.entropy_weight=0.002 \
        model.model.loss_kwargs.align_weight=1.0 \
        model.model.loss_kwargs.lambda_repel=15 \
        mode=train \
        exp_name=tri_humor
done

# ================================== visionandtouch ==================================
for seed in {42..46}; do
    python main_multibench_tri-mod.py \
        seed=${seed} \
        model=ursyn \
        trainer.devices=1 \
        trainer.max_epochs=100 \
        data.data_module.dataset=visionandtouch-bin \
        model.model.encoder.embed_dim=512 \
        optim.lr=0.001 \
        optim.weight_decay=0.01 \
        model.model.loss_kwargs.entropy_weight=0.003 \
        model.model.loss_kwargs.align_weight=1.0 \
        model.model.loss_kwargs.lambda_repel=2 \
        mode=train \
        exp_name=tri_visionandtouch
done
