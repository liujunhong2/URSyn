#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# conda activate ursyn
# HYDRA_FULL_ERROR=1 bash run_scripts/visionandtouch_reg_run.sh

export TOKENIZERS_PARALLELISM=true

# ================================== visionandtouch (regression) ==================================
for seed in {42..46}; do
    python main_multibench_reg.py \
        seed=${seed} \
        model=ursyn \
        trainer.devices=1 \
        trainer.max_epochs=100 \
        data.data_module.dataset=visionandtouch \
        model.model.encoder.embed_dim=128 \
        optim.lr=0.0001 \
        optim.weight_decay=0.01 \
        model.model.loss_kwargs.epsilon=0.21 \
        model.model.loss_kwargs.entropy_weight=0.01 \
        model.model.loss_kwargs.align_weight=1.0 \
        model.model.loss_kwargs.lambda_repel=20 \
        model.model.loss_kwargs.lambda_R=2.0 \
        model.model.loss_kwargs.lambda_UR_dis=1 \
        model.model.loss_kwargs.lambda_S=2.2 \
        model.model.loss_kwargs.lambda_S_dis=5 \
        mode=train \
        exp_name=visionandtouch_reg
done
