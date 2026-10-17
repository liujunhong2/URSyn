#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# conda activate ursyn
# HYDRA_FULL_ERROR=1 bash run_scripts/trifeatures_run.sh

# Shared hyper-parameters, see configs/model/ursyn.yaml for the defaults.
declare -a loss_args=(
    "model.model.loss_kwargs.epsilon=0.19"
    "model.model.loss_kwargs.entropy_weight=0.03"
    "model.model.loss_kwargs.align_weight=1.0"
    "model.model.loss_kwargs.lambda_repel=10"
    "model.model.loss_kwargs.lambda_R=1.1"
    "model.model.loss_kwargs.lambda_UR_dis=3"
    "model.model.loss_kwargs.lambda_S=4.0"
    "model.model.loss_kwargs.lambda_S_dis=4"
)

# ================================== trifeatures_non_syn ==================================
for seed in {42..46}; do
    python main_trifeatures.py \
        seed=${seed} \
        model=ursyn \
        trainer.devices=1 \
        trainer.strategy=ddp_find_unused_parameters_true \
        trainer.max_epochs=100 \
        model.model.encoder.embed_dim=512 \
        data.data_module.biased=false \
        optim.lr=0.0003 \
        optim.weight_decay=0.0001 \
        "${loss_args[@]}" \
        mode=train \
        exp_name=trifeatures_non_syn
done

# ================================== trifeatures_syn ==================================
for seed in {42..46}; do
    python main_trifeatures.py \
        seed=${seed} \
        model=ursyn \
        trainer.devices=1 \
        trainer.strategy=ddp_find_unused_parameters_true \
        trainer.max_epochs=100 \
        model.model.encoder.embed_dim=512 \
        data.data_module.biased=true \
        optim.lr=0.0003 \
        optim.weight_decay=0.0001 \
        "${loss_args[@]}" \
        mode=train \
        exp_name=trifeatures_syn
done
