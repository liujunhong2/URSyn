#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# conda activate ursyn
# HYDRA_FULL_ERROR=1 bash run_scripts/multibench_run.sh

# Regularization coefficients (entropy_weight / align_weight / lambda_repel) and
# learning rates follow the paper. epsilon, lambda_R, lambda_UR_dis, lambda_S
# and lambda_S_dis are selected per dataset.

# ================================== mosi ==================================
for seed in {42..46}; do
    python main_multibench.py \
        seed=${seed} \
        model=ursyn \
        trainer.devices=1 \
        trainer.strategy=ddp_find_unused_parameters_false \
        trainer.max_epochs=100 \
        data.data_module.dataset=mosi \
        optim.lr=0.001 \
        optim.weight_decay=0.1 \
        model.model.loss_kwargs.epsilon=0.25 \
        model.model.loss_kwargs.entropy_weight=0.1 \
        model.model.loss_kwargs.align_weight=1.0 \
        model.model.loss_kwargs.lambda_repel=10 \
        model.model.loss_kwargs.lambda_R=1.5 \
        model.model.loss_kwargs.lambda_UR_dis=25 \
        model.model.loss_kwargs.lambda_S=2.2 \
        model.model.loss_kwargs.lambda_S_dis=20 \
        mode=train \
        exp_name=bi_mosi
done

# ================================== humor ==================================
for seed in {42..46}; do
    python main_multibench.py \
        seed=${seed} \
        model=ursyn \
        trainer.devices=1 \
        trainer.strategy=ddp_find_unused_parameters_false \
        trainer.max_epochs=100 \
        data.data_module.dataset=humor \
        optim.lr=0.001 \
        optim.weight_decay=0.01 \
        model.model.loss_kwargs.epsilon=0.2 \
        model.model.loss_kwargs.entropy_weight=0.1 \
        model.model.loss_kwargs.align_weight=1.0 \
        model.model.loss_kwargs.lambda_repel=20 \
        model.model.loss_kwargs.lambda_R=0.7 \
        model.model.loss_kwargs.lambda_UR_dis=25 \
        model.model.loss_kwargs.lambda_S=1.9 \
        model.model.loss_kwargs.lambda_S_dis=3.0 \
        mode=train \
        exp_name=bi_humor
done

# ================================== sarcasm ==================================
for seed in {42..46}; do
    python main_multibench.py \
        seed=${seed} \
        model=ursyn \
        trainer.devices=1 \
        trainer.strategy=ddp_find_unused_parameters_false \
        trainer.max_epochs=100 \
        data.data_module.dataset=sarcasm \
        optim.lr=0.001 \
        optim.weight_decay=0.01 \
        model.model.loss_kwargs.epsilon=0.23 \
        model.model.loss_kwargs.entropy_weight=0.05 \
        model.model.loss_kwargs.align_weight=1.0 \
        model.model.loss_kwargs.lambda_repel=20 \
        model.model.loss_kwargs.lambda_R=1.9 \
        model.model.loss_kwargs.lambda_UR_dis=2 \
        model.model.loss_kwargs.lambda_S=2.2 \
        model.model.loss_kwargs.lambda_S_dis=5 \
        mode=train \
        exp_name=bi_sarcasm
done

# ================================== mimic ==================================
for seed in {42..46}; do
    python main_multibench.py \
        seed=${seed} \
        model=ursyn \
        trainer.devices=1 \
        trainer.strategy=ddp_find_unused_parameters_false \
        trainer.max_epochs=100 \
        data.data_module.dataset=mimic \
        model.model.encoder.embed_dim=512 \
        optim.lr=0.001 \
        optim.weight_decay=0.01 \
        model.model.loss_kwargs.entropy_weight=0.02 \
        model.model.loss_kwargs.align_weight=1.0 \
        model.model.loss_kwargs.lambda_repel=7 \
        mode=train \
        exp_name=bi_mimic
done

# ================================== visionandtouch ==================================
for seed in {42..46}; do
    python main_multibench.py \
        seed=${seed} \
        model=ursyn \
        trainer.devices=1 \
        trainer.strategy=ddp_find_unused_parameters_false \
        trainer.max_epochs=100 \
        data.data_module.dataset=visionandtouch-bin \
        model.model.encoder.embed_dim=512 \
        optim.lr=0.0001 \
        optim.weight_decay=0.01 \
        model.model.loss_kwargs.entropy_weight=0.001 \
        model.model.loss_kwargs.align_weight=1.0 \
        model.model.loss_kwargs.lambda_repel=20 \
        mode=train \
        exp_name=bi_visionandtouch
done
