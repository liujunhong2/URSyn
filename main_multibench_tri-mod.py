import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import numpy as np
import os
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


@hydra.main(version_base=None, config_name="train_multibench_tri-mod", config_path="./configs")
def main(cfg: DictConfig):
    """Training/test of URSyn on the MultiBench dataset with three modalities."""

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed)

    dataset = cfg.data.data_module.dataset
    encoders = instantiate(cfg[dataset]["encoders"])
    adapters = instantiate(cfg[dataset]["adapters"])
    kwargs = {"encoder": {"encoders": encoders, "input_adapters": adapters}}

    model = instantiate(cfg.model.model, optim_kwargs=cfg.optim, **kwargs)
    model.save_hyperparameters(cfg)

    data_module = instantiate(cfg.data.data_module,
                              model=cfg.model.name,
                              modalities=cfg[dataset]["modalities"],
                              task=cfg[dataset]["task"],
                              **cfg[dataset]["kwargs"])

    downstream_data_module = instantiate(cfg.data.data_module,
                                         model="Sup",
                                         modalities=cfg[dataset]["modalities"],
                                         task=cfg[dataset]["task"])

    logger = TensorBoardLogger(
        save_dir=build_root_dir(cfg),
        name="logs",
        default_hp_metric=False
    )

    linear_probe_callback = instantiate(
        cfg.linear_probing,
        downstream_data_modules=[downstream_data_module],
        names=[dataset]
    )

    checkpoint_best = ModelCheckpoint(
        dirpath=os.path.join(logger.log_dir, "checkpoints"),
        filename="best-epoch={epoch:02d}-acc1={acc1:.4f}",
        monitor="acc1",
        mode="max",
        save_top_k=1,
        save_last=False,
        verbose=True,
        auto_insert_metric_name=False
    )

    early_stop = EarlyStopping(
        monitor="acc1",
        patience=35,
        mode="max",
        verbose=True
    )

    trainer = instantiate(
        cfg.trainer,
        default_root_dir=build_root_dir(cfg),
        logger=logger,
        callbacks=[linear_probe_callback, checkpoint_best, early_stop]
    )

    if cfg.mode == "train":
        trainer.fit(model, datamodule=data_module)
    else:
        trainer.test(model, datamodule=data_module, ckpt_path=cfg.ckpt_path)


def build_root_dir(cfg: DictConfig):
    root_dir = os.path.join(cfg.trainer.default_root_dir, cfg.model.name, cfg.data.data_module.dataset)

    if cfg.mode == "test":
        if cfg.ckpt_path is None:
            print(UserWarning("`ckpt_path` is not set during testing."))
        else:
            root_dir = os.path.join(os.path.dirname(cfg.ckpt_path), "test")

    if cfg.exp_name is not None:
        root_dir = os.path.join(root_dir, cfg.exp_name)

    return root_dir


if __name__ == "__main__":
    main()
