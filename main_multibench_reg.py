from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate
import numpy as np
import os
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping


class EpochInfoCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        pl_module.set_current_epoch(trainer.current_epoch)


class TotalEpochsCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        pl_module.set_total_epochs(trainer.max_epochs)


@hydra.main(version_base=None, config_name="train_multibench", config_path="./configs")
def main(cfg: DictConfig):
    """Training/test of URSyn on the regression MultiBench task (visionandtouch)."""

    torch.manual_seed(cfg.seed)
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

    logger = TensorBoardLogger(build_root_dir(cfg), name="logs")
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(logger.log_dir, "checkpoints"),
        filename="{epoch:02d}-{mse:.6f}",
        monitor="mse",
        mode="min",
        save_top_k=1,
        save_last=False,
        verbose=True,
        auto_insert_metric_name=False
    )
    early_stopping_callback = EarlyStopping(
        monitor="mse",
        patience=45,
        verbose=True,
        mode="min",
    )

    trainer = instantiate(
        cfg.trainer,
        default_root_dir=build_root_dir(cfg),
        logger=[logger],
        callbacks=[
            EpochInfoCallback(),
            TotalEpochsCallback(),
            instantiate(cfg.linear_probing_reg, downstream_data_modules=[downstream_data_module], names=[dataset]),
            early_stopping_callback,
            checkpoint_callback
        ],
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


if __name__ == '__main__':
    main()
