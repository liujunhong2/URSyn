from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate
import numpy as np
import os
import torch

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from utils import CheckNaNGradCallback


@hydra.main(version_base=None, config_name="train_mmimdb", config_path="./configs")
def main(cfg: DictConfig):
    """Training/test of URSyn on the MM-IMDb dataset."""

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed)

    dataset = "mmimdb"
    kwargs = {}

    if cfg.model.name in {"URSyn", "CoMM"}:
        encoders = instantiate(cfg[dataset]["encoders"])
        adapters = instantiate(cfg[dataset]["adapters"])
        kwargs["encoder"] = {
            "encoders": encoders,
            "input_adapters": adapters
        }

    model = instantiate(cfg.model.model, optim_kwargs=cfg.optim, **kwargs)
    model.save_hyperparameters(cfg)

    data_module = instantiate(cfg.data.data_module, model=cfg.model.name)
    downstream_data_module = instantiate(cfg.data.data_module, model="Sup")


    logger = TensorBoardLogger(
        save_dir=build_root_dir(cfg),
        name="logs",
        default_hp_metric=False
    )

    monitor_metric = "f1_mean"

    linear_probe_callback = instantiate(
        cfg.linear_probing,
        downstream_data_modules=[downstream_data_module],
        names=[dataset]
    )

    checkpoint_best = ModelCheckpoint(
        dirpath=os.path.join(logger.log_dir, "checkpoints"),
        filename=f"best-epoch={{epoch:02d}}-{monitor_metric}={{{monitor_metric}:.4f}}",
        monitor=monitor_metric,
        mode="max",
        save_top_k=1,
        save_last=False,
        verbose=True,
        auto_insert_metric_name=False
    )

    early_stop = EarlyStopping(
        monitor=monitor_metric,
        patience=getattr(cfg, "early_stop_patience", 35),
        mode="max",
        verbose=True
    )

    nan_grad_guard = CheckNaNGradCallback()

    callbacks = [
        linear_probe_callback,
        checkpoint_best,
        early_stop,
        nan_grad_guard
    ]

    trainer = instantiate(
        cfg.trainer,
        default_root_dir=build_root_dir(cfg),
        logger=logger,
        callbacks=callbacks
    )

    if cfg.mode == "train":
        trainer.fit(model, datamodule=data_module)
    else:
        trainer.test(
            model,
            datamodule=data_module,
            ckpt_path=getattr(cfg, "ckpt_path", None)
        )

def build_root_dir(cfg: DictConfig):
    root_dir = os.path.join(cfg.trainer.default_root_dir, cfg.model.name, "mmimdb")

    if cfg.mode == "test":
        ckpt_path = getattr(cfg, "ckpt_path", None)
        if ckpt_path is None:
            print(UserWarning("`ckpt_path` is not set during testing."))
        else:
            root_dir = os.path.join(os.path.dirname(ckpt_path), "test")

    if getattr(cfg, "exp_name", None) is not None:
        root_dir = os.path.join(root_dir, cfg.exp_name)

    return root_dir


if __name__ == "__main__":
    main()
