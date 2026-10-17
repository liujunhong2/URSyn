import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import numpy as np
import os
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from evaluation.linear_probe import LinearProbingCallback


@hydra.main(version_base=None, config_name="train_trifeatures", config_path="./configs")
def main(cfg: DictConfig):
    """Training/test of URSyn on synthetic toy data (bimodal trifeatures) with
    controllable attributes (shape, color, texture)."""

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    kwargs = {"encoder": {
        "encoders": instantiate(cfg.model.encoders),
        "input_adapters": instantiate(cfg.model.adapters)}}

    model = instantiate(cfg.model.model, optim_kwargs=cfg.optim, **kwargs)
    model.save_hyperparameters(cfg)

    data_module = instantiate(cfg.data.data_module, model=cfg.model.name)

    downstream_names = ["share", "unique1", "unique2", "synergy"]
    downstream_data_modules = [instantiate(cfg.data.data_module, model="Sup", biased=False, task=t)
                               for t in downstream_names]

    trainer = instantiate(
        cfg.trainer,
        default_root_dir=build_root_dir(cfg),
        logger=[TensorBoardLogger(build_root_dir(cfg), name="logs")],
        callbacks=[LinearProbingCallback(downstream_data_modules,
                                         names=downstream_names,
                                         val_loaders=False)]
    )

    if cfg.mode == "train":
        trainer.fit(model, datamodule=data_module)
    else:
        trainer.test(model, datamodule=data_module, ckpt_path=cfg.ckpt_path)


def build_root_dir(cfg: DictConfig):
    root_dir = os.path.join(cfg.trainer.default_root_dir, cfg.model.name, "bimodal_trifeatures")

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
