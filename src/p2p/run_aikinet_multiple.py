"""AIKiNET multiclass starting script"""

import os

import hydra
import mlflow
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from loguru import logger
from omegaconf import DictConfig

from p2p.runners.aikinet_runner_multiple import AIKiNETMultipleRunner
from p2p.utils.utils import Metrics


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> float:
    """Executes the AIKiNET multi-class pipeline.

    Args:
        cfg: Hydra config object.

    Returns:
        Returns the reduced metrics across the epochs (default: F1).
        The chosen metric represents the aggregated metric across
        all the classes.
    """
    mlflow.set_experiment(cfg.params.experiment_name)
    with mlflow.start_run(run_name=cfg.params.run_name):
        if cfg.dataset.test_fold:
            # Override folds with the manually chosen fold
            cfg.dataset.test_folds = [cfg.dataset.test_fold]
        metrics_folds: list[Metrics] = []
        cfg.dataset.test_folds.sort()
        for fold in cfg.dataset.test_folds:
            logger.info(f"Started processing fold: {fold}")
            cfg.dataset.test_fold = fold
            with mlflow.start_run(
                run_name=f"fold_{fold}", description=cfg.params.desc, nested=True
            ):
                mlflow.log_params(cfg.params)
                mlflow.log_params(cfg.model)
                mlflow.log_params(cfg.dataset)
                mlflow.log_param("cwd", os.getcwd())

                aikinet_runner = AIKiNETMultipleRunner(
                    cfg=cfg,
                    save_checkpoint=cfg.params.save_checkpoint,
                    log_checkpoint=cfg.params.log_checkpoint,
                    backbone=cfg.model.backbone,
                    pretrained=cfg.model.pretrained,
                    method_seed=cfg.model.method_seed,
                    data_seed=cfg.dataset.data_seed,
                )

                logger.info("Started training...")
                metrics_folds.append(aikinet_runner.run())
                logger.info("Finished training.")

        return metrics_folds


if __name__ == "__main__":
    main()
