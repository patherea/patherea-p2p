"""BaseRunner implementation"""

import random
import sys
from abc import ABC, abstractmethod

import albumentations as A
import imgaug
import mlflow
import numpy as np
import torch
import torchvision.transforms as transforms
from loguru import logger
from omegaconf.dictconfig import DictConfig
from torch import nn
from tqdm import tqdm

from p2p.utils.utils import Metrics, reduce_metrics  # noqa


class BaseRunner(ABC):
    @abstractmethod
    def __init__(
        self,
        cfg: DictConfig,
        backbone: str,
        pretrained: bool = False,
        method_seed: int = None,
        data_seed: int = None,
        save_checkpoint: bool = False,
        log_checkpoint: bool = False,
    ) -> None:
        """Base runner implementation.

        This is the abstract class to implement various cell detection &
        classification methods on various datasets.

        Args:
            cfg:
                Config object
            backbone:
                Backbone to be used (e.g., VGG-16).
            pretrained:
                Bool whether to use ImageNet pretrained weights (default=False).
            method_seed:
                Seed to control the method and augmentation. (default=None)
                None = random initialization.
            data_seed:
                Seed to control the dataset split. (default=None)
                None = random initialization.
            save_checkpoint:
                Whether to save checkpoints on every save_eval_freq (default=False).
            log_checkpoint:
                Whether to save checkpoint/model to MLFlow (default=False).
                It saves the latest and best checkpoint.
        """
        super().__init__()

        self.cfg = cfg
        self.start_epoch = 1
        self.method_seed = method_seed if method_seed else random.randrange(sys.maxsize)
        self.data_seed = data_seed if data_seed else random.randrange(sys.maxsize)
        self.save_checkpoint = save_checkpoint
        self.log_checkpoint = log_checkpoint
        self.backbone = backbone
        self.pretrained = pretrained
        self.normalize_t_base = A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.normalize_t_imagenet = A.Normalize(
            mean=cfg.model.normalize_mean, std=cfg.model.normalize_std
        )

        self.normalize_t = (
            self.normalize_t_imagenet if self.pretrained else self.normalize_t_base
        )
        self.invTrans = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=(1 / np.asarray(self.normalize_t.std)).tolist(),
                ),
                transforms.Normalize(
                    mean=(-np.asarray(self.normalize_t.mean)).tolist(),
                    std=[1.0, 1.0, 1.0],
                ),
            ]
        )

        # Random seed is set for augmentations
        # and torch model initializations.
        # https://github.com/albumentations-team/albumentations/issues/93
        random.seed(self.method_seed)
        imgaug.random.seed(self.method_seed)
        torch.manual_seed(self.method_seed)

    def run(self, reduce: str = "max") -> Metrics:
        """Defines the execution pipeline.

        This method implements the standard training/eval/test pipeline for
        cell detection & classification.

        Args:
            reduce:
                Aggregation strategy for the metrics (default: mean).
                Mean and max reduce strategies are only supported for now.
                Reduction is performed across the epochs.

        Returns:
            Combined (for all epochs) metrics are returned (MAE, Pr, Re, F1).
            Aggregation is performed as specified using the reduce parameter.
            Checkpoints are saved to Hydra local folder (optional).
            Final checkpoint is persisted using MLFlow (optional).
        """
        logger.info(f"Started training at epoch: {self.start_epoch}.")
        pbar = tqdm(
            range(self.start_epoch, self.cfg.dataset.epochs + 1),
            desc="epochs (TRAIN)",
            leave=True,
            position=0,
        )
        metrics_test_all = []  # noqa
        for epoch in pbar:
            model = self._train(epoch)

            if epoch % self.cfg.params.save_eval_freq == 0:
                self.validate(
                    epoch,
                    n_epoch_validate=self.cfg.dataset.epochs_validate,
                    save_patches=self.cfg.debug.save_val_patches,
                )
                metrics_test: Metrics = self.test(
                    epoch,
                    save_images=self.cfg.debug.save_test_images,
                    save_patches=self.cfg.debug.save_test_patches,
                )
                if len(metrics_test_all) > 0:
                    better = metrics_test.f1 > metrics_test_all[-1].f1
                else:
                    better = True
                metrics_test_all.append(metrics_test)

                if self.save_checkpoint:
                    self._save_checkpoint(epoch)

                if self.log_checkpoint and better:
                    mlflow.pytorch.log_state_dict(
                        model.state_dict(), artifact_path="checkpoint_best"
                    )
                if self.log_checkpoint:
                    mlflow.pytorch.log_state_dict(
                        model.state_dict(), artifact_path="checkpoint_latest"
                    )

        return reduce_metrics(metrics_test_all, reduce)

    @abstractmethod
    def _train(self, epoch: int) -> nn.Module:
        raise NotImplementedError()

    @abstractmethod
    def validate(
        self, epoch: int, n_epoch_validate: int = 1, save_patches: bool = False
    ) -> Metrics:
        raise NotImplementedError()

    @abstractmethod
    def test(
        self, epoch: int, save_images: bool = True, save_patches: bool = False
    ) -> Metrics:
        raise NotImplementedError()

    @abstractmethod
    def _save_checkpoint(self, epoch: int) -> None:
        raise NotImplementedError()

    @abstractmethod
    def load_checkpoint(self, checkpoint_path: str) -> None:
        raise NotImplementedError()
