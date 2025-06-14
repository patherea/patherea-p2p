"""AIKiNET runner (multiclass)"""

import time
from collections import defaultdict
from pathlib import Path
from typing import Union

import albumentations as A
import hydra
import mlflow
import numpy as np
import torch
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from flash.core.optimizers import LinearWarmupCosineAnnealingLR
from loguru import logger
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from p2p.datasets.aikinet_multiple import COLORS, MAPPING, AIKiNETMultiDataset
from p2p.models.backbones.convnext import ConvNext_FPN
from p2p.models.backbones.vgg import VGG16_FPN
from p2p.models.backbones.vit_adapter import ViTAdapterWrapper
from p2p.models.p2pnet import P2PNet
from p2p.runners.base_runner import BaseRunner
from p2p.utils.utils import (
    Metrics,
    compute_f1_from_metrics,
    compute_image_metrics,
    compute_image_metrics_dist,
    extract_image_patches,
    get_peaks,
    local_to_global_peaks,
    reduce_metrics,
    save_predicted_images,
    save_predicted_images_all,
)


class AIKiNETMultipleRunner(BaseRunner):
    def __init__(
        self,
        cfg: DictConfig,
        backbone: str,
        pretrained: Union[bool, str, Path] = False,
        method_seed: int = None,
        data_seed: int = None,
        save_checkpoint: bool = False,
        log_checkpoint: bool = False,
    ) -> None:
        """Implements AIKiNET dataset runner.

        Args:
            cfg:
                Config object
            backbone:
                Backbone to be used (e.g., VGG-16). For the methods that
                do not use the common backbones, use the method name (e.g., riad).
            pretrained:
                Bool: Whether to use ImageNet pretrained weights (default=False).
                Str/Path: Path to the pretrained model to be used in the backbone.
            method_seed:
                Seed to control the method and augmentation. (default=None)
                None = random initialization.
            data_seed:
                Seed to control the dataset split. (default=None)
                None = random initialization.
            save_checkpoint:
                Whether to save checkpoints on every save_eval_freq (default=False).
            log_checkpoint:
                Whether to save final checkpoint/model to MLFlow (default=False).
        """
        super().__init__(
            cfg,
            backbone,
            pretrained,
            method_seed,
            data_seed,
            save_checkpoint,
            log_checkpoint,
        )

        # Selection of the used classes
        self.classes = {
            key: MAPPING[key]
            for key in set(cfg.dataset.classes).intersection(MAPPING.keys())
        }
        # Add detection class (0), if given in cfg.
        if 0 in cfg.dataset.classes:
            self.classes[0] = "detection"
        # Sort the keys
        self.classes = dict(sorted(self.classes.items()))

        # Initialize transforms for train & val & test
        # mask_X & keypoints_X -> X matches with class ID in a mapping dict.
        # Default targets are used for the combined (i.e. "detection") class.
        self.transforms = {
            "train": A.Compose(
                [
                    A.RandomCrop(width=cfg.model.width, height=cfg.model.height),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    self.normalize_t,
                    ToTensorV2(),
                ],
                additional_targets={
                    "keypoints_1": "keypoints",
                    "keypoints_2": "keypoints",
                    "keypoints_3": "keypoints",
                    "keypoints_4": "keypoints",
                    "keypoints_5": "keypoints",
                },
                keypoint_params=A.KeypointParams(format="xy"),
            ),
            "val": A.Compose(
                [
                    A.RandomCrop(width=cfg.model.width, height=cfg.model.height),
                    self.normalize_t,
                    ToTensorV2(),
                ],
                additional_targets={
                    "keypoints_1": "keypoints",
                    "keypoints_2": "keypoints",
                    "keypoints_3": "keypoints",
                    "keypoints_4": "keypoints",
                    "keypoints_5": "keypoints",
                },
                keypoint_params=A.KeypointParams(format="xy"),
            ),
            "test": A.Compose(
                [self.normalize_t, ToTensorV2()],
                additional_targets={
                    "keypoints_1": "keypoints",
                    "keypoints_2": "keypoints",
                    "keypoints_3": "keypoints",
                    "keypoints_4": "keypoints",
                    "keypoints_5": "keypoints",
                },
                keypoint_params=A.KeypointParams(format="xy"),
            ),
        }

        # Initialize Datasets & DataLoaders
        self.dataset_train = AIKiNETMultiDataset(
            cfg.dataset.data_dir,
            self.transforms["train"],
            train=True,
            seed=self.data_seed,
            test_fold=cfg.dataset.test_fold,
            test_ratio=cfg.dataset.test_ratio,
            solo_other_pos=cfg.dataset.solo_other_pos,
            train_ratio=cfg.dataset.train_ratio,
        )
        self.dataset_val = AIKiNETMultiDataset(
            cfg.dataset.data_dir,
            self.transforms["val"],
            train=False,
            seed=self.data_seed,
            test_fold=cfg.dataset.test_fold,
            test_ratio=cfg.dataset.test_ratio,
        )
        self.dataset_test = AIKiNETMultiDataset(
            cfg.dataset.data_dir,
            self.transforms["test"],
            train=False,
            seed=self.data_seed,
            test_fold=cfg.dataset.test_fold,
            test_ratio=cfg.dataset.test_ratio,
        )
        self.dataloader_train = DataLoader(
            self.dataset_train,
            batch_size=cfg.model.batch_size,
            num_workers=cfg.model.num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=AIKiNETMultiDataset.custom_collate_gt,
        )
        self.dataloader_val = DataLoader(
            self.dataset_val,
            batch_size=1,
            num_workers=cfg.model.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=AIKiNETMultiDataset.custom_collate_gt,
        )
        self.dataloader_test = DataLoader(
            self.dataset_test,
            batch_size=1,
            num_workers=cfg.model.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=AIKiNETMultiDataset.custom_collate_gt,
        )

        logger.info(f"Size of TRAIN dataset: {len(self.dataset_train)}")
        logger.info(f"Size of VAL dataset: {len(self.dataset_val)}")
        logger.info(f"Size of TEST dataset: {len(self.dataset_test)}")

        # Initialize the model
        if backbone == "VGG-16":
            backbone_model = VGG16_FPN(
                pretrained=pretrained,
                fpn=cfg.model.use_fpn,
                fpn_features=cfg.model.num_fpn_features,
                debug=False,
            )
        elif backbone == "ConvNext":
            backbone_model = ConvNext_FPN(
                pretrained=pretrained,
                variant=cfg.model.backbone_variant,
                fpn=cfg.model.use_fpn,
                fpn_features=cfg.model.num_fpn_features,
                debug=False,
            )
        elif backbone == "ViT-Adapter":
            backbone_model = ViTAdapterWrapper(
                pretrained=pretrained,
                variant=cfg.model.backbone_variant,
                mlp_layer=cfg.model.vit_mlp_layer,
                freeze_backbone=cfg.model.freeze_backbone,
            ).to(device=cfg.params.device)
        else:
            raise ValueError(f"Backbone: {backbone} is not supported!")

        self.model = P2PNet(
            backbone=backbone_model,
            img_size=cfg.model.height,
            level=cfg.model.levels[0] if cfg.model.use_fpn else None,
            K=tuple(cfg.model.n_anchors),
            num_cls=len(MAPPING) if len(self.classes) > 1 else 1,
            point_off_w=cfg.model.point_off_w,
            device=cfg.params.device if backbone == "ViT-Adapter" else "cpu",
            n_features_out=cfg.model.num_fpn_features,
        ).to(device=cfg.params.device)
        self.h_p2p = cfg.model.h_p2p

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=cfg.model.lr,
            weight_decay=cfg.model.wd,
        )

        self.scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=self.optimizer,
            warmup_epochs=cfg.model.T_0,
            max_epochs=cfg.dataset.epochs,
            eta_min=cfg.model.eta_min,
        )

        # Log random seeds that were generated and used
        mlflow.log_param("method_seed_used", self.method_seed)
        mlflow.log_param("data_seed_used", self.data_seed)

    def _train(self, epoch: int) -> nn.Module:
        """Trains a single epoch of the model.

        Trains a global model for a single epoch and return the trained model.
        The current loss (averaged across batches in the epoch) and learning
        rates are logged to MLFlow.

        Args:
            epoch: Current epoch of the training.

        Returns:
            Trained model after the current epoch.
            The loss and learning rates are logged to MLFlow.
        """
        self.model.train()
        loss_sum = 0
        loss_reg_sum = 0
        loss_cls_sum = 0
        for img, gt_coords in self.dataloader_train:
            # We use only selected classes.
            # We add the class dimension to the gt points (2+1)
            # [Tensor(N, 3)] * batch_size; N is the number of all GT points
            # in 1 sample with 3 = (x, y, cls_idx).
            # TODO: Implement this in the dataloader instead!
            gt_coords = [
                torch.cat(
                    [
                        (
                            torch.from_numpy(
                                np.hstack(
                                    (
                                        gt_coord[gt_idx],
                                        np.full((gt_coord[gt_idx].shape[0], 1), gt_idx),
                                    )
                                )
                            )
                            if gt_coord[gt_idx].shape[0] > 0
                            else torch.empty(0, 3)
                        )
                        # cls
                        for gt_idx in list(self.classes.keys())
                    ]
                ).to(self.cfg.params.device, dtype=torch.long)
                # batch
                for gt_coord in gt_coords
            ]
            img = img.to(self.cfg.params.device)

            # reg_out_*: (B, fH * fW * K, 2)
            # cls_out_*: (B,fH * fW * K, num_cls)
            (
                reg_out_1_1_org,
                cls_out_1_1_org,
                reg_out_1_1,
                cls_out_1_1,
                reg_out_1_n,
                cls_out_1_n,
            ) = self.model(img)

            loss_type = self.cfg.model.loss_type
            # Computes matches and losses for the 1:n - if enabled
            if self.h_p2p:
                matches_1_1: list[tuple[np.ndarray]] = P2PNet.match_targets(
                    reg_out_1_1,
                    cls_out_1_1,
                    gt_coords,
                    self.cfg.model.matching_p_w,
                    self.cfg.model.matching_c_w,
                    n_matches=1,
                )
                matches_1_n: list[tuple[np.ndarray]] = P2PNet.match_targets(
                    reg_out_1_n,
                    cls_out_1_n,
                    gt_coords,
                    self.cfg.model.matching_p_w,
                    self.cfg.model.matching_c_w,
                    n_matches=self.cfg.model.matching_n,
                )
                reg_loss_1_1 = P2PNet.reg_loss(
                    reg_out_1_1, gt_coords, matches_1_1, self.cfg.params.device
                )
                reg_loss_1_n = P2PNet.reg_loss(
                    reg_out_1_n, gt_coords, matches_1_n, self.cfg.params.device
                )

                if loss_type == "ce_focal":
                    cls_loss_1_1 = P2PNet.cls_loss_focal_softmax(
                        cls_out_1_1,
                        matches_1_1,
                        gt_coords,
                        self.cfg.model.gamma,
                        self.cfg.model.alpha,
                        self.cfg.params.device,
                    )
                    cls_loss_1_n = P2PNet.cls_loss_focal_softmax(
                        cls_out_1_n,
                        matches_1_n,
                        gt_coords,
                        self.cfg.model.gamma,
                        self.cfg.model.alpha,
                        self.cfg.params.device,
                    )
                elif loss_type == "ce":
                    cls_loss_1_1 = P2PNet.cls_loss_ce(
                        cls_out_1_1,
                        matches_1_1,
                        gt_coords,
                        np.array(self.cfg.dataset.loss_cls_w),
                        self.cfg.params.device,
                    )
                    cls_loss_1_n = P2PNet.cls_loss_ce(
                        cls_out_1_n,
                        matches_1_n,
                        gt_coords,
                        np.array(self.cfg.dataset.loss_cls_w),
                        self.cfg.params.device,
                    )
                elif loss_type == "sig_focal":
                    cls_loss_1_1 = P2PNet.cls_loss_focal_sigmoid(
                        cls_out_1_1,
                        matches_1_1,
                        gt_coords,
                        self.cfg.model.gamma,
                        self.cfg.model.alpha,
                        self.cfg.params.device,
                    )
                    cls_loss_1_n = P2PNet.cls_loss_focal_sigmoid(
                        cls_out_1_n,
                        matches_1_n,
                        gt_coords,
                        self.cfg.model.gamma,
                        self.cfg.model.alpha,
                        self.cfg.params.device,
                    )

                loss_1_1 = (
                    self.cfg.model.loss_total_w_cls * cls_loss_1_1
                    + self.cfg.model.loss_total_w_reg * reg_loss_1_1
                ) * self.cfg.model.loss_scale
                loss_1_n = (
                    self.cfg.model.loss_total_w_cls * cls_loss_1_n
                    + self.cfg.model.loss_total_w_reg * reg_loss_1_n
                ) * self.cfg.model.loss_scale

                loss = self.cfg.model.loss_total_1_n * loss_1_n + loss_1_1
                reg_loss = self.cfg.model.loss_total_1_n * reg_loss_1_n + reg_loss_1_1
                cls_loss = self.cfg.model.loss_total_1_n * cls_loss_1_n + cls_loss_1_1
            else:
                # Computes 1:1 matches (default)
                matches_1_1: list[tuple[np.ndarray]] = P2PNet.match_targets(
                    reg_out_1_1_org,
                    cls_out_1_1_org,
                    gt_coords,
                    self.cfg.model.matching_p_w,
                    self.cfg.model.matching_c_w,
                    n_matches=1,
                )

                reg_loss = P2PNet.reg_loss(
                    reg_out_1_1_org, gt_coords, matches_1_1, self.cfg.params.device
                )

                # Computes losses for 1:1 matching (default)
                if loss_type == "ce_focal":
                    cls_loss = P2PNet.cls_loss_focal_softmax(
                        cls_out_1_1_org,
                        matches_1_1,
                        gt_coords,
                        self.cfg.model.gamma,
                        self.cfg.model.alpha,
                        self.cfg.params.device,
                    )
                elif loss_type == "ce":
                    cls_loss = P2PNet.cls_loss_ce(
                        cls_out_1_1_org,
                        matches_1_1,
                        gt_coords,
                        np.array(self.cfg.dataset.loss_cls_w),
                        self.cfg.params.device,
                    )
                elif loss_type == "sig_focal":
                    cls_loss = P2PNet.cls_loss_focal_sigmoid(
                        cls_out_1_1_org,
                        matches_1_1,
                        gt_coords,
                        self.cfg.model.gamma,
                        self.cfg.model.alpha,
                        self.cfg.params.device,
                    )

                loss = (
                    self.cfg.model.loss_total_w_cls * cls_loss
                    + self.cfg.model.loss_total_w_reg * reg_loss
                ) * self.cfg.model.loss_scale

            loss_sum += loss.item()
            loss_reg_sum += reg_loss.item()
            loss_cls_sum += cls_loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.scheduler.step()

        # Log the per-epoch averaged combined and per-class losses.
        # Note: Currently only single-class version supported.
        mlflow.log_metric("Loss", loss_sum / len(self.dataloader_train), step=epoch)
        mlflow.log_metric(
            "Loss_reg", loss_reg_sum / len(self.dataloader_train), step=epoch
        )
        mlflow.log_metric(
            "Loss_cls", loss_cls_sum / len(self.dataloader_train), step=epoch
        )
        mlflow.log_metric("LR", self.optimizer.param_groups[0]["lr"], step=epoch)

        return self.model

    def validate(
        self, epoch: int, n_epoch_validate: int = 1, save_patches: bool = False
    ) -> Metrics:
        """Implements validation phase.

        Args:
            epoch:
                Current epoch of the training.
            n_epoch_validate:
                How many epochs of random patch sampling and evaluations
                to perform (defaults to 1).
            save_patches:
                Saves epoch 0 results to epoch/val/val_image_X.png (default: False)

        Returns:
            Average metrics (MAE, Pr, Re, F1) are returned and logged to MLFlow.
            Global average is returned (classes are combined).
            Predicted patches from epoch 0 are saved (optional).
        """
        logger.info(
            f"Started evaluation (VAL) at epoch: {epoch} "
            f"for {n_epoch_validate} epochs (random crop)."
        )

        assert self.dataloader_val.batch_size == 1, (
            "Validation DataLoader batch size should be 1 and not "
            f"{self.dataloader_val.batch_size}!"
        )

        self.model.eval()
        num_cls = len(self.classes)
        metrics_cls_all = defaultdict(list)
        pbar = tqdm(
            range(1, n_epoch_validate + 1), desc="epochs (VAL)", leave=True, position=0
        )
        time_total = 0
        n_samples = 0
        for epoch_eval in pbar:
            for idx, (img, gt_coords) in enumerate(self.dataloader_val):
                img = img.to(self.cfg.params.device)
                # reg_out_*: (B, fH * fW * K, 2)
                # cls_out_*: (B,fH * fW * K, num_cls)
                start_time = time.time()
                (
                    reg_out_1_1_org,
                    cls_out_1_1_org,
                    reg_out_1_1,
                    cls_out_1_1,
                    _,
                    _,
                ) = self.model(img)
                time_total += time.time() - start_time
                n_samples += 1
                reg_out = reg_out_1_1 if self.h_p2p else reg_out_1_1_org
                cls_out = cls_out_1_1 if self.h_p2p else cls_out_1_1_org

                img = self.invTrans(img[0]).permute(1, 2, 0).cpu().detach().numpy()
                # Transform to (M, 2) - batch size fixed to 1
                reg_out = torch.squeeze(reg_out).cpu().type(torch.long).detach().numpy()
                cls_out = torch.squeeze(cls_out).cpu().detach().softmax(-1).numpy()
                # Get predictions for the individual classes
                for class_idx in self.classes.keys():
                    gt_coords_c = gt_coords[0][class_idx]

                    # Evaluate only on non-empty GT samples
                    # if gt_coords_c.shape[0] == 0:
                    #    continue

                    p_peaks, p_conf, _ = get_peaks(
                        reg_out,
                        cls_out,
                        class_idx if num_cls > 1 else 1,
                        (
                            self.cfg.dataset.det_thresh
                            if isinstance(self.cfg.dataset.det_thresh, float)
                            else self.cfg.dataset.det_thresh[class_idx - 1]
                        ),
                    )
                    if self.cfg.dataset.hungarian_bool:
                        metrics_c, row_ind, col_ind = compute_image_metrics(
                            gt_coords_c, p_peaks, self.cfg.dataset.d_test_gt
                        )
                    else:
                        metrics_c, row_ind, col_ind = compute_image_metrics_dist(
                            gt_coords_c, p_peaks, self.cfg.dataset.d_test_gt
                        )
                    metrics_cls_all[class_idx].append(metrics_c)

                    if epoch_eval == 1 and save_patches:
                        class_name = self.classes[class_idx]
                        save_predicted_images(
                            img=img,
                            gt_coords=gt_coords_c,
                            pred_peaks=p_peaks,
                            pred_peaks_c=p_conf,
                            ref_anchors=self.model.img_anchors.cpu().numpy(),
                            pred_anchors=reg_out,
                            pred_conf=cls_out[..., class_idx if num_cls > 1 else 1],
                            dist=self.cfg.dataset.d_test_gt,
                            epoch=epoch,
                            filename=f"val_image_{idx}_{class_name}",
                            test=False,
                            draw_gt=True,
                            metrics=metrics_cls_all[class_idx][-1],
                            matches=(row_ind, col_ind),
                            draw_matches=True,
                        )

        time_sample = time_total / n_samples
        mlflow.log_metrics({"AVG_T_INF": time_sample}, step=epoch)

        metrics_all = []
        # Note: defaultdict is empty by default until first append
        if len(metrics_cls_all) > 0:
            for class_idx, metrics_cls in metrics_cls_all.items():
                metrics_reduced = compute_f1_from_metrics(metrics_cls)
                metrics_all.append(metrics_reduced)
                cls_name = self.classes[class_idx]
                mlflow.log_metrics(
                    {
                        f"MAE_VAL_{cls_name}": metrics_reduced.mae,
                        f"Precision_VAL_{cls_name}": metrics_reduced.precision,
                        f"Recall_VAL_{cls_name}": metrics_reduced.recall,
                        f"F1_VAL_{cls_name}": metrics_reduced.f1,
                    },
                    step=epoch,
                )
        else:
            logger.info("Empty predictions! Cannot compute metrics!")

        metrics_reduced = reduce_metrics(metrics_all) if metrics_all else Metrics()
        logger.info(
            f"METRICS-VAL (AVG - {n_epoch_validate} epochs): "
            f"MAE: {metrics_reduced.mae}, "
            f"Precision: {metrics_reduced.precision}, "
            f"Recall: {metrics_reduced.recall}, "
            f"F1: {metrics_reduced.f1}"
        )

        return metrics_reduced

    def test(
        self, epoch: int, save_images: bool = True, save_patches: bool = False
    ) -> Metrics:
        """Implements test phase.

        Args:
            epoch:
                Current epoch of the training.
            save_images:
                Save resulting full resolution images into:
                epoch/test/test_image_X.png (default: True)
            save_patches:
                Save resulting patches (all) into:
                epoch/test/test_image_X_patchIDX.png (default: False).
                Note: Ground truth coordinates are not visualized!

        Returns:
            verage metrics (MAE, Pr, Re, F1) are returned and logged to MLFlow.
            Global average is returned (classes are combined).
            Predicted full images are saved (optional).
        """
        logger.info(f"Started evaluation (TEST) at epoch: {epoch}.")

        assert self.dataloader_test.batch_size == 1, (
            "Validation DataLoader batch size should be 1 and not "
            f"{self.dataloader_val.batch_size}!"
        )

        self.model.eval()
        num_cls = len(self.classes)
        metrics_cls_all = defaultdict(list)
        pbar = tqdm(self.dataloader_test, desc="Images (TEST)", leave=True, position=0)
        for idx, (img, gt_coords) in enumerate(pbar):
            img = img.to(self.cfg.params.device)

            # Get patches (e.g., 50% overlap)
            # [b=1, n_p_r, n_p_c, c=3, kernel, kernel]
            patches, _, pad = extract_image_patches(
                x=img,
                kernel=self.cfg.model.height,
                stride=self.cfg.dataset.test_stride,
                pad_value=1.0,
            )

            peaks_cls_all = defaultdict(lambda: np.zeros((0, 2)))
            peaks_cls_all_conf = defaultdict(lambda: np.zeros(0))
            for row in range(patches.shape[1]):
                for col in range(patches.shape[2]):
                    patch = patches[0, row, col]
                    # reg_out_*: (B, fH * fW * K, 2)
                    # cls_out_*: (B,fH * fW * K, num_cls)
                    (
                        reg_out_p_1_1_org,
                        cls_out_p_1_1_org,
                        reg_out_p_1_1,
                        cls_out_p_1_1,
                        _,
                        _,
                    ) = self.model(patch.unsqueeze(0))
                    reg_out_p = reg_out_p_1_1 if self.h_p2p else reg_out_p_1_1_org
                    cls_out_p = cls_out_p_1_1 if self.h_p2p else cls_out_p_1_1_org
                    # Transform to (M, 2) - batch size fixed to 1
                    # Index 0 is used to get 1:1 matchings only
                    reg_out_p = (
                        torch.squeeze(reg_out_p).cpu().type(torch.long).detach().numpy()
                    )
                    cls_out_p = (
                        torch.squeeze(cls_out_p).cpu().detach().softmax(-1).numpy()
                    )
                    patch_np = (
                        self.invTrans(patch).permute(1, 2, 0).cpu().detach().numpy()
                    )
                    # Get predictions for the individual classes
                    for class_idx in self.classes.keys():
                        p_peaks, p_conf, _ = get_peaks(
                            reg_out_p,
                            cls_out_p,
                            class_idx if num_cls > 1 else 1,
                            (
                                self.cfg.dataset.det_thresh
                                if isinstance(self.cfg.dataset.det_thresh, float)
                                else self.cfg.dataset.det_thresh[class_idx - 1]
                            ),
                        )
                        peaks_full_p = local_to_global_peaks(
                            p_peaks, (col, row), self.cfg.dataset.test_stride, pad
                        )
                        peaks_cls_all[class_idx] = np.append(
                            peaks_cls_all[class_idx], peaks_full_p, axis=0
                        )
                        peaks_cls_all_conf[class_idx] = np.append(
                            peaks_cls_all_conf[class_idx], p_conf, axis=0
                        )

                        if save_patches:
                            class_name = self.classes[class_idx]
                            save_predicted_images(
                                img=patch_np,
                                gt_coords=np.zeros((0, 2)),
                                pred_peaks=p_peaks,
                                pred_peaks_c=p_conf,
                                ref_anchors=self.model.img_anchors.cpu().numpy(),
                                pred_anchors=reg_out_p,
                                pred_conf=cls_out_p[
                                    ..., class_idx if num_cls > 1 else 1
                                ],
                                dist=self.cfg.dataset.d_test_gt,
                                epoch=epoch,
                                filename=(
                                    f"test_image_{idx}_patch_{row}_{col}_{class_name}"
                                ),
                                test=True,
                                draw_gt=False,
                            )

            img_np = self.invTrans(img[0]).permute(1, 2, 0).cpu().detach().numpy()
            for class_idx in self.classes.keys():
                gt_coords_c = gt_coords[0][class_idx]

                # Evaluate only on non-empty GT samples
                # if gt_coords_c.shape[0] == 0:
                #    continue

                if self.cfg.dataset.hungarian_bool:
                    metrics_c, row_ind, col_ind = compute_image_metrics(
                        gt_coords_c,
                        peaks_cls_all[class_idx],
                        self.cfg.dataset.d_test_gt,
                    )
                else:
                    metrics_c, row_ind, col_ind = compute_image_metrics_dist(
                        gt_coords_c,
                        peaks_cls_all[class_idx],
                        self.cfg.dataset.d_test_gt,
                    )
                metrics_cls_all[class_idx].append(metrics_c)
                # class_name = self.classes[class_idx]
                # logger.info(
                #    f"F1 score for img {idx}, class {class_name}:"
                #    f" {metrics_cls_all[class_idx][-1].f1}"
                # )

                if save_images:
                    # Plot a separate image for each class
                    class_name = self.classes[class_idx]
                    save_predicted_images(
                        img=img_np,
                        gt_coords=gt_coords_c,
                        pred_peaks=peaks_cls_all[class_idx],
                        pred_peaks_c=peaks_cls_all_conf[class_idx],
                        ref_anchors=np.zeros((0, 2)),
                        pred_anchors=np.zeros((0, 2)),
                        pred_conf=np.zeros((0, 2)),
                        dist=self.cfg.dataset.d_test_gt,
                        epoch=epoch,
                        filename=f"test_image_i{idx}_{class_name}",
                        test=True,
                        draw_gt=False,
                        metrics=metrics_cls_all[class_idx][-1],
                        matches=(row_ind, col_ind),
                        draw_matches=True,
                    )

            if save_images:
                # Plot all classes in the same image
                save_predicted_images_all(
                    img=img_np,
                    gt_peaks=gt_coords[0],
                    pred_peaks=peaks_cls_all,
                    dist=self.cfg.dataset.d_test_gt,
                    epoch=epoch,
                    filename=f"test_image_i{idx}_e{epoch}_all",
                    color_classes=COLORS,
                )

        metrics_all = []
        if len(metrics_cls_all) > 0:
            for class_idx, metrics_cls in metrics_cls_all.items():
                metrics_reduced = compute_f1_from_metrics(metrics_cls)
                metrics_all.append(metrics_reduced)
                cls_name = self.classes[class_idx]
                mlflow.log_metrics(
                    {
                        f"MAE_TEST_{cls_name}": metrics_reduced.mae,
                        f"Precision_TEST_{cls_name}": metrics_reduced.precision,
                        f"Recall_TEST_{cls_name}": metrics_reduced.recall,
                        f"F1_TEST_{cls_name}": metrics_reduced.f1,
                    },
                    step=epoch,
                )
        else:
            logger.info("Empty predictions! Cannot compute metrics!")

        metrics_reduced = reduce_metrics(metrics_all) if metrics_all else Metrics()
        logger.info(
            "METRICS-TEST: "
            f"MAE: {metrics_reduced.mae}, "
            f"Precision: {metrics_reduced.precision}, "
            f"Recall: {metrics_reduced.recall}, "
            f"F1: {metrics_reduced.f1}"
        )

        return metrics_reduced

    def _save_checkpoint(self, epoch: int) -> None:
        """Saves curent model state.

        Args:
            epoch: Current epoch of the training.

        Returns:
            Current model state to epochs/{epoch}/model_{epoch}.pth.
        """
        Path(f"epochs/{epoch}").mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving checkpoint at epoch: {epoch}.")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            },
            f"epochs/{epoch}/model_{epoch}.pth",
        )

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Loads the model from a checkpoint.

        Args:
            checkpoint_path: Path to the saved checkpoint (*.pth)
        """
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.start_epoch = checkpoint["epoch"]


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> float:
    crc_runner = AIKiNETMultipleRunner(
        cfg=cfg,
        save_checkpoint=False,
        backbone=cfg.model.backbone,
        pretrained=True,
        method_seed=cfg.model.method_seed,
        data_seed=cfg.dataset.data_seed,
    )

    crc_runner._train(0)


if __name__ == "__main__":
    main()
