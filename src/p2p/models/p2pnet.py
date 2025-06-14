"""Implements the vanilla P2PNet"""

import math
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from scipy.optimize import linear_sum_assignment
from torchvision.models.resnet import BasicBlock
from torchvision.ops.focal_loss import sigmoid_focal_loss

from p2p.models.backbones.convnext import ConvNext_FPN  # noqa
from p2p.models.backbones.vgg import VGG16_FPN  # noqa
from p2p.models.backbones.vit_adapter import ViTAdapterWrapper
from p2p.models.utils import compute_stride, generate_anchors, visualize_anchors  # noqa


class P2PNet(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        img_size: int = 224,
        level: Union[int, None] = 3,
        K: tuple[int, int] = (2, 2),
        n_features_out: int = 256,
        point_off_w: int = 100,
        num_cls: int = 1,
        device: str = "cpu",
    ) -> None:
        """Initializes P2PNet.

        Args:
            backbone:
                Backbone to be used for feature extraction (e.g., VGG-16).
            img_size:
                Input image patch size (defaults to 224).
            level:
                Level of the FPN to use (defaults to 3).
                Level is 1-indexed. Vanilla P2PNet implementation uses only
                one level of the FPN (level = 3). None for not using FPN. None
                is only applicable for ConvNet backbones (e.g., VGG, ConvNext).
            K:
                Number of reference points (row, col) in a patch of size
                stride x stride (defaults to [2, 2]).
            n_features_out:
                Number of output features (defaults to 256).
            point_off_w:
                Normalization term for the predicted anchor offsets
                (defaults to 100).
            num_cls:
                Number of predicted classes (+1 is added for the bg class).
            device:
                Torch device to be used. Currently only used in compute_stride.
                Needs to be "cuda:x" for ViT-Adapter backbone.
        """
        super().__init__()
        self.backbone = backbone
        self.img_size = img_size

        # Max 1 level of the FPN is supported currently.
        self.level = level
        if level is not None:
            self.level -= 1

        self.stride = compute_stride(backbone, img_size, self.level, device)
        self.K = K
        self.n_anchors = math.prod(K)
        self.point_off_w = point_off_w
        self.num_cls = num_cls

        # Anchors are currently generated just for one level of the FPN.
        self.register_buffer(
            "img_anchors",
            torch.from_numpy(
                generate_anchors(
                    (img_size // self.stride, img_size // self.stride), self.stride, K
                )
            ).type(torch.float),
        )

        self.prediction_head = PredictionHead(
            n_features_in=backbone.get_n_features,
            n_features_out=n_features_out,
            num_anchors=self.n_anchors,
            # +1: Additional bg class
            num_cls=self.num_cls + 1,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Implements P2PNet inference.

        H, W represent the input img dimensions.
        fH, fW represent the feature map dimensions.

        Args:
            x: Input tensor (img: B, C, H, W).

        Returns:
            Regression head output (1:1 original): (B, fH * fW * K, 2).
            Classification head output (1:1 original): (B,fH * fW * K, num_cls).
            Regression head output (1:1 after 1:n): (B, fH * fW * K, 2).
            Classification head output (1:1 after 1:n): (B, fH * fW * K, num_cls).
            Regression head output (1:n for H-P2P loss): (B, fH * fW * K, 2).
            Classification head output (1:n for H-P2P loss): (B,fH * fW * K, num_cls).
        """
        # The vanilla implementation only uses one level of FPN features.
        features_fpn = self.backbone(x)
        if self.level is not None:
            features_fpn = features_fpn[self.level]
        (
            reg_out_1_1_org,
            cls_out_1_1_org,
            reg_out_1_1,
            cls_out_1_1,
            reg_out_1_n,
            cls_out_1_n,
        ) = self.prediction_head(features_fpn)

        anchors_ref = self.img_anchors.repeat(x.shape[0], 1, 1)
        # logger.debug(torch.mean(torch.abs(self.point_off_w * reg_out)))
        out_coord_1_1_org = anchors_ref + self.point_off_w * reg_out_1_1_org
        out_coord_1_1 = anchors_ref + self.point_off_w * reg_out_1_1
        out_coord_1_n = anchors_ref + self.point_off_w * reg_out_1_n

        return (
            out_coord_1_1_org,
            cls_out_1_1_org,
            out_coord_1_1,
            cls_out_1_1,
            out_coord_1_n,
            cls_out_1_n,
        )

    @staticmethod
    @torch.no_grad()
    def match_targets(
        reg_out: torch.Tensor,
        cls_out: torch.Tensor,
        gt_coords: list[torch.Tensor],
        point_w: float,
        cls_w: float = 1.0,
        n_matches: int = 1,
    ) -> list[tuple[np.ndarray]]:
        """Performs Hungarian matching.

        Args:
            reg_out: Regression head output: (B, fH * fW * K, 2).
            cls_out: Classification head output: (B, fH * fW * K, num_cls + 1).
            gt_coords: List (b_size) of GT coords and classes list[Tensor(N_all, 3)].
            point_w: Pixel distance weight term.
            cls_w: Classification (bg/fg) weight term (defaults to 1).
            n_matches: Number of matches per anchor - HP2P (defaults to 1).

        Returns:
            List of the size of a batch, containing tuples of (idx_i, idx_j),
            where idx_i is the selected prediction for the gt point at idx_j.
        """
        out_points = []
        num_cls = cls_out.shape[-1]
        for b_idx in range(reg_out.shape[0]):
            # N_all x 3 (x, y, cls_idx)
            gt_coords_b = gt_coords[b_idx]

            if len(gt_coords_b) == 0:
                # No gt - nothing to match
                out_points.append((np.array([]), np.array([])))
                continue

            # Distance cost
            # M X N matrix (M = anchors, N = gt points)
            # Currently L2 dist is used (TODO: move to config?)
            dist_cost = torch.cdist(
                reg_out[b_idx], gt_coords_b[:, :2].type(torch.float), p=2
            )

            # Cls cost
            if num_cls > 2:
                # Multi-class setting
                cls_out_norm = -cls_out.softmax(-1)
                cls_cost = cls_out_norm[b_idx, :, gt_coords_b[:, 2]]
            else:
                # Single-class only and cls 1 == fg
                # Dimension extended for broadcasting -> (M, 1)
                # Softmax applied to the last channel
                cls_cost = -cls_out.softmax(-1)[b_idx, :, 1].unsqueeze(dim=1)

            cost_matrix = point_w * dist_cost + cls_w * cls_cost

            # Repeat matrix for n_matches (Hungarian ensemble matching)
            cost_matrix_r = np.repeat(cost_matrix.cpu().numpy(), n_matches, axis=1)
            a_ind, gt_ind = linear_sum_assignment(cost_matrix_r)
            gt_ind = (gt_ind / n_matches).astype(np.int64)
            out_points.append((a_ind, gt_ind))

        return out_points

    @staticmethod
    def reg_loss(
        reg_out: torch.Tensor,
        gt_coords: list[torch.Tensor],
        matches: list[tuple[np.ndarray]],
        device: str = "cuda",
    ) -> torch.Tensor:
        """Computes regression loss.

        MSE loss is computed only on matched points.

        Args:
            reg_out: Regression head output: (B, fH * fW * K, 2).
            gt_coords: List (b_size) of GT coords and classes list[Tensor(N_all, 3)].
            matches: List of batch size of matched indices (idx_i, idx_j).
            device: PyTorch device (defaults to cuda).

        Returns:
            Returns the normalized MSE loss.
        """
        reg_loss_sum = torch.tensor(0, dtype=torch.float, device=device)
        for b_idx in range(reg_out.shape[0]):
            pred_idx, gt_idx = matches[b_idx][0], matches[b_idx][1]
            if len(gt_idx) == 0:
                continue
            reg_curr = reg_out[b_idx, pred_idx, :]
            gt_curr = gt_coords[b_idx][gt_idx, :2].type(torch.float)
            reg_loss_sum += F.mse_loss(reg_curr, gt_curr, reduction="mean")
        reg_loss_sum /= reg_out.shape[0]

        return reg_loss_sum

    @staticmethod
    def cls_loss_ce(
        cls_out: torch.Tensor,
        matches: list[tuple[np.ndarray]],
        gt_coords: list[torch.Tensor],
        loss_cls_w: np.ndarray,
        device: str = "cuda",
    ) -> torch.Tensor:
        """Computes classification loss (CE).

        Args:
            cls_out:
                Classification head output: (B, fH * fW * K, num_cls + 1).
            matches:
                List of batch size of matched indices (idx_i, idx_j).
            gt_coords:
                List (b_size) of GT coords and classes list[Tensor(N_all, 3)].
            loss_cls_w:
                Tensor of size num_cls with the per-class weights.
                Fg class weight is used as a constant for all the fg classes,
                if per class weighyt is not specified.
            device:
                PyTorch device (defaults to cuda).

        Returns:
            Returns the normalized CE loss.
        """
        num_cls = cls_out.shape[-1]
        cls_loss_sum = torch.tensor(0, dtype=torch.float, device=device)
        for b_idx in range(cls_out.shape[0]):
            pred_idx, gt_idx = matches[b_idx][0], matches[b_idx][1]
            targets = torch.zeros((cls_out.shape[1]), dtype=torch.long, device=device)
            # Initialize cls weights with a bg class
            cls_w = torch.full((num_cls,), loss_cls_w[0]).to(device, torch.float)
            if len(pred_idx) > 0:
                if num_cls > 2:
                    targets[pred_idx] = gt_coords[b_idx][gt_idx, 2]
                    if len(loss_cls_w) == 2:
                        # fg weight used for all classes.
                        cls_w = torch.full((num_cls,), loss_cls_w[1]).to(
                            device, torch.float
                        )
                        cls_w[0] = loss_cls_w[0]
                    else:
                        # Per-class weight provided.
                        assert len(loss_cls_w) == num_cls, (
                            "Number of class weights must be equal to the number of"
                            " classes + bg!"
                        )
                        cls_w = torch.from_numpy(loss_cls_w).to(device, torch.float)
                else:
                    targets[pred_idx] = 1
                    cls_w = torch.from_numpy(loss_cls_w).to(device, torch.float)

            cls_loss_sum += F.cross_entropy(
                cls_out[b_idx],
                targets,
                cls_w,
                reduction="mean",
            )
        cls_loss_sum /= cls_out.shape[0]

        return cls_loss_sum

    @staticmethod
    def cls_loss_gce(
        cls_out: torch.Tensor,
        matches: list[tuple[np.ndarray]],
        gt_coords: list[torch.Tensor],
        loss_cls_w: np.ndarray,
        q: float = 0.4,
        device: str = "cuda",
    ) -> torch.Tensor:
        """Computes classification loss (GCE).

        GCE: https://arxiv.org/pdf/1805.07836.pdf
        Used in E2E: https://arxiv.org/pdf/2207.00176.pdf
        Partially followed by:
        https://github.com/AlanChou/Truncated-Loss/blob/master/TruncatedLoss.py

        Args:
            cls_out:
                Classification head output: (B, fH * fW * K, num_cls + 1).
            matches:
                List of batch size of matched indices (idx_i, idx_j).
            gt_coords:
                List (b_size) of GT coords and classes list[Tensor(N_all, 3)].
            loss_cls_w:
                Tensor of size num_cls with the per-class weights.
                Fg class weight is used as a constant for all the fg classes,
                if per class weighyt is not specified.
            q:
                Convergence/noise robustness ratio (0, 1].
            device:
                PyTorch device (defaults to cuda).

        Returns:
            Returns the normalized GCE loss.
        """
        num_cls = cls_out.shape[-1]
        cls_loss_sum = torch.tensor(0, dtype=torch.float, device=device)
        for b_idx in range(cls_out.shape[0]):
            pred_idx, gt_idx = matches[b_idx][0], matches[b_idx][1]
            targets = torch.zeros((cls_out.shape[1]), dtype=torch.long, device=device)
            # Initialize cls weights with a bg class
            cls_w = torch.full((num_cls,), loss_cls_w[0]).to(device, torch.float)
            if len(pred_idx) > 0:
                if num_cls > 2:
                    targets[pred_idx] = gt_coords[b_idx][gt_idx, 2]
                    if len(loss_cls_w) == 2:
                        cls_w = torch.full((num_cls,), loss_cls_w[1]).to(
                            device, torch.float
                        )
                        cls_w[0] = loss_cls_w[0]
                    else:
                        cls_w = torch.from_numpy(loss_cls_w).to(device, torch.float)
                else:
                    targets[pred_idx] = 1
                    cls_w = torch.from_numpy(loss_cls_w).to(device, torch.float)

            p = cls_out[b_idx].softmax(-1)
            Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
            for cls_idx in range(num_cls):
                cls_loss_sum += (
                    torch.mean((1 - Yg[targets == cls_idx] ** q) / q) * cls_w[cls_idx]
                )
            # Add L2 regularization term here for p?
        cls_loss_sum /= cls_out.shape[0]

        return cls_loss_sum

    @staticmethod
    def cls_loss_focal_softmax(
        cls_out: torch.Tensor,
        matches: list[tuple[np.ndarray]],
        gt_coords: list[torch.Tensor],
        gamma: float,
        alpha: float,
        device: str = "cuda",
    ) -> torch.Tensor:
        """Computes classification loss (Focal - Softmax).

        Args:
            cls_out:
                Classification head output: (B, fH * fW * K, num_cls + 1).
            matches:
                List of batch size of matched indices (idx_i, idx_j).
            gt_coords:
                List (b_size) of GT coords and classes list[Tensor(N_all, 3)].
            loss_cls_w:
                Tensor of size num_cls with the per-class weights.
                Currently only bg and fg class weights are used.
                Fg class weight is used as a constant for all the fg classes.
            gamma:
                Exponent of the modulating factor (1 - p_t) to balance
                easy vs hard examples.
            alpha:
                Weighting factor in range (0,1) to balance positive vs.
                negative examples or -1 for ignore.
            device:
                PyTorch device (defaults to cuda).

        Returns:
            Returns the normalized focal loss (softmax).
        """
        num_cls = cls_out.shape[-1]
        cls_loss_sum = torch.tensor(0, dtype=torch.float, device=device)
        for b_idx in range(cls_out.shape[0]):
            pred_idx, gt_idx = matches[b_idx][0], matches[b_idx][1]
            targets = torch.zeros((cls_out.shape[1]), dtype=torch.long, device=device)
            if len(pred_idx) > 0:
                if num_cls > 2:
                    targets[pred_idx] = gt_coords[b_idx][gt_idx, 2]
                else:
                    targets[pred_idx] = 1

            cls_loss_ce = F.cross_entropy(
                cls_out[b_idx],
                targets,
                reduction="none",
            )
            pt = torch.exp(-cls_loss_ce)
            cls_loss_sum += (alpha * (1 - pt) ** gamma * cls_loss_ce).mean()

        cls_loss_sum /= cls_out.shape[0]

        return cls_loss_sum

    @staticmethod
    def cls_loss_focal_sigmoid(
        cls_out: torch.Tensor,
        matches: list[tuple[np.ndarray]],
        gt_coords: list[torch.Tensor],
        gamma: float,
        alpha: float,
        device: str = "cuda",
    ) -> torch.Tensor:
        """Computes classification loss (Focal - Sigmoid).

        Can probably be optimized using one-hot-encoding:
        https://github.com/pytorch/vision/issues/3250#issuecomment-761135092

        Args:
            cls_out:
                Classification head output: (B, fH * fW * K, num_cls + 1).
            matches:
                List of batch size of matched indices (idx_i, idx_j).
            gt_coords:
                List (b_size) of GT coords and classes list[Tensor(N_all, 3)].
            gamma:
                Exponent of the modulating factor (1 - p_t) to balance
                easy vs hard examples.
            alpha:
                Weighting factor in range (0,1) to balance positive vs.
                negative examples or -1 for ignore.
            device:
                PyTorch device (defaults to cuda).

        Returns:
            Returns the normalized focal loss.
        """
        cls_loss_sum = torch.tensor(0, dtype=torch.float, device=device)
        for b_idx in range(cls_out.shape[0]):
            pred_idx, gt_idx = matches[b_idx][0], matches[b_idx][1]
            targets = torch.zeros((cls_out.shape[1]), dtype=torch.long, device=device)
            cls_loss = torch.tensor(0, dtype=torch.float, device=device)
            for c_idx in range(1, cls_out.shape[-1]):
                targets[pred_idx[gt_coords[b_idx][gt_idx, 2] == c_idx]] = 1
                cls_loss += sigmoid_focal_loss(
                    cls_out[b_idx, :, c_idx], targets, gamma, alpha, "mean"
                )
            cls_loss /= cls_out.shape[-1] - 1
            cls_loss_sum += cls_loss
        cls_loss_sum /= cls_out.shape[0]

        return cls_loss_sum


class PredictionHead(nn.Module):
    def __init__(
        self,
        n_features_in: int,
        n_features_out: int = 256,
        num_anchors: int = 4,
        num_cls: int = 2,
    ) -> None:
        """Initializes P2PNet prediction head.

        Architecture of the prediction head partially follows E2E method:
        https://arxiv.org/pdf/2207.00176.pdf

        Args:
            n_features_in:
                Number of input features.
            n_features_out:
                Number of output features (defaults to 256).
            num_anchors:
                Number of anchor points (defaults to 4).
            num_cls:
                Number of classes in the classification head (defaults to 2).
                Classification head predicts if the anchor is fg or bg.
        """
        super().__init__()
        self.K = num_anchors
        self.num_cls = num_cls

        self.regression_base = nn.Sequential(
            BasicBlock(n_features_in, n_features_out),
            BasicBlock(n_features_out, n_features_out),
        )
        self.regression_1_n = nn.Conv2d(
            n_features_out, self.K * 2, kernel_size=3, padding=1
        )
        self.regression_1_1 = nn.Conv2d(
            self.K * 2, self.K * 2, kernel_size=3, padding=1
        )
        self.regression_1_1_org = nn.Conv2d(
            n_features_out, self.K * 2, kernel_size=1, padding=0
        )

        self.classification_base = nn.Sequential(
            BasicBlock(n_features_in, n_features_out),
            BasicBlock(n_features_out, n_features_out),
        )
        self.classification_1_n = nn.Conv2d(
            n_features_out, self.K * num_cls, kernel_size=3, padding=1
        )
        self.classification_1_1 = nn.Conv2d(
            self.K * num_cls, self.K * num_cls, kernel_size=3, padding=1
        )
        self.classification_1_1_org = nn.Conv2d(
            n_features_out, self.K * num_cls, kernel_size=1, padding=0
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Implements P2PNet prediction head inference.

        H, W represent the prediction head input feature map dimensions.
        fH, fW represent the feature map dimensions.

        Args:
            x: Input tensor (feature map: B, C, H, W).

        Returns:
            Regression head output (1:1 original): (B, fH * fW * K, 2).
            Classification head output (1:1 original): (B,fH * fW * K, num_cls).
            Regression head output (1:1 after 1:n): (B, fH * fW * K, 2).
            Classification head output (1:1 after 1:n): (B, fH * fW * K, num_cls).
            Regression head output (1:n for H-P2P loss): (B, fH * fW * K, 2).
            Classification head output (1:n for H-P2P loss): (B,fH * fW * K, num_cls).
        """
        reg_out_base = self.regression_base(x)
        reg_out_1_n = self.regression_1_n(reg_out_base)
        reg_out_1_n_r = nn.ReLU()(reg_out_1_n)
        reg_out_1_1 = self.regression_1_1(reg_out_1_n_r)
        reg_out_1_1_org = self.regression_1_1_org(reg_out_base)

        reg_out_1_n = reg_out_1_n.permute(0, 2, 3, 1)
        reg_out_1_n = reg_out_1_n.contiguous().view(reg_out_1_n.shape[0], -1, 2)
        reg_out_1_1 = reg_out_1_1.permute(0, 2, 3, 1)
        reg_out_1_1 = reg_out_1_1.contiguous().view(reg_out_1_1.shape[0], -1, 2)
        reg_out_1_1_org = reg_out_1_1_org.permute(0, 2, 3, 1)
        reg_out_1_1_org = reg_out_1_1_org.contiguous().view(
            reg_out_1_1_org.shape[0], -1, 2
        )

        cls_out_base = self.classification_base(x)
        cls_out_1_n = self.classification_1_n(cls_out_base)
        cls_out_1_1 = self.classification_1_1(cls_out_1_n)
        cls_out_1_1_org = self.classification_1_1_org(cls_out_base)

        cls_out_1_n = cls_out_1_n.permute(0, 2, 3, 1)
        b_size, h, w, _ = cls_out_1_n.shape
        cls_out_1_n = cls_out_1_n.view(b_size, w, h, self.K, self.num_cls)
        cls_out_1_n = cls_out_1_n.contiguous().view(b_size, -1, self.num_cls)
        cls_out_1_1 = cls_out_1_1.permute(0, 2, 3, 1)
        b_size, h, w, _ = cls_out_1_1.shape
        cls_out_1_1 = cls_out_1_1.view(b_size, w, h, self.K, self.num_cls)
        cls_out_1_1 = cls_out_1_1.contiguous().view(b_size, -1, self.num_cls)
        cls_out_1_1_org = cls_out_1_1_org.permute(0, 2, 3, 1)
        b_size, h, w, _ = cls_out_1_1_org.shape
        cls_out_1_1_org = cls_out_1_1_org.view(b_size, w, h, self.K, self.num_cls)
        cls_out_1_1_org = cls_out_1_1_org.contiguous().view(b_size, -1, self.num_cls)

        return (
            reg_out_1_1_org,
            cls_out_1_1_org,
            reg_out_1_1,
            cls_out_1_1,
            reg_out_1_n,
            cls_out_1_n,
        )


if __name__ == "__main__":
    backbone = ViTAdapterWrapper(variant="B").cuda(0)
    # ViT-Adapter: B = 768, S = 384, T = 192
    # 256 for VGG and ConvNext
    p2pnet = P2PNet(
        backbone=backbone,
        img_size=224,
        level=3,
        K=(2, 2),
        n_features_out=768,
        # device="cuda:0",
    ).cuda(0)
    dummy_input = torch.zeros((32, 3, 224, 224), dtype=torch.float).cuda(0)
    (
        out_coord_1_1_org,
        cls_out_1_1_org,
        out_coord_1_1,
        cls_out_1_1,
        out_coord_1_n,
        cls_out_1_n,
    ) = p2pnet(dummy_input)

    f1, f2, f3, f4 = backbone(dummy_input)

    logger.info(f"f1 shape: {f1.shape}")
    logger.info(f"f2 shape: {f2.shape}")
    logger.info(f"f3 shape: {f3.shape}")
    logger.info(f"f4 shape: {f4.shape}")

    """for idx, feature_fpn in enumerate(features_fpn):
        logger.info(f"Level {idx}: {feature_fpn.shape}")

    logger.debug(f"Anchors shape: {p2pnet.img_anchors.shape}")
    logger.debug(f"Regression head output shape: {reg_out.shape}")
    logger.debug(f"Classification head output shape: {cls_out.shape}")

    visualize_anchors(
        dummy_input.permute(0, 2, 3, 1).cpu().numpy()[0],
        p2pnet.img_anchors,
        "anchors_demo",
    )"""
