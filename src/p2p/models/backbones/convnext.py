"""Implements ConvNexts with FPN"""

from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
from loguru import logger

import p2p.models.backbones.utils.convnext_official as models


class ConvNext_FPN(nn.Module):
    def __init__(
        self,
        variant: str = "T",
        pretrained: Union[bool, str, Path] = True,
        fpn: bool = True,
        fpn_features: int = 256,
        debug: bool = False,
    ) -> None:
        """Implements ConvNext backbone with optional use of FPN.

        PyTorch ConvNext implementation:
        https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py
        (1) ConvNext: https://arxiv.org/pdf/2201.03545.pdf
        (2) FPN: https://arxiv.org/pdf/1612.03144.pdf

        Args:
            pretrained:
                Bool: Whether to use ImageNet pretrained weights (default=False).
                Str/Path: Path to the pretrained model to be used in the backbone.
            variant:
                ConvNext configuration - "T", "S", "B", "L".
            fpn:
                Use of Feature Pyramid Networks (defaults to True).
            fpn_features:
                Number of channels used in FPN (defaults to 256).
            debug:
                Print debug information (defaults to False).
        """
        super().__init__()
        self.fpn = fpn
        self.fpn_features = fpn_features
        self.debug = debug

        if variant == "T":
            if isinstance(pretrained, bool):
                self.backbone = models.convnext_tiny(pretrained=pretrained)
            elif isinstance(pretrained, str) or isinstance(pretrained, Path):
                # Pretrained using SparK. Saves the model in "module"
                self.backbone = models.convnext_tiny(pretrained=False)
                self.backbone.load_state_dict(
                    torch.load(pretrained)["module"], strict=False
                )
            else:
                raise ValueError(
                    f"Pretrained type: {type(pretrained)} is not supported!"
                )
            self.blks_nf = [96, 192, 384, 768]
        elif variant == "S":
            if isinstance(pretrained, bool):
                self.fbackbone = models.convnext_small(pretrained=pretrained)
            elif isinstance(pretrained, str) or isinstance(pretrained, Path):
                # Pretrained using SparK. Saves the model in "module".
                self.backbone = models.convnext_small(pretrained=False)
                self.fbackbone.load_state_dict(
                    torch.load(pretrained)["module"], strict=False
                )
            else:
                raise ValueError(
                    f"Pretrained type: {type(pretrained)} is not supported!"
                )
            self.blks_nf = [96, 192, 384, 768]
        elif variant == "B":
            if isinstance(pretrained, bool):
                self.backbone = models.convnext_base(pretrained=pretrained)
            elif isinstance(pretrained, str) or isinstance(pretrained, Path):
                # Pretrained using SparK. Saves the model in "module".
                self.backbone = models.convnext_base(pretrained=False)
                self.backbone.load_state_dict(
                    torch.load(pretrained)["module"], strict=False
                )
            else:
                raise ValueError(
                    f"Pretrained type: {type(pretrained)} is not supported!"
                )
            self.blks_nf = [128, 256, 512, 1024]
        elif variant == "L":
            if isinstance(pretrained, bool):
                self.backbone = models.convnext_large(pretrained=pretrained)
            elif isinstance(pretrained, str) or isinstance(pretrained, Path):
                # Pretrained using SparK. Saves the model in "module".
                self.backbone = models.convnext_large(pretrained=False)
                self.backbone.load_state_dict(
                    torch.load(pretrained)["module"], strict=False
                )
            else:
                raise ValueError(
                    f"Pretrained type: {type(pretrained)} is not supported!"
                )
            self.blks_nf = [192, 384, 768, 1536]
        else:
            raise ValueError(f"ConvNext variant: {variant} is not supported!")

        # FPN
        self.fpn_blks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Conv2d(
                            self.blks_nf[idx],
                            fpn_features,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                        ),
                        nn.Upsample(scale_factor=2, mode="nearest"),
                        nn.Conv2d(
                            fpn_features,
                            fpn_features,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                    ]
                )
                for idx, _ in enumerate(self.blks_nf)
            ]
        )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, list[torch.Tensor]]:
        """Implements ConvNext inference with optional FPN.

        Args:
            x: Input tensor (img: B, C, H, W).

        Returns:
            Returns the last feature map (B, 512, 7, 7) if fpn = False,
            or list of FPN feature maps (P2-P5) if fpn = True.
        """
        if self.fpn:
            # ConvNext: C2 - C5
            _, outs = self.backbone.forward_features(x)

            # FPN: P2 - P5
            outs_fpn = [self.fpn_blks[-1][0](outs[-1])]
            # Upsampled from the top
            outs_fpn_top = [self.fpn_blks[-1][1](outs_fpn[-1])]
            for idx in reversed(range(len(outs) - 1)):
                outs_fpn.append(self.fpn_blks[idx][0](outs[idx]) + outs_fpn_top[-1])
                outs_fpn_top.append(self.fpn_blks[idx][1](outs_fpn[-1]))
                outs_fpn[-1] = self.fpn_blks[idx][2](outs_fpn[-1])

            return list(reversed(outs_fpn))
        else:
            _, outs = self.backbone.forward_features(x)
            return outs[-1]

    @property
    def get_n_features(self):
        """Getter for the number of the output features."""
        if self.fpn:
            return self.fpn_features
        else:
            return self.blks_nf[-1]


if __name__ == "__main__":
    """Implements inference for one example."""
    # pretrained = True
    pretrained = "/ceph/hpc/data/MFIP/outputs/SparK/16_node_64_GPU_ID_13285816_a_convnext_base_b_4096_e_1600_train/convnext_base_1kpretrained_timm_style.pth"  # noqa
    backbone = ConvNext_FPN(
        pretrained=pretrained, variant="B", fpn=True, fpn_features=256, debug=False
    )
    dummy_input = torch.rand((32, 3, 224, 224), dtype=torch.float)
    out = backbone(dummy_input)

    if backbone.fpn:
        for idx, fpn_out in enumerate(out):
            logger.debug(f"blk{idx + 1}: {fpn_out.shape}")
    else:
        logger.debug(f"Output shape: {out.shape}")
