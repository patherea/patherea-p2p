"""Implements VGG-16 with FPN"""

from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
import torchvision.models as models
from loguru import logger


class VGG16_FPN(nn.Module):
    def __init__(
        self,
        pretrained: Union[bool, str, Path] = True,
        fpn: bool = True,
        fpn_features: int = 256,
        debug: bool = False,
    ) -> None:
        """Implements VGG-16 backbone with optional use of FPN.

        We are using the default PyTorch implementation that also includes
        batch normalization. This is configuration D from original VGG paper,
        with added BN after the conv layer (conv, BN, relu).

        PyTorch: https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py
        (1) VGG-16: https://arxiv.org/pdf/1409.1556v6.pdf
        (2) FPN: https://arxiv.org/pdf/1612.03144.pdf

        Args:
            pretrained:
                Bool: Whether to use ImageNet pretrained weights (default=False).
                Str/Path: Path to the pretrained model to be used in the backbone.
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

        if isinstance(pretrained, bool):
            backbone = models.vgg16_bn(pretrained=pretrained)
        elif isinstance(pretrained, str) or isinstance(pretrained, Path):
            # We don't use any other pretrained VGG models beyond ImageNet.
            # This code here is for consistency with other architectures.
            backbone = models.vgg16_bn(pretrained=False)
            backbone.load_state_dict(torch.load(pretrained))
        else:
            raise ValueError(f"Pretrained type: {type(pretrained)} is not supported!")
        features = list(backbone.features.children())

        if debug:
            for el in list(enumerate(features)):
                logger.debug(el)

        # Downsample stride - output size (224 x 224 input)
        # 2x downsample - 112 x 112
        self.blk2 = nn.Sequential(*features[:13])
        # 4x downsample - 56 x 56
        self.blk3 = nn.Sequential(*features[13:23])
        # 8x downsample - 28 x 28
        self.blk4 = nn.Sequential(*features[23:33])
        # 16x downsample - 14 x 14
        self.blk5 = nn.Sequential(*features[33:43])
        # 32x downsample - only 1 additional maxpool (7 x 7)
        self.vgg_full = nn.Sequential(*features[:44])

        self.blks = [self.blk2, self.blk3, self.blk4, self.blk5]
        # As per the VGG paper (1), Table 1, configuration D.
        # We skip the first VGG block for the FPN.
        self.blks_nf = [128, 256, 512, 512]

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
                for idx, _ in enumerate(self.blks)
            ]
        )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, list[torch.Tensor]]:
        """Implements VGG inference with optional FPN.

        Args:
            x: Input tensor (img: B, C, H, W).

        Returns:
            Returns the last feature map (B, 512, 7, 7) if fpn = False,
            or list of FPN feature maps (P2-P5) if fpn = True.
        """
        if self.fpn:
            # VGG: C2 - C5
            outs_vgg = [self.blks[0](x)]
            for blk in self.blks[1:]:
                outs_vgg.append(blk(outs_vgg[-1]))

            # FPN: P2 - P5
            outs_fpn = [self.fpn_blks[-1][0](outs_vgg[-1])]
            # Upsampled from the top
            outs_fpn_top = [self.fpn_blks[-1][1](outs_fpn[-1])]
            for idx in reversed(range(len(outs_vgg) - 1)):
                outs_fpn.append(self.fpn_blks[idx][0](outs_vgg[idx]) + outs_fpn_top[-1])
                outs_fpn_top.append(self.fpn_blks[idx][1](outs_fpn[-1]))
                outs_fpn[-1] = self.fpn_blks[idx][2](outs_fpn[-1])

            return list(reversed(outs_fpn))
        else:
            return self.vgg_full(x)

    @property
    def get_n_features(self):
        """Getter for the number of the output features."""
        if self.fpn:
            return self.fpn_features
        else:
            return self.blks_nf[-1]


if __name__ == "__main__":
    """Implements inference for one example."""
    backbone = VGG16_FPN(pretrained=False, fpn=True, fpn_features=256, debug=False)
    dummy_input = torch.rand((32, 3, 224, 224), dtype=torch.float)
    out = backbone(dummy_input)

    if backbone.fpn:
        for idx, fpn_out in enumerate(out):
            logger.debug(f"blk{idx + 1}: {fpn_out.shape}")
    else:
        logger.debug(f"Output shape: {out.shape}")
