# Copyright (c) Shanghai AI Lab. All rights reserved.
import math
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from timm.models.layers import trunc_normal_
from torch.nn.init import normal_

from p2p.models.backbones.utils.adapter_modules import (
    InteractionBlock,
    SpatialPriorModule,
    deform_inputs,
)
from p2p.models.backbones.utils.ms_deform_attn import MSDeformAttn
from p2p.models.backbones.utils.vit import TIMMVisionTransformer


class ViTAdapter(TIMMVisionTransformer):
    def __init__(
        self,
        pretrain_size=224,
        num_heads=12,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=6,
        init_values=0.0,
        interaction_indexes=None,
        with_cffn=True,
        cffn_ratio=0.25,
        deform_ratio=1.0,
        add_vit_feature=True,
        pretrained=None,
        use_extra_extractor=True,
        with_cp=False,
        *args,
        **kwargs,
    ):
        super().__init__(
            num_heads=num_heads, pretrained=pretrained, with_cp=with_cp, *args, **kwargs
        )

        # self.num_classes = 80
        self.cls_token = None
        self.num_block = len(self.blocks)
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        embed_dim = self.embed_dim

        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(
            inplanes=conv_inplane, embed_dim=embed_dim, with_cp=False
        )
        self.interactions = nn.Sequential(
            *[
                InteractionBlock(
                    dim=embed_dim,
                    num_heads=deform_num_heads,
                    n_points=n_points,
                    init_values=init_values,
                    drop_path=self.drop_path_rate,
                    norm_layer=self.norm_layer,
                    with_cffn=with_cffn,
                    cffn_ratio=cffn_ratio,
                    deform_ratio=deform_ratio,
                    extra_extractor=(
                        (True if i == len(interaction_indexes) - 1 else False)
                        and use_extra_extractor
                    ),
                    with_cp=with_cp,
                )
                for i in range(len(interaction_indexes))
            ]
        )
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        # self.norm1 = nn.SyncBatchNorm(embed_dim)
        # self.norm2 = nn.SyncBatchNorm(embed_dim)
        # self.norm3 = nn.SyncBatchNorm(embed_dim)
        # self.norm4 = nn.SyncBatchNorm(embed_dim)
        self.norm1 = nn.BatchNorm2d(embed_dim)
        self.norm2 = nn.BatchNorm2d(embed_dim)
        self.norm3 = nn.BatchNorm2d(embed_dim)
        self.norm4 = nn.BatchNorm2d(embed_dim)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1, self.pretrain_size[0] // 16, self.pretrain_size[1] // 16, -1
        ).permute(0, 3, 1, 2)
        pos_embed = (
            F.interpolate(pos_embed, size=(H, W), mode="bicubic", align_corners=False)
            .reshape(1, -1, H * W)
            .permute(0, 2, 1)
        )
        return pos_embed

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x):
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        x, H, W = self.patch_embed(x)
        bs, n, dim = x.shape
        pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H, W)
        x = self.pos_drop(x + pos_embed)

        # Interaction
        outs = list()
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(
                x,
                c,
                self.blocks[indexes[0] : indexes[-1] + 1],
                deform_inputs1,
                deform_inputs2,
                H,
                W,
            )
            outs.append(x.transpose(1, 2).view(bs, dim, H, W).contiguous())

        # Split & Reshape
        c2 = c[:, 0 : c2.size(1), :]
        c3 = c[:, c2.size(1) : c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1) :, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            x1 = F.interpolate(x1, scale_factor=4, mode="bilinear", align_corners=False)
            x2 = F.interpolate(x2, scale_factor=2, mode="bilinear", align_corners=False)
            x4 = F.interpolate(
                x4, scale_factor=0.5, mode="bilinear", align_corners=False
            )
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        return [f1, f2, f3, f4]

    @property
    def get_n_features(self):
        """Getter for the number of the output features."""
        return self.embed_dim


class ViTAdapterWrapper(ViTAdapter):
    def __init__(
        self,
        variant: str = "B",
        pretrained: Union[str, Path] = "",
        mlp_layer: bool = True,
        freeze_backbone: bool = False,
    ):
        if not isinstance(pretrained, bool) and not isinstance(pretrained, str):
            raise ValueError(
                "Pretrained value must be Boolean (Random/ImageNet) or String (path)!"
            )
        if variant == "B":
            model_args = dict(
                img_size=224,
                patch_size=16,
                embed_dim=768,
                depth=12,
                num_heads=12,
                drop_path_rate=0.3,
                deform_ratio=0.5,
                deform_num_heads=12,
                mlp_layer=mlp_layer,
                freeze_backbone=freeze_backbone,
                layer_scale=True,
            )
            interaction_indexes = [[0, 2], [3, 5], [6, 8], [9, 11]]
            window_attn = [False] * 12
            window_size = [None] * 12
            if isinstance(pretrained, bool) and pretrained is True:
                pretrained = "https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth"  # noqa
        elif variant == "S":
            model_args = dict(
                img_size=224,
                patch_size=16,
                embed_dim=384,
                depth=12,
                num_heads=6,
                drop_path_rate=0.2,
                deform_ratio=1.0,
                deform_num_heads=6,
                mlp_layer=mlp_layer,
                freeze_backbone=freeze_backbone,
                layer_scale=True,
            )
            interaction_indexes = [[0, 2], [3, 5], [6, 8], [9, 11]]
            window_attn = [False] * 12
            window_size = [None] * 12
            if isinstance(pretrained, bool) and pretrained is True:
                pretrained = "https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth"  # noqa
        elif variant == "T":
            model_args = dict(
                img_size=224,
                patch_size=16,
                embed_dim=192,
                depth=12,
                num_heads=3,
                drop_path_rate=0.1,
                deform_ratio=1.0,
                deform_num_heads=6,
                mlp_layer=mlp_layer,
                freeze_backbone=freeze_backbone,
                layer_scale=True,
            )
            interaction_indexes = [[0, 2], [3, 5], [6, 8], [9, 11]]
            window_attn = [False] * 12
            window_size = [None] * 12
            if isinstance(pretrained, bool) and pretrained is True:
                pretrained = "https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth"  # noqa
        elif variant == "L":
            model_args = dict(
                img_size=224,
                patch_size=16,
                embed_dim=1024,
                depth=24,
                num_heads=16,
                drop_path_rate=0.4,
                deform_ratio=0.5,
                deform_num_heads=16,
                mlp_layer=mlp_layer,
                freeze_backbone=freeze_backbone,
                layer_scale=False,
            )
            interaction_indexes = [[0, 5], [6, 11], [12, 17], [18, 23]]
            window_attn = [False] * 24
            window_size = [None] * 24
            if isinstance(pretrained, bool) and pretrained is True:
                pretrained = "https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth"  # noqa
            else:
                # UNI-pretrained model did use LayerScale
                model_args["layer_scale"] = True
        else:
            raise ValueError(f"ViT backbone variant: {variant} is not supported!")

        super().__init__(
            mlp_ratio=4.0,
            conv_inplane=64,
            n_points=4,
            cffn_ratio=0.25,
            interaction_indexes=interaction_indexes,
            window_attn=window_attn,
            window_size=window_size,
            pretrained=pretrained,
            **model_args,
        )


if __name__ == "__main__":
    logger.info("Running demo code in ViT-Adapter model...")

    import os

    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    rank = 0
    dist.init_process_group("nccl", rank=rank, world_size=1)

    pretrained = True
    # pretrained = "/ceph/hpc/data/MFIP/histai_hibou/hibou-b_14to16.pth"  # noqa
    pretrained = "/ceph/hpc/data/MFIP/uni/dinov2_uni_mass100k_vit_large_patch16_14to16.pth"  # noqa
    vit_adapter = ViTAdapterWrapper(
        variant="L",
        pretrained=pretrained,
        mlp_layer=True,
        freeze_backbone=True,
    ).to(rank)
    ddp_model = DDP(vit_adapter, device_ids=[rank])

    dummy_input = torch.zeros((32, 3, 224, 224), dtype=torch.float, device=rank)
    f1, f2, f3, f4 = ddp_model(dummy_input)

    logger.info(f"f1 shape: {f1.shape}")
    logger.info(f"f2 shape: {f2.shape}")
    logger.info(f"f3 shape: {f3.shape}")
    logger.info(f"f4 shape: {f4.shape}")

    dist.destroy_process_group()
