"""Implements helper functions to implement the P2P networks."""

from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def generate_anchors_patch(stride: int = 8, K: tuple[int, int] = (2, 2)) -> np.ndarray:
    """Generates anchor points for the stride patch.

    Generates locations of the anchor points in a patch of size
    stride x stride with the origin in the center of the patch.

    Implementation closely follows the original P2PNet implementation.
    https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet

    Args:
        stride:
            Network stride (defaults to 8).
        K:
            Number of a reference points (row, col) in a patch of size
            stride x stride (defaults to (2, 2)).

    Returns:
        (K, 2) anchor point offsets from the center of the patch of size stride.
    """
    row, line = K
    row_step = stride / row
    line_step = stride / line

    shift_x = (np.arange(1, line + 1) - 0.5) * line_step - stride / 2
    shift_y = (np.arange(1, row + 1) - 0.5) * row_step - stride / 2
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    anchor_points = np.vstack((shift_x.ravel(), shift_y.ravel())).transpose()

    return anchor_points


def generate_anchors_image(
    f_shape: Tuple[int, int], stride: int, anchor_points: np.ndarray
) -> np.ndarray:
    """Generates anchor points for the input image patch.

    Implementation closely follows the original P2PNet implementation.
    https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet

    Args:
        f_shape:
            (H, W) feature map shape.
        stride:
            Network stride (defaults to 8).
        anchor_points:
            (K, 2) anchor point offsets from the center of the patch
            of size stride.

    Returns:
       (K * H * W, 2) anchor points on the input image patch.
    """
    shift_x = (np.arange(0, f_shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, f_shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((shift_x.ravel(), shift_y.ravel())).transpose()

    A = anchor_points.shape[0]
    K = shifts.shape[0]
    # broadcasting
    all_anchor_points = anchor_points.reshape((1, A, 2)) + shifts.reshape(
        (1, K, 2)
    ).transpose((1, 0, 2))
    all_anchor_points = all_anchor_points.reshape((K * A, 2))

    return all_anchor_points


def generate_anchors(
    f_shape: tuple[int, int], stride: int = 8, K: tuple[int, int] = [2, 2]
) -> np.ndarray:
    """Wrapper to generate the anchors on the input image.

    Args:
        f_shape:
            Feature map dimensions (H, W).
        stride:
            Network stride (defaults to 8).
        K:
            Number of reference points (row, col) in a patch of size
            stride x stride (defaults to (2, 2)).

    Returns:
        (K * H * W, 2) anchor points on the input image patch.
    """
    anchors_patch = generate_anchors_patch(stride, K)

    return generate_anchors_image(f_shape, stride, anchors_patch)


def visualize_anchors(img: np.ndarray, anchors: np.ndarray, filename: str) -> None:
    """Visualizes anchors on a given image.

    Args:
        img: Input image (H, W, C).
        anchors: Anchors (M, 2).
        filename: Filename to be save to ({filename}.png)
    """
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(img)
    fig.get_axes()[0].plot(anchors[:, 1], anchors[:, 0], "r*")
    plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
    plt.tight_layout()
    plt.savefig(f"{filename}.png")
    plt.close()


def compute_stride(
    backbone: nn.Module,
    img_size: int = 224,
    level: Union[int, None] = 2,
    device: str = "cpu",
) -> int:
    """Computes backbone stride.

    Args:
        backbone: Backbone to be used for feature extraction (e.g., VGG-16).
        img_size: Input image patch size (defaults to 224).
        level: Level of the FPN to use - 0-indexed (defaults to 2).
        device: Torch device to be used for the dummy input.

    Returns:
        int: Stride factor.
    """
    dummy_input = torch.rand(
        (1, 3, img_size, img_size), dtype=torch.float, device=device
    )
    out = backbone(dummy_input)
    if level is not None:
        out = out[level]

    return int(img_size / out.shape[2])
