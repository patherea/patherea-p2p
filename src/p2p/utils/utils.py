"""Implements helper functions"""

import math
from pathlib import Path
from typing import NamedTuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from numpy import ndarray
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


class Metrics(NamedTuple):
    """Container for image-level performance metrics."""

    n_tp: float = np.nan
    n_fp: float = np.nan
    n_fn: float = np.nan
    precision: float = np.nan
    recall: float = np.nan
    f1: float = np.nan
    mae: float = np.nan


def compute_image_metrics(
    gt_coords: np.ndarray, pred_coords: np.ndarray, thresh: Union[float, ndarray]
) -> tuple[Metrics, tuple[np.ndarray]]:
    """Computes detection evaluation metrics.

    Evaluation implemented using the Hungarian algorithm from (Section 3.4):
    https://arxiv.org/pdf/2001.03360.pdf
    The code is partially taken from the official NWPU evaluation code:
    https://github.com/gjy3035/NWPU-Crowd-Sample-Code-for-Localization

    Important: Hungarian matching must be performed on the Boolean matrix.

    Args:
        gt_coords: N X 2 ndarray of ground truth cell locations (X, Y).
        pred_coords: M X 2 ndarray of predicted cell locations (X, Y).
        thresh: Distance threshold (float) or ndarray of dist. thresholds per GT.

    Returns:
        Tuple of: TP, FP, FN, Precision, Recall, F1, MAE (cell counting) and
        tuple of col_ind (GTs) and row_ind (preds) from the Hungarian matching.
    """
    # Check for special conditions
    if gt_coords.shape[0] == 0 and pred_coords.shape[0] == 0:
        return Metrics(0, 0, 0, 1, 1, 1, 0), np.zeros((0,)), np.zeros((0,))

    if gt_coords.shape[0] == 0:
        return (
            Metrics(0, pred_coords.shape[0], 0, 0, 0, 0, 0),
            np.zeros((0,)),
            np.zeros((0,)),
        )

    if pred_coords.shape[0] == 0:
        return (
            Metrics(0, 0, gt_coords.shape[0], 0, 0, 0, 0),
            np.zeros((0,)),
            np.zeros((0,)),
        )

    # Computes the cost matrix using the L2 distance.
    c_matrix = cdist(pred_coords, gt_coords)

    # Filter matches that are not within the defined gt region.
    # Expand dims so that thresh vector is broadcasted across columns.
    thresh = np.expand_dims(thresh, axis=0) if isinstance(thresh, ndarray) else thresh
    c_matrix_bool = c_matrix <= thresh
    # Hungarian matching on the boolean matrix.
    _, assign = hungarian(c_matrix_bool)

    # Computes number of TPs, FPs and FNs.
    fn_gt_index = np.array(np.where(assign.sum(0) == 0))[0]
    tp_pred_index = np.array(np.where(assign.sum(1) == 1))[0]
    fp_pred_index = np.array(np.where(assign.sum(1) == 0))[0]
    n_tp = tp_pred_index.shape[0]
    n_fp = fp_pred_index.shape[0]
    n_fn = fn_gt_index.shape[0]

    ind = assign.nonzero()

    if n_tp + n_fp == 0 or n_tp + n_fn == 0:
        recall, precision, f1 = 0, 0, 0
    else:
        recall = n_tp / (n_tp + n_fn)
        precision = n_tp / (n_tp + n_fp)
        f1 = (
            0
            if precision + recall == 0
            else (2 * precision * recall) / (precision + recall)
        )

    mae = math.fabs(len(gt_coords) - len(pred_coords))

    return Metrics(n_tp, n_fp, n_fn, precision, recall, f1, mae), ind[1], ind[0]


def compute_image_metrics_dist(
    gt_coords: np.ndarray, pred_coords: np.ndarray, thresh: Union[float, ndarray]
) -> tuple[Metrics, tuple[np.ndarray]]:
    """Computes detection evaluation metrics.
    Evaluation implemented using the Hungarian algorithm from (Section 3.4):
    https://arxiv.org/pdf/2001.03360.pdf
    The code is partially taken from the official NWPU evaluation code:
    https://github.com/gjy3035/NWPU-Crowd-Sample-Code-for-Localization

    This implementation uses scipy's linear_sum_assignment for the Hungarian algorithm
    on the distance matrix directly, with thresholding applied afterward.

    Args:
        gt_coords: N X 2 ndarray of ground truth cell locations (X, Y).
        pred_coords: M X 2 ndarray of predicted cell locations (X, Y).
        thresh: Distance threshold (float) or ndarray of dist. thresholds per GT.

    Returns:
        Tuple of: TP, FP, FN, Precision, Recall, F1, MAE (cell counting) and
        tuple of col_ind (GTs) and row_ind (preds) from the Hungarian matching.
    """
    # Check for special conditions
    if gt_coords.shape[0] == 0 and pred_coords.shape[0] == 0:
        return Metrics(0, 0, 0, 1, 1, 1, 0), np.zeros((0,)), np.zeros((0,))
    if gt_coords.shape[0] == 0:
        return (
            Metrics(0, pred_coords.shape[0], 0, 0, 0, 0, 0),
            np.zeros((0,)),
            np.zeros((0,)),
        )
    if pred_coords.shape[0] == 0:
        return (
            Metrics(0, 0, gt_coords.shape[0], 0, 0, 0, 0),
            np.zeros((0,)),
            np.zeros((0,)),
        )

    # Computes the cost matrix using the L2 distance.
    c_matrix = cdist(pred_coords, gt_coords)

    # Apply Hungarian algorithm directly on the distance matrix using scipy's implementation
    row_ind, col_ind = linear_sum_assignment(c_matrix)

    # Apply thresholding after Hungarian matching
    # Expand dims so that thresh vector is broadcasted across matched pairs
    if isinstance(thresh, ndarray):
        # For variable thresholds per GT point
        matched_distances = c_matrix[row_ind, col_ind]
        valid_matches = matched_distances <= thresh[col_ind]
    else:
        # For a single threshold value
        matched_distances = c_matrix[row_ind, col_ind]
        valid_matches = matched_distances <= thresh

    # Get valid matches after thresholding
    valid_row_ind = row_ind[valid_matches]
    valid_col_ind = col_ind[valid_matches]

    # Create assignment matrix for counting TP, FP, FN
    assign = np.zeros((pred_coords.shape[0], gt_coords.shape[0]), dtype=bool)
    if len(valid_row_ind) > 0:
        assign[valid_row_ind, valid_col_ind] = True

    # Compute metrics
    tp_pred_index = np.array(np.where(assign.sum(1) > 0))[0]
    fp_pred_index = np.array(np.where(assign.sum(1) == 0))[0]
    fn_gt_index = np.array(np.where(assign.sum(0) == 0))[0]

    n_tp = tp_pred_index.shape[0]
    n_fp = fp_pred_index.shape[0]
    n_fn = fn_gt_index.shape[0]

    if n_tp + n_fp == 0 or n_tp + n_fn == 0:
        recall, precision, f1 = 0, 0, 0
    else:
        recall = n_tp / (n_tp + n_fn)
        precision = n_tp / (n_tp + n_fp)
        f1 = (
            0
            if precision + recall == 0
            else (2 * precision * recall) / (precision + recall)
        )

    mae = math.fabs(len(gt_coords) - len(pred_coords))

    return (
        Metrics(n_tp, n_fp, n_fn, precision, recall, f1, mae),
        valid_col_ind,
        valid_row_ind,
    )


def hungarian(matches_bool: np.ndarray) -> tuple[int, np.ndarray]:
    """Implements Hungarian matching on boolean matrix.

    The code is taken from the official NWPU evaluation code:
    https://github.com/gjy3035/NWPU-Crowd-Sample-Code-for-Localization

    Args:
        matches_bool: Boolean matching matrix (preds, gts).

    Returns:
        Number of matches (TPs) and the boolean match matrix (preds, gts).
    """
    # Matrix to adjacent matrix
    edges = np.argwhere(matches_bool)
    lnum, rnum = matches_bool.shape
    graph = [[] for _ in range(lnum)]
    for edge in edges:
        graph[edge[0]].append(edge[1])

    # Deep first search
    match = [-1 for _ in range(rnum)]
    vis = [-1 for _ in range(rnum)]

    def dfs(u):
        for v in graph[u]:
            if vis[v]:
                continue
            vis[v] = True
            if match[v] == -1 or dfs(match[v]):
                match[v] = u
                return True
        return False

    # for loop
    ans = 0
    for a in range(lnum):
        for i in range(rnum):
            vis[i] = False
        if dfs(a):
            ans += 1

    # Assignment matrix
    assign = np.zeros((lnum, rnum), dtype=bool)
    for i, m in enumerate(match):
        if m >= 0:
            assign[m, i] = True

    return ans, assign


def reduce_metrics(metrics: list[Metrics], reduce: str = "mean") -> Metrics:
    """Reduces metrics.

    Args:
        metrics: List of Metrics to aggregate.
        reduce: Reduction strategy ("mean", "max", "sum").

    Returns:
        Reduced Metrics object.

    Raises:
        NotImplementedError: If reduce parameter is given that is not supported.
    """
    if reduce == "mean":
        return Metrics(*np.nanmean(metrics, axis=0, dtype=np.float32))
    elif reduce == "max":
        return Metrics(*np.nanmax(metrics, axis=0))
    elif reduce == "sum":
        return Metrics(*np.nansum(metrics, axis=0))
    else:
        raise NotImplementedError(
            f"{reduce} reduction strategy is not supported! "
            "Only 'mean' and 'max' are supported for now."
        )


def compute_f1_from_metrics(metrics: Metrics) -> Metrics:
    """Computes overall F1 metrics.

    This method performs F1 metric calculation on the object level, instead of
    doing the F1 metric averaging across image samples. F1 is computed from
    the overall precision and recall values, based on summed-up number of TPs,
    FPs and FNs. MAE is still reported as an average across image samples.
    This represents the official metric to compare with. Image-level averages
    are not the best approach in the case of images without the presence of
    certain classes (F1=0 or F1=1 dilemma + numbers can be biased).

    This follows the NWPU evaluation implementation:
    https://github.com/gjy3035/NWPU-Crowd-Sample-Code-for-Localization

    Args:
        metrics: Input Metrics object (non-aggregated).

    Returns:
        Metrics: Aggregated metrics across all objects (F1, tp, ).
    """
    metrics_s = reduce_metrics(metrics, reduce="sum")
    metrics_m = reduce_metrics(metrics, reduce="mean")
    ap = metrics_s.n_tp / (metrics_s.n_tp + metrics_s.n_fp + 1e-20)
    ar = metrics_s.n_tp / (metrics_s.n_tp + metrics_s.n_fn + 1e-20)
    f1 = 2 * ap * ar / (ap + ar)

    return Metrics(
        metrics_s.n_tp, metrics_s.n_fp, metrics_s.n_fn, ap, ar, f1, metrics_m.mae
    )


def get_peaks(
    reg_out: np.ndarray, cls_out: np.ndarray, cls_idx: int, thresh: float = 0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns detected peaks for the given class.

    M = fH * fW * K
    N = Number of shifted anchors with fg probability > thresh.
    Note: Softmax needs to be applied before to cls_out.
    Peaks are also filtered for potential (exact) duplicate detections.

    Args:
        reg_out: Regression head output: (M, 2).
        cls_out: Classification head output (M, num_cls + 1).
        cls_idx: Class index to be filtered ([1, num_cls]).
        thresh: Minimum fg object cls score (defaults to 0.5).

    Returns:
        (N, 2) array of coordinates of the detected peaks.
        (N,) array of probabilities of the detected peaks.
        (N,) detection mask (> thresh).
    """
    det_mask = cls_out[:, cls_idx] > thresh
    peaks, idx = np.unique(reg_out[det_mask, :], return_index=True, axis=0)
    peaks_conf = (cls_out[det_mask, cls_idx])[idx]
    det_mask = det_mask[idx]

    return peaks, peaks_conf, det_mask


def save_predicted_images(
    img: np.ndarray,
    gt_coords: np.ndarray,
    pred_peaks: np.ndarray,
    pred_peaks_c: np.ndarray,
    ref_anchors: np.ndarray,
    pred_anchors: np.ndarray,
    pred_conf: np.ndarray,
    dist: Union[float, np.ndarray],
    epoch: int,
    filename: str,
    test: bool = False,
    draw_gt: bool = False,
    metrics: Metrics = None,
    matches: tuple[np.ndarray] = None,
    draw_matches: bool = False,
) -> None:
    """Visualizes predictions on original images.

    Args:
        img: Input image patch.
        gt_coords: Ground truth coordinates (N, 2).
        pred_peaks: Predicted peaks (M, 2).
        pred_peaks_c: Confidence of predicted peaks (M, 1).
        ref_anchors: Reference anchors along the grid K (M, 2).
        pred_anchors: Predicted anchors (ref + offset) (M, 2).
        pred_conf: Predicted class confidences (M, 1).
        dist: Evaluation distance threshold (fixed thresh, or per GT).
        epoch: Current epoch at which validation is performed.
        filename: Filename of the output image.
        test: Output to test dir, instead of val.
        draw_gt: Whether to draw GT on the input img.
        metrics: Tuple of TP, FP, FN, Precision, Recall, F1, MAE.
        matches: Indices of matched indices (idx_i, idx_j).
        draw_matches: Whether to draw matches between GTs and predictions.

    Returns:
        Image (img | anchors | predictions)
        Saved to {epoch}/{val|test} folder.
    """
    Path(f"epochs/{epoch}/{'test' if test else 'val'}").mkdir(
        parents=True, exist_ok=True
    )

    fig = plt.figure(figsize=(24, 6))

    plt.subplot(141)
    plt.title("Input Image Patch")
    plt.imshow(img)

    # Plot GT points (overlay)
    if draw_gt:
        for idx, gt_coord in enumerate(gt_coords):
            dist_c = dist[idx] if isinstance(dist, np.ndarray) else dist
            fig.get_axes()[0].add_patch(
                plt.Circle(
                    (gt_coord[0], gt_coord[1]), dist_c // 2, color="g", fill=True
                )
            )
    plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

    # Visualize reference anchor points (overlay)
    plt.subplot(142)
    plt.title("Reference Anchor Points")
    plt.imshow(img)
    fig.get_axes()[1].plot(ref_anchors[:, 0], ref_anchors[:, 1], "ro")
    plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

    # Visualize predicted anchor points (overlay)
    plt.subplot(143)
    plt.title("Predicted Anchor Points")
    plt.imshow(img)
    fig.get_axes()[2].plot(pred_anchors[:, 0], pred_anchors[:, 1], "ro")
    plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

    # Add predicted confidence text (all anchors)
    for idx, c_txt in enumerate(pred_conf):
        txt_int = int(c_txt * 100)
        if txt_int > 0:
            fig.get_axes()[2].text(
                pred_anchors[idx, 0],
                pred_anchors[idx, 1],
                txt_int,
                c="b",
                fontweight="bold",
            )

    # Visualize GT points, predictions and GT radious (overlay)
    plt.subplot(144)
    if metrics:
        n_gt, n_p = gt_coords.shape[0], pred_peaks.shape[0]
        plt.title(
            f"NG: {n_gt}, NP: {n_p}, TP: {metrics.n_tp}, FP: {metrics.n_fp}, FN:"
            f" {metrics.n_fn}, Predictions - F1: {metrics.f1:.2f}, Pre:"
            f" {metrics.precision:.2f}, Rec: {metrics.recall:.2f}, MAE:"
            f" {metrics.mae:.2f}"
        )
    else:
        plt.title("Predictions")
    plt.imshow(img)
    if pred_peaks.shape[0] > 0:
        fig.get_axes()[3].plot(pred_peaks[:, 0], pred_peaks[:, 1], "ro")

        # Add predicted confidence text (only matches)
        for idx, c_txt in enumerate(pred_peaks_c):
            fig.get_axes()[3].text(
                pred_peaks[idx, 0],
                pred_peaks[idx, 1],
                int(c_txt * 100),
                c="b",
                fontweight="bold",
            )

    if gt_coords.shape[0] > 0 and draw_gt:
        fig.get_axes()[3].plot(gt_coords[:, 0], gt_coords[:, 1], "go")
    for idx, gt_coord in enumerate(gt_coords):
        dist_c = dist[idx] if isinstance(dist, np.ndarray) else dist
        fig.get_axes()[2].add_patch(
            plt.Circle((gt_coord[0], gt_coord[1]), dist_c, color="g", fill=False)
        )
        fig.get_axes()[3].add_patch(
            plt.Circle((gt_coord[0], gt_coord[1]), dist_c, color="g", fill=False)
        )
    plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

    # Draw matching lines between GT and predicted (matched) anchors.
    if draw_matches and len(matches[0]) > 0 and len(matches[1]) > 0:
        matched_gt = gt_coords[matches[0]]
        matched_a = pred_peaks[matches[1]]
        for idx in range(len(matched_gt)):
            matched_gt_c = matched_gt[idx]
            matched_a_c = matched_a[idx]
            fig.get_axes()[3].plot(
                (matched_a_c[0], matched_gt_c[0]),
                (matched_a_c[1], matched_gt_c[1]),
                color="orange",
                linestyle="-",
            )

    plt.tight_layout()
    plt.savefig(
        f"epochs/{epoch}/{'test' if test else 'val'}/{filename}.png",
        bbox_inches="tight",
    )
    plt.close()


def save_predicted_images_all(
    img: np.ndarray,
    gt_peaks: dict[str, np.ndarray],
    pred_peaks: dict[str, np.ndarray],
    dist: float,
    epoch: int,
    filename: str,
    color_classes: dict[int, str],
    img_size: tuple[int, int] = (6, 6),
    test: bool = True,
) -> None:
    """Visualizes predictions on original images.

    Args:
        img: Input image patch.
        gt_peaks: Ground truth coordinates dict[class_name, (N, 2)].
        pred_peaks: Predicted peaks dict[class_name, (M, 2)].
        dist: Evaluation distance threshold (fixed thresh).
        epoch: Current epoch at which validation is performed.
        filename: Filename of the output image.
        color_classes: Dictionary of colors per class.

    Returns:
        Image (img | anchors | predictions)
        Saved to {epoch}/test|val folder.
    """
    Path(f"epochs/{epoch}/{'test' if test else 'val'}").mkdir(
        parents=True, exist_ok=True
    )
    plt.figure(figsize=img_size)
    plt.imshow(img)

    # Plot predicted points (overlay)
    for class_name, pred_coords in pred_peaks.items():
        if class_name == 0:
            # Skip detection and normal cell
            continue
        for idx in range(pred_coords.shape[0]):
            plt.gca().add_patch(
                plt.Circle(
                    (pred_coords[idx, 0], pred_coords[idx, 1]),
                    dist // 4,
                    color=tuple(c / 255 for c in color_classes[class_name]),
                    fill=True,
                )
            )
    # Plot GT radius (overlay)
    for class_name in color_classes.keys():
        gt_coords = gt_peaks[class_name]
        for gt_coord in gt_coords:
            plt.gca().add_patch(
                plt.Circle(
                    (gt_coord[0], gt_coord[1]),
                    dist,
                    color=tuple(c / 255 for c in color_classes[class_name]),
                    fill=False,
                )
            )
    plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
    plt.tight_layout()
    plt.savefig(
        f"epochs/{epoch}/{'test' if test else 'val'}/{filename}.png",
        bbox_inches="tight",
    )
    plt.close()


def extract_image_patches(
    x: torch.Tensor,
    kernel: int,
    stride: int = 1,
    pad_value: float = 0.0,
    dilation: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int, int]]:
    """Extracts image patches.

    Args:
        x:
            Torch Tensor input image [b, c, h, w] - can be on a CPU or GPU.
        kernel:
            Kernel dimension (i.e. patch size).
        stride:
            Stride to perform unfolding (e.g., kernel // 2 for 50% overlap).
        pad_value:
            Value used for padding.
        dilation:
            Dilation, if used. Currently fixed to 1.

    Returns:
        Tuple(Tensor of shape: [b, n_p_r, n_p_c, c, kernel, kernel],
        Padded input image,
        Padding: (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom).
    """
    # Do TF "SAME" Padding
    b, c, h, w = x.shape
    h2 = math.ceil(h / stride)
    w2 = math.ceil(w / stride)
    pad_row = (h2 - 1) * stride + (kernel - 1) * dilation + 1 - h
    pad_col = (w2 - 1) * stride + (kernel - 1) * dilation + 1 - w
    x = F.pad(
        x,
        (pad_col // 2, pad_col - pad_col // 2, pad_row // 2, pad_row - pad_row // 2),
        value=pad_value,
    )

    # Extract patches
    # [b, c, n_p_r, n_p_c, k_h, k_w]
    patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)

    return (
        patches.permute(0, 2, 3, 1, 4, 5).contiguous(),
        x,
        (pad_row // 2, pad_row - pad_row // 2, pad_col // 2, pad_col - pad_col // 2),
    )


def local_to_global_peaks(
    peaks: np.ndarray,
    patch_loc: tuple[int, int],
    stride: int,
    padding: tuple[int, int, int, int],
) -> np.ndarray:
    """Converts patch-level coordinates to image-level ones.

    Args:
        peaks: (N, 2) array of coordinates of the detected peaks.
        patch_loc: (row, column) location of the extracted patch.
        stride:  Stride to perform unfolding.
        padding: (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom)

    Returns:
        (N, 2) peak locations, converted to image-level coordinates.
    """
    off_x = patch_loc[0] * stride - padding[2]
    off_y = patch_loc[1] * stride - padding[0]
    peaks_full = peaks + np.array([off_x, off_y])

    return peaks_full
