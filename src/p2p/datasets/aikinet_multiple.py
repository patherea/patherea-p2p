"""AIKiNET/Patherea dataset"""

import os
import random
import uuid
from glob import iglob
from pathlib import Path
from typing import Dict, List, Tuple, Union

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations import Compose
from albumentations.pytorch import ToTensorV2
from loguru import logger
from numpy import ndarray
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

MAPPING = {
    1: "positive_tumor_cell",
    2: "negative_tumor_cell",
    3: "normal_cell",
    4: "others_positive",
    5: "others_negative",
}

COLORS = {
    0: (0, 255, 0),
    1: (0, 255, 0),
    2: (255, 255, 0),
    3: (0, 0, 255),
    4: (128, 0, 128),
    5: (255, 165, 0),
}


class AIKiNETMultiDataset(Dataset):
    def __init__(
        self,
        data_dir: Union[Path, str],
        transforms: Compose,
        train: bool = True,
        test_fold: int = None,
        test_ratio: float = 0.5,
        seed: int = 1,
        debug: bool = False,
        solo_other_pos: bool = True,
        train_ratio: float = 1.0,
    ) -> None:
        """Initiates AIKiNET dataset object.

        AIKiNET-multi dataset is configured through the AIKiNET config file.
        This is our custom NET dataset collected at the Institute of Pathology.

        Args:
            data_dir: Path to directory with training data.
            transforms: List of transforms.
            train: Select train or test mode.
            test_fold: Which cross-val set to validate/test on (e.g., 1, 2, 3).
            test_ratio: The ratio of samples in the test set (default: 0.5).
            seed: Set a custom seed for train/test random shuffling.
            debug: If true, transformed images are saved.
            solo_other_pos: If true, allow samples where only other_pos present.
            train_ratio: The proportion of training data used (defaults to 100%).
        """
        self.data_dir = Path(data_dir)
        self.transforms = transforms
        self.train = train
        self.test_ratio = test_ratio
        self.seed = seed
        self.debug = debug
        self.solo_other_pos = solo_other_pos
        self.train_ratio = train_ratio
        random.seed(self.seed)

        if test_fold:
            # Use folds for train/test split
            paths_f = os.path.join(
                data_dir,
                "folds",
                f"fold_{test_fold}_{'train' if self.train else 'test'}.txt",
            )
            with open(paths_f, "r") as f:
                self.paths = [
                    os.path.join(data_dir, "images", x.strip()) for x in f.readlines()
                ]
            # Optionally subsample traning data (SSL experiment)
            if self.train:
                random.shuffle(self.paths)
                end_idx_train = int(self.train_ratio * len(self.paths))
                self.paths = self.paths[:end_idx_train]
        else:
            # Split train/test online using "test_ratio"
            self.paths = list(sorted(iglob(os.path.join(data_dir, "images", "*.png"))))
            random.shuffle(self.paths)
            end_idx_train = int((1 - test_ratio) * len(self.paths))
            self.paths = (
                self.paths[:end_idx_train] if self.train else self.paths[end_idx_train:]
            )
            # Write split paths to output file
            output_paths = os.path.join(
                self.data_dir, f"paths_{'train' if self.train else 'test'}.txt"
            )
            with open(output_paths, "w") as f:
                for line in self.paths:
                    f.write(f"{os.path.basename(line)}\n")

        self.paths, self.gt_peaks = self._get_peaks()

    def __getitem__(self, index: int) -> Tuple[Tensor, List[ndarray]]:
        """Loads and returns a sample from the dataset at the given index.

        Keypoints are represented as list of lists, where the first list
        represents the individual classes, indexed as per the MAPPING and
        index 0 always representing the combined "detection" class.

        Args:
            index: Index of the sample to be returned.

        Returns:
            Tuple of Tensors (Image, Keypoints)
            After custom collate:
            Image shape: [B, C, H, W]
            Keypoints shape: List[List[ndarray[N, 2]]]
        """
        img = cv2.imread(self.paths[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        data_dict = self.transforms(
            image=img,
            keypoints=list(map(tuple, self.gt_peaks[index]["positive_tumor_cell"])),
            keypoints_1=list(map(tuple, self.gt_peaks[index]["positive_tumor_cell"])),
            keypoints_2=list(map(tuple, self.gt_peaks[index]["negative_tumor_cell"])),
            keypoints_3=list(map(tuple, self.gt_peaks[index]["normal_cell"])),
            keypoints_4=list(map(tuple, self.gt_peaks[index]["others_positive"])),
            keypoints_5=list(map(tuple, self.gt_peaks[index]["others_negative"])),
        )

        if self.debug:
            self._save_transformed_images(
                data_dict, os.path.splitext(os.path.basename(self.paths[index]))[0]
            )
            self._save_labelled_images(
                data_dict, os.path.splitext(os.path.basename(self.paths[index]))[0]
            )

        # List of Lists:
        # [ndarray_keypoints_all,
        # ndarray_keypoints_class_1,
        # ndarray_keypoints_class_2, ...]
        # -> collate: [[], [],...[n_batch]]
        keypoints_combined = [
            np.array(data_dict[f"keypoints_{idx}"]) for idx in MAPPING
        ]
        keypoints_combined = [np.array(data_dict["keypoints"])] + keypoints_combined

        return (
            data_dict["image"],
            keypoints_combined,
        )

    def _get_peaks(self) -> List[Dict[str, ndarray]]:
        """Gets peaks from the GT masks.

        Return peaks location [n_peaks, 2] ndarray obtained from the ground
        truth (GT) masks for each of the classes.
        Matlab ground truth annotations need to be converted first to masks
        using the "_mat_to_mask" method (check docstring).

        Returns:
            Peak locations: List[{mask_name, [n_peaks, 2]}]
        """
        gt_peaks = []
        img_paths = []
        logger.info(f"Started with {len(self.paths)} samples!")
        for img_path in self.paths:
            filename = Path(img_path).stem

            mask_paths = {
                "positive_tumor_cell": os.path.join(
                    Path(img_path).parents[1],
                    "labels",
                    f"{filename}_positive_tumor_cell.png",
                ),
                "negative_tumor_cell": os.path.join(
                    Path(img_path).parents[1],
                    "labels",
                    f"{filename}_negative_tumor_cell.png",
                ),
                "normal_cell": os.path.join(
                    Path(img_path).parents[1], "labels", f"{filename}_normal_cell.png"
                ),
                "others_positive": os.path.join(
                    Path(img_path).parents[1],
                    "labels",
                    f"{filename}_others_positive.png",
                ),
                "others_negative": os.path.join(
                    Path(img_path).parents[1],
                    "labels",
                    f"{filename}_others_negative.png",
                ),
            }

            gt_peaks_dict = {}
            for mask_name, mask_path in mask_paths.items():
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                peaks_idx = np.nonzero(mask == 255)
                peaks_idx = np.concatenate(
                    (
                        np.expand_dims(peaks_idx[1], axis=1),
                        np.expand_dims(peaks_idx[0], axis=1),
                    ),
                    axis=1,
                )
                gt_peaks_dict[mask_name] = peaks_idx
            # Drop others_positive samples where no positive_tumor_cell present
            # This boosts positive/others_positive discrimination
            if (
                len(gt_peaks_dict["others_positive"]) > 0
                and len(gt_peaks_dict["positive_tumor_cell"]) == 0
                and self.train
                and not self.solo_other_pos
            ):
                continue

            gt_peaks.append(gt_peaks_dict)
            img_paths.append(img_path)

        logger.info(f"Finished with {len(img_paths)} samples!")
        return img_paths, gt_peaks

    def _save_transformed_images(
        self, data_dict: Dict[str, Tensor], filename: str
    ) -> None:
        """Saves outputs of the data augmentation phase.

        Augmented images are visualized:
        [image_cls_1, gt_mask_cls_1,...,image_cls_n, gt_mask_cls_n]
        GT mask represents a mask with 255 values on peaks.

        Args:
            data_dict: Data dict from the Transform in __get_item__.
            filename: Filename for the output image.

        Returns:
            Images are saved to data_dir/transformations/filename.png.
        """
        Path(os.path.join(self.data_dir, "transformations")).mkdir(
            parents=True, exist_ok=True
        )
        img = data_dict["image"]
        keypoints = np.array(data_dict["keypoints"])

        img = img.permute(1, 2, 0).detach().numpy()
        mask = np.zeros(img.shape)
        if len(keypoints) > 0:
            mask[keypoints[:, 1], keypoints[:, 0]] = 255
        fig = plt.figure(figsize=(36, 6))

        n_plots = len(MAPPING) + 2
        plt.subplot(1, n_plots, 1)
        plt.title("Input Image")
        plt.imshow(img)
        plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

        plt.subplot(1, n_plots, 2)
        plt.title("GT Mask (all)")
        plt.imshow(mask, cmap="jet")
        plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

        subplot_id = 3
        for idx, class_name in MAPPING.items():
            plt.subplot(1, n_plots, subplot_id)
            subplot_id += 1
            plt.title(f"GT Mask ({class_name})")
            keypoints = np.array(data_dict[f"keypoints_{idx}"])
            mask = np.zeros(img.shape)
            if len(keypoints) > 0:
                mask[keypoints[:, 1], keypoints[:, 0]] = 255
            plt.imshow(mask, cmap="jet")
            plt.tick_params(
                labelbottom=False, labelleft=False, bottom=False, left=False
            )

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.data_dir, "transformations", f"{filename}_{uuid.uuid4().hex}.png"
            )
        )
        plt.close(fig)

    def _save_labelled_images(
        self, data_dict: Dict[str, Tensor], filename: str
    ) -> None:
        """Saves labeled images (as raw images).

        This is used only for visualization purposes.

        Args:
            data_dict: Data dict from the Transform in __get_item__.
            filename: Filename for the output image.

        Returns:
            Images are saved to data_dir/visualization/filename.png.
        """
        Path(os.path.join(self.data_dir, "visualization")).mkdir(
            parents=True, exist_ok=True
        )
        img = data_dict["image"]
        img = img.permute(1, 2, 0).detach().numpy()
        keypoints = np.array(data_dict["keypoints"], dtype=int)

        for idx, _ in MAPPING.items():
            keypoints = np.array(data_dict[f"keypoints_{idx}"])
            if len(keypoints) == 0:
                continue
            for point in keypoints:
                cv2.circle(img, (point[0], point[1]), 5, COLORS[idx], -1)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            os.path.join(
                self.data_dir, "visualization", f"{filename}_{uuid.uuid4().hex}.png"
            ),
            img,
        )

    def __len__(self) -> int:
        return len(self.paths)

    @staticmethod
    def custom_collate_gt(batch):
        return (
            torch.stack([img[0] for img in batch]),
            [gt[1] for gt in batch],
        )


if __name__ == "__main__":
    logger.info("Running demo code in the Patherea Dataset module...")

    transform = A.Compose(
        [
            A.RandomCrop(width=224, height=224),
            # A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
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
    )

    crc_dataset = AIKiNETMultiDataset(
        data_dir="/ceph/hpc/data/MFIP/patherea/datasets/AIKINET_LNET_224_v1.0",
        transforms=transform,
        train=False,
        test_ratio=0.5,
        seed=1,
        debug=True,
    )

    dataloader = DataLoader(
        crc_dataset,
        batch_size=1,
        num_workers=1,
        shuffle=True,
        pin_memory=True,
        collate_fn=crc_dataset.custom_collate_gt,
    )

    for img, keypoints in dataloader:
        logger.info(f"Img shape: {img.shape}, keypoints shape: {keypoints[0][0].shape}")
