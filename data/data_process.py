"""Dataset loaders with training-only stepwise motion-pattern labels."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .motion_patterns import MotionPatternClusterer


def _make_clusterer(k: int, t: int, e: float, delta_t: float, seed: int) -> MotionPatternClusterer:
    return MotionPatternClusterer(
        num_clusters=k, max_iter=t, tolerance=e, delta_t=delta_t, random_seed=seed
    )


def _label_trajectories(
    trajectories: np.ndarray,
    mode: str,
    clusterer: Optional[MotionPatternClusterer],
    k: int,
    t: int,
    e: float,
    delta_t: float,
    seed: int,
) -> tuple[np.ndarray, MotionPatternClusterer]:
    if mode == "train":
        if clusterer is not None and clusterer.fitted:
            labels = clusterer.predict(trajectories)
        else:
            clusterer = _make_clusterer(k, t, e, delta_t, seed)
            labels = clusterer.fit_predict(trajectories)
    else:
        if clusterer is None or not clusterer.fitted:
            raise ValueError("validation/test data require a clusterer fitted on the training split")
        labels = clusterer.predict(trajectories)
    return labels.astype(np.int64), clusterer


class NBADataset(Dataset):
    """Preprocessed NBA split released by the baseline used in the paper."""

    def __init__(
        self,
        data_path: str = "./data",
        obs_len: int = 10,
        pred_len: int = 20,
        t: int = 100,
        k: int = 3,
        e: float = 1e-5,
        training: bool = True,
        clusterer: Optional[MotionPatternClusterer] = None,
        delta_t: float = 0.2,
        random_seed: int = 0,
        validation_split: float = 0.8,
    ):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        mode = "train" if training else "test"
        filename = "nba_train.npy" if training else "nba_test.npy"
        self.trajs = np.load(os.path.join(data_path, "nba", filename)).astype(np.float32)
        self.trajs /= 94 / 28
        split = round(len(self.trajs) * validation_split)
        self.trajs = self.trajs[:split] if training else self.trajs[split:]
        if self.trajs.shape[1:] != (obs_len + pred_len, 11, 2):
            raise ValueError(f"unexpected NBA trajectory shape: {self.trajs.shape}")

        flat = self.trajs.transpose(0, 2, 1, 3).reshape(-1, obs_len + pred_len, 2)
        labels, self.clusterer = _label_trajectories(
            flat, mode, clusterer, k, t, e, delta_t, random_seed
        )
        self.labels = labels.reshape(len(self.trajs), 11, obs_len + pred_len)
        self.traj_abs = torch.from_numpy(self.trajs).permute(0, 2, 1, 3).float()

    def __len__(self) -> int:
        return len(self.traj_abs)

    def __getitem__(self, index: int):
        trajectory = self.traj_abs[index]
        return {
            "pre_motion_3D": trajectory[:, :self.obs_len],
            "fut_motion_3D": trajectory[:, self.obs_len:],
            "pre_motion_mask": torch.ones(11, self.obs_len),
            "fut_motion_mask": torch.ones(11, self.pred_len),
            "cluster_label": torch.from_numpy(self.labels[index]).long(),
        }


class SDDDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        obs_len: int,
        pred_len: int,
        t: int = 100,
        k: int = 3,
        e: float = 1e-5,
        flip_aug: bool = False,
        mode: str = "train",
        clusterer: Optional[MotionPatternClusterer] = None,
        delta_t: float = 0.4,
        random_seed: int = 0,
    ):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        filename = "train_8_12.npy" if mode == "train" else "val_8_12.npy"
        trajectories = np.load(os.path.join(data_path, "sdd", filename))[:, :, :2]
        if trajectories.shape[1] != obs_len + pred_len:
            raise ValueError(f"unexpected SDD trajectory shape: {trajectories.shape}")
        if flip_aug:
            if mode != "train":
                raise ValueError("SDD flip augmentation is only valid for the training split")
            flipped = np.flip(trajectories, axis=1).copy()
            trajectories = np.concatenate([trajectories, flipped], axis=0)
        self.sdd_dataset = trajectories.astype(np.float32)
        self.labels, self.clusterer = _label_trajectories(
            self.sdd_dataset, mode, clusterer, k, t, e, delta_t, random_seed
        )

    def __len__(self) -> int:
        return len(self.sdd_dataset)

    def __getitem__(self, index: int):
        trajectory = self.sdd_dataset[index]
        return {
            "past_traj": torch.from_numpy(trajectory[:self.obs_len]).float(),
            "fut_traj": torch.from_numpy(trajectory[self.obs_len:]).float(),
            "cluster_label": torch.from_numpy(self.labels[index]).long(),
        }
