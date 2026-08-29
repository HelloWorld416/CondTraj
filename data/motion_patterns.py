"""Training-only stepwise motion-pattern clustering from Section IV-A."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


def _wrapped_angle_delta(angle: np.ndarray) -> np.ndarray:
    """Return adjacent angle differences in [-pi, pi]."""
    return np.arctan2(np.sin(angle[..., 1:] - angle[..., :-1]),
                      np.cos(angle[..., 1:] - angle[..., :-1]))


def extract_motion_features(trajectories: np.ndarray, delta_t: float) -> np.ndarray:
    """Compute [velocity, acceleration, heading, angular velocity] per frame.

    The final frame repeats the final available finite-difference feature so the
    label sequence has the same length as the input trajectory.
    """
    trajectories = np.asarray(trajectories, dtype=np.float64)
    if trajectories.ndim != 3 or trajectories.shape[-1] != 2:
        raise ValueError("trajectories must have shape [N, T, 2]")
    if trajectories.shape[1] < 3:
        raise ValueError("at least three trajectory frames are required")
    if delta_t <= 0:
        raise ValueError("delta_t must be positive")

    displacement = np.diff(trajectories, axis=1)
    velocity = np.linalg.norm(displacement, axis=-1) / delta_t
    heading = np.arctan2(displacement[..., 1], displacement[..., 0])

    acceleration = np.diff(velocity, axis=1) / delta_t
    angular_velocity = _wrapped_angle_delta(heading) / delta_t

    velocity = np.concatenate([velocity, velocity[..., -1:]], axis=1)
    heading = np.concatenate([heading, heading[..., -1:]], axis=1)
    acceleration = np.concatenate(
        [acceleration, acceleration[..., -1:], acceleration[..., -1:]], axis=1)
    angular_velocity = np.concatenate(
        [angular_velocity, angular_velocity[..., -1:], angular_velocity[..., -1:]], axis=1)
    return np.stack([velocity, acceleration, heading, angular_velocity], axis=-1)


def smooth_labels(labels: np.ndarray, width: int = 3) -> np.ndarray:
    """Centered mode filter; ties retain the unsmoothed center label."""
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 2:
        raise ValueError("labels must have shape [N, T]")
    if width != 3:
        raise ValueError("the paper specifies a smoothing width of 3")

    smoothed = labels.copy()
    # For a width-three categorical mode, only matching neighbors can replace
    # the center. All-distinct windows are ties and therefore keep the center.
    matching_neighbors = labels[:, :-2] == labels[:, 2:]
    interior = smoothed[:, 1:-1]
    interior[matching_neighbors] = labels[:, :-2][matching_neighbors]
    return smoothed


@dataclass
class MotionPatternClusterer:
    """A frozen scaler and deterministic K-Means vocabulary."""

    num_clusters: int = 3
    max_iter: int = 100
    tolerance: float = 1e-5
    delta_t: float = 0.4
    random_seed: int = 0
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None
    centroids: np.ndarray | None = None

    @property
    def fitted(self) -> bool:
        return self.mean is not None and self.scale is not None and self.centroids is not None

    def _standardize(self, features: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None:
            raise RuntimeError("the motion-pattern scaler has not been fitted")
        return (features - self.mean) / self.scale

    @staticmethod
    def _nearest(samples: np.ndarray, centroids: np.ndarray, chunk_size: int = 262144):
        labels = np.empty(len(samples), dtype=np.int64)
        min_distances = np.empty(len(samples), dtype=np.float64)
        for start in range(0, len(samples), chunk_size):
            end = min(start + chunk_size, len(samples))
            distances = ((samples[start:end, None] - centroids[None]) ** 2).sum(axis=-1)
            labels[start:end] = distances.argmin(axis=1)
            min_distances[start:end] = distances.min(axis=1)
        return labels, min_distances

    def fit(self, trajectories: np.ndarray) -> "MotionPatternClusterer":
        features = extract_motion_features(trajectories, self.delta_t).reshape(-1, 4)
        self.mean = features.mean(axis=0)
        self.scale = features.std(axis=0)
        self.scale = np.where(self.scale < 1e-8, 1.0, self.scale)
        features -= self.mean
        features /= self.scale
        samples = features

        if len(samples) < self.num_clusters:
            raise ValueError("fewer feature vectors than requested clusters")
        rng = np.random.default_rng(self.random_seed)
        centroids = samples[rng.choice(len(samples), self.num_clusters, replace=False)].copy()
        for _ in range(self.max_iter):
            labels, min_distances = self._nearest(samples, centroids)
            updated = centroids.copy()
            for cluster_id in range(self.num_clusters):
                members = samples[labels == cluster_id]
                if len(members):
                    updated[cluster_id] = members.mean(axis=0)
                else:
                    replacement = min_distances.argmax()
                    updated[cluster_id] = samples[replacement]
                    min_distances[replacement] = -1
            shift = np.linalg.norm(updated - centroids, axis=1).max()
            centroids = updated
            if shift <= self.tolerance:
                break
        self.centroids = centroids
        return self

    def predict(self, trajectories: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("fit must be called using training trajectories first")
        features = extract_motion_features(trajectories, self.delta_t)
        shape = features.shape[:2]
        features -= self.mean
        features /= self.scale
        labels, _ = self._nearest(features.reshape(-1, 4), self.centroids)
        return smooth_labels(labels.reshape(shape), width=3)

    def fit_predict(self, trajectories: np.ndarray) -> np.ndarray:
        return self.fit(trajectories).predict(trajectories)

    def semantic_descriptions(self) -> Dict[int, str]:
        """Interpret clusters from their raw kinematic centroid values."""
        if not self.fitted:
            raise RuntimeError("fit must be called before interpreting centroids")
        raw = self.centroids * self.scale + self.mean
        acceleration_threshold = max(1e-8, 0.1 * self.scale[1])
        turn_threshold = max(1e-8, 0.1 * self.scale[3])
        stationary_threshold = max(1e-8, 0.1 * self.mean[0])
        descriptions = {}
        for cluster_id, (speed, acceleration, _, angular_velocity) in enumerate(raw):
            if speed <= stationary_threshold:
                descriptions[cluster_id] = "stationary"
                continue
            if acceleration > acceleration_threshold:
                pace = "accelerating"
            elif acceleration < -acceleration_threshold:
                pace = "decelerating"
            else:
                pace = "constant-speed"
            if angular_velocity > turn_threshold:
                direction = "left-turn"
            elif angular_velocity < -turn_threshold:
                direction = "right-turn"
            else:
                direction = "straight"
            descriptions[cluster_id] = f"{pace} {direction}"
        return descriptions

    def state_dict(self) -> Dict[str, object]:
        if not self.fitted:
            raise RuntimeError("cannot serialize an unfitted clusterer")
        return {
            "num_clusters": self.num_clusters,
            "max_iter": self.max_iter,
            "tolerance": self.tolerance,
            "delta_t": self.delta_t,
            "random_seed": self.random_seed,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "centroids": self.centroids.tolist(),
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, object]) -> "MotionPatternClusterer":
        return cls(
            num_clusters=int(state["num_clusters"]),
            max_iter=int(state["max_iter"]),
            tolerance=float(state["tolerance"]),
            delta_t=float(state["delta_t"]),
            random_seed=int(state["random_seed"]),
            mean=np.asarray(state["mean"], dtype=np.float64),
            scale=np.asarray(state["scale"], dtype=np.float64),
            centroids=np.asarray(state["centroids"], dtype=np.float64),
        )
