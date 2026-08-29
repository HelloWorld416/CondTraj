import unittest
import os
import tempfile

import numpy as np

from data.data_process import SDDDataset
from data.motion_patterns import (
    MotionPatternClusterer,
    extract_motion_features,
    smooth_labels,
)
class MotionPatternTests(unittest.TestCase):
    def test_straight_constant_speed_features(self):
        x = np.arange(6, dtype=np.float64)
        trajectory = np.stack([x, np.zeros_like(x)], axis=-1)[None]
        features = extract_motion_features(trajectory, delta_t=1.0)
        np.testing.assert_allclose(features[..., 0], 1.0)
        np.testing.assert_allclose(features[..., 1], 0.0)
        np.testing.assert_allclose(features[..., 2], 0.0)
        np.testing.assert_allclose(features[..., 3], 0.0)

    def test_smoothing_retains_center_on_tie(self):
        labels = np.array([[0, 1, 1, 2]])
        expected = np.array([[0, 1, 1, 2]])
        np.testing.assert_array_equal(smooth_labels(labels), expected)

    def test_frozen_state_reproduces_predictions(self):
        rng = np.random.default_rng(0)
        trajectories = rng.normal(size=(20, 8, 2)).cumsum(axis=1)
        clusterer = MotionPatternClusterer(num_clusters=3, random_seed=0)
        expected = clusterer.fit_predict(trajectories)
        restored = MotionPatternClusterer.from_state_dict(clusterer.state_dict())
        np.testing.assert_array_equal(restored.predict(trajectories), expected)

    def test_sdd_training_flip_augmentation_doubles_dataset(self):
        trajectory = np.stack(
            [np.arange(20), np.zeros(20)], axis=-1
        ).astype(np.float32)
        trajectories = np.stack([trajectory, trajectory + 1], axis=0)
        with tempfile.TemporaryDirectory() as directory:
            sdd_directory = os.path.join(directory, "sdd")
            os.makedirs(sdd_directory)
            np.save(os.path.join(sdd_directory, "train_8_12.npy"), trajectories)

            dataset = SDDDataset(
                directory,
                obs_len=8,
                pred_len=12,
                mode="train",
                flip_aug=True,
                random_seed=0,
            )
            self.assertEqual(len(dataset), 2 * len(trajectories))
            np.testing.assert_array_equal(
                dataset.sdd_dataset[len(trajectories):], trajectories[:, ::-1]
            )


if __name__ == "__main__":
    unittest.main()
