import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from build_trainer import Trainer
from model.classifier import Classifier, LinearChainCRF
from tool.tools import (
    dct_transform,
    ddpm_sample_iter,
    ddpm_sample_loop_accelerate,
    get_dct_matrix,
    idct_transform,
    make_beta_schedule,
)


class CRFAndDCTTests(unittest.TestCase):
    def test_crf_partition_and_decode_match_brute_force(self):
        torch.manual_seed(0)
        crf = LinearChainCRF(2)
        emissions = torch.randn(1, 3, 2)
        paths = torch.tensor(list(itertools.product(range(2), repeat=3)))
        expanded = emissions.expand(len(paths), -1, -1)
        scores = crf.sequence_score(expanded, paths)
        expected_partition = torch.logsumexp(scores, dim=0)
        self.assertTrue(torch.allclose(crf.log_partition(emissions)[0], expected_partition))
        expected_path = paths[scores.argmax()]
        self.assertTrue(torch.equal(crf.decode(emissions)[0], expected_path))

    def test_classifier_emits_one_label_distribution_per_time_step(self):
        model = Classifier(
            num_frames=5,
            label_frames=8,
            latent_dim=16,
            ff_size=32,
            num_layers=1,
            num_heads=2,
            output_dim=3,
            dropout=0.0,
        )
        emissions = model(torch.randn(2, 5, 2), torch.tensor([10, 20]))
        self.assertEqual(emissions.shape, (2, 8, 3))
        labels = model.decode(emissions)
        self.assertEqual(labels.shape, (2, 8))
        self.assertTrue(torch.isfinite(model.crf_loss(emissions, labels)))

    def test_full_dct_round_trip(self):
        dct, inverse = get_dct_matrix(8)
        trajectory = torch.randn(3, 8, 2, dtype=torch.float64)
        frequencies = dct_transform(trajectory, dct, 8)
        restored = idct_transform(frequencies, inverse, 8)
        self.assertTrue(torch.allclose(trajectory, restored, atol=1e-10))

    def test_reverse_sampler_uses_iccv_step_order(self):
        cfg = SimpleNamespace(denoising_step=5, obs_len=1, n_pre=2)
        current = torch.randn(1, 2, 2)
        identity = torch.eye(2)
        steps = []

        def record_step(cfg, model, current, step, *args, **kwargs):
            steps.append(step)
            return current

        with patch("tool.tools.ddpm_sample_iter", side_effect=record_step):
            ddpm_sample_loop_accelerate(
                cfg,
                model=None,
                cond_fn=None,
                history_dct=current,
                initial_dct=current,
                dct_m=identity,
                idct_m=identity,
                pattern_labels=None,
                exp="Test",
            )
        self.assertEqual(steps, [4, 3, 2, 1, 0])

    def test_reverse_sampler_uses_100_step_linear_beta(self):
        class ZeroNoiseModel:
            def __init__(self):
                self.beta = None

            def generate_accelerate(self, current, beta, history):
                self.beta = beta.detach().clone()
                return torch.zeros_like(current)

        model = ZeroNoiseModel()
        current = torch.ones(2, 3, 2)
        ddpm_sample_iter(
            SimpleNamespace(),
            model,
            current,
            step=4,
            history_dct=current,
            labels=None,
            exp="Test",
            cond_fn=None,
        )
        expected = make_beta_schedule(
            "linear", n_timesteps=100, start=1e-4, end=5e-2
        )[4]
        self.assertTrue(torch.allclose(model.beta, expected.expand_as(model.beta)))

    def test_generator_losses_and_apd_match_iccv_definitions(self):
        future = torch.zeros(2, 1, 2)
        predictions = torch.tensor(
            [
                [[[1.0, 0.0]], [[3.0, 0.0]]],
                [[[4.0, 0.0]], [[2.0, 0.0]]],
            ]
        )
        self.assertAlmostEqual(Trainer.marginal_loss(predictions, future).item(), 1.5)
        self.assertAlmostEqual(Trainer.joint_loss(predictions, future).item(), 2.5)
        self.assertAlmostEqual(Trainer.average_pairwise_distance(predictions), 4.0)


if __name__ == "__main__":
    unittest.main()
