import unittest
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from build_trainer import Trainer
from main import print_trainable_parameters, reverse_encoder_for_dataset
from model.denoising_model import CondTraj


class TransformerStructureTests(unittest.TestCase):
    def test_reverse_encoder_is_selected_by_dataset(self):
        self.assertEqual(reverse_encoder_for_dataset("SDD"), "transformer")
        self.assertEqual(reverse_encoder_for_dataset("NBA"), "mamba")

    def test_sdd_condtraj_uses_transformer_reverse_encoder(self):
        model = CondTraj(
            input_feats=2,
            num_frames=10,
            latent_dim=256,
            num_layers=2,
            num_heads=2,
            num_sample=2,
            dropout=0.0,
            reverse_encoder="transformer",
        )
        self.assertIsInstance(model.context_encoder, torch.nn.TransformerEncoder)
        self.assertIsInstance(model.denoising_transformer, torch.nn.TransformerEncoder)
        self.assertEqual(len(model.context_encoder.layers), 2)
        self.assertEqual(len(model.denoising_transformer.layers), 2)
        self.assertEqual(model.sample_decoder.layers[0].in_features, 256 + 32)
        self.assertEqual(model.mean_decoder.layers[0].in_features, 256)
        self.assertEqual(model.var_decoder.layers[0].in_features, 256)

        trajectory = torch.randn(2, 10, 2)
        timesteps = torch.tensor([10, 20])
        samples, mean, variance = model(trajectory, timesteps, mod=trajectory)
        self.assertEqual(samples.shape, (2, 2, 10, 2))
        self.assertEqual(mean.shape, (2, 10, 2))
        self.assertEqual(variance.shape, (2, 10, 1))

        predicted_noise = model.generate_accelerate(
            trajectory, torch.tensor([0.01, 0.02]), trajectory
        )
        self.assertEqual(predicted_noise.shape, trajectory.shape)

    @unittest.skipUnless(torch.cuda.is_available(), "Mamba reverse encoder requires CUDA")
    def test_nba_condtraj_uses_mamba_reverse_encoder(self):
        from model.mamba_model import Block

        model = CondTraj(
            input_feats=2,
            num_frames=20,
            latent_dim=512,
            num_layers=2,
            num_heads=2,
            num_sample=20,
            dropout=0.2,
            reverse_encoder="mamba",
        ).cuda().eval()
        self.assertEqual(model.reverse_encoder, "mamba")
        self.assertIsInstance(model.context_encoder, Block)
        self.assertIsInstance(model.mamba_block1, Block)
        self.assertIsInstance(model.mamba_block2, Block)
        trajectory = torch.randn(2, 20, 2, device="cuda")
        predicted_noise = model.generate_accelerate(
            trajectory, torch.tensor([0.01, 0.02], device="cuda"), trajectory
        )
        self.assertEqual(predicted_noise.shape, trajectory.shape)

    def test_step_lr_configuration_is_complete(self):
        expected_steps = {"sdd": 10, "nba": 8}
        for dataset, expected_step in expected_steps.items():
            config = OmegaConf.load(f"configs/{dataset}.yml")
            self.assertEqual(config.decay_step, expected_step)
            self.assertEqual(config.decay_gamma, 0.5)

    def test_generator_configuration_is_consistent(self):
        for dataset in ("sdd", "nba"):
            config = OmegaConf.load(f"configs/{dataset}.yml")
            self.assertEqual(config.num_epoch, 100)
            self.assertGreater(config.denoising_step, 0)
            self.assertGreater(config.latent_dims, 0)
            self.assertGreaterEqual(config.hyper_param1, 0.0)
            self.assertGreaterEqual(config.hyper_param2, 0.0)

    def test_nba_specific_training_configuration(self):
        nba = OmegaConf.load("configs/nba.yml")
        sdd = OmegaConf.load("configs/sdd.yml")
        self.assertEqual(nba.batchsize, 16)
        self.assertEqual(sdd.batchsize, 64)

    def test_nba_metrics_report_all_four_horizons(self):
        trainer = Trainer.__new__(Trainer)
        trainer.cfg = SimpleNamespace(dataset_type="NBA")
        output = trainer.format_metrics(
            1.0,
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.4, 0.5],
        )
        for second in range(1, 5):
            self.assertIn(f"minADE({second}s)", output)
            self.assertIn(f"minFDE({second}s)", output)

    def test_trainable_parameter_counter(self):
        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
        model[1].weight.requires_grad_(False)
        expected = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        self.assertEqual(print_trainable_parameters("TestModel", model), expected)


if __name__ == "__main__":
    unittest.main()
