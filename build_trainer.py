"""The two training stages described in Section IV-D."""

from __future__ import annotations

import os
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.data_process import NBADataset, SDDDataset
from data.motion_patterns import MotionPatternClusterer
from tool.tools import (
    NBAdata_process,
    SDDdata_process,
    dct_transform,
    ddpm_sample_loop_accelerate,
    idct_transform,
    noise_motion,
    normalize_traj,
    print_log,
    sample_timesteps,
)


def build_datasets(cfg, clusterer_state=None):
    clusterer = (
        MotionPatternClusterer.from_state_dict(clusterer_state)
        if clusterer_state is not None
        else None
    )
    common = dict(
        t=cfg.Iter,
        k=cfg.pattern,
        e=cfg.epsilon,
        random_seed=cfg.manual_seed,
    )
    if cfg.dataset_type == "SDD":
        train = SDDDataset(
            cfg.data_path,
            cfg.obs_len,
            cfg.pred_len,
            mode="train",
            flip_aug=False,
            clusterer=clusterer,
            **common,
        )
        test = SDDDataset(
            cfg.data_path,
            cfg.obs_len,
            cfg.pred_len,
            mode="test",
            clusterer=train.clusterer,
            **common,
        )
    elif cfg.dataset_type == "NBA":
        train = NBADataset(
            cfg.data_path,
            cfg.obs_len,
            cfg.pred_len,
            training=True,
            clusterer=clusterer,
            **common,
        )
        test = NBADataset(
            cfg.data_path,
            cfg.obs_len,
            cfg.pred_len,
            training=False,
            clusterer=train.clusterer,
            **common,
        )
    else:
        raise ValueError(f"unknown dataset type: {cfg.dataset_type}")
    return train, test


def build_test_dataset(cfg, clusterer_state):
    clusterer = MotionPatternClusterer.from_state_dict(clusterer_state)
    common = dict(
        t=cfg.Iter,
        k=cfg.pattern,
        e=cfg.epsilon,
        random_seed=cfg.manual_seed,
        clusterer=clusterer,
    )
    if cfg.dataset_type == "SDD":
        return SDDDataset(
            cfg.data_path,
            cfg.obs_len,
            cfg.pred_len,
            mode="test",
            flip_aug=False,
            **common,
        )
    if cfg.dataset_type == "NBA":
        return NBADataset(
            cfg.data_path, cfg.obs_len, cfg.pred_len, training=False, **common
        )
    raise ValueError(f"unknown dataset type: {cfg.dataset_type}")


def build_loaders(cfg, train_dataset, test_dataset, test_batch_size=None):
    test_batch_size = test_batch_size or cfg.batchsize
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batchsize,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    return train_loader, test_loader


def process_batch(cfg, data):
    if cfg.dataset_type == "SDD":
        result = SDDdata_process(cfg, data)
        trajectory, padded, future, labels = result[0], result[1], result[2], result[4]
    elif cfg.dataset_type == "NBA":
        result = NBAdata_process(cfg, data)
        trajectory, padded, future, labels = result[0], result[1], result[2], result[4]
    else:
        raise ValueError(f"unknown dataset type: {cfg.dataset_type}")
    return trajectory, padded, future, labels


class ClassifierTrainer:
    """Stage one: noisy-DCT classifier training with CRF NLL."""

    def __init__(self, cfg, classifier, dct_m, log, checkpoint_path):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier = classifier.to(self.device)
        self.dct_m = dct_m.float().to(self.device)
        self.log = log
        self.checkpoint_path = checkpoint_path
        self.train_dataset, self.test_dataset = build_datasets(cfg)
        self.train_loader, self.test_loader = build_loaders(
            cfg, self.train_dataset, self.test_dataset
        )
        self.optimizer = torch.optim.AdamW(
            classifier.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.cfg.decay_step, gamma=self.cfg.decay_gamma)

    def _run_epoch(self, loader, training):
        self.classifier.train(training)
        total_loss = 0.0
        total_correct = 0
        total_labels = 0
        total_trajectories = 0
        for data in tqdm(loader, leave=False):
            trajectory, _, _, labels = process_batch(self.cfg, data)
            dct = dct_transform(trajectory, self.dct_m, self.cfg.n_pre)
            timesteps = sample_timesteps(
                len(dct), self.cfg.noise_steps, device=self.device
            )
            noised, _ = noise_motion(dct, timesteps, self.cfg)
            with torch.set_grad_enabled(training):
                emissions = self.classifier(noised, timesteps)
                loss = self.classifier.crf_loss(emissions, labels)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), 1.0)
                    self.optimizer.step()
            decoded = self.classifier.decode(emissions.detach())
            total_loss += loss.item() * len(dct)
            total_trajectories += len(dct)
            total_correct += (decoded == labels).sum().item()
            total_labels += labels.numel()
        return total_loss / total_trajectories, total_correct / total_labels

    def loop(self):
        best_loss = float("inf")
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        for epoch in range(1, self.cfg.classifier_epochs + 1):
            train_loss, train_accuracy = self._run_epoch(self.train_loader, True)
            val_loss, val_accuracy = self._run_epoch(self.test_loader, False)
            print_log(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] classifier {epoch:03d} "
                f"train_nll={train_loss:.4f} train_acc={train_accuracy:.4f} "
                f"val_nll={val_loss:.4f} val_acc={val_accuracy:.4f}",
                self.log,
            )
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(
                    {
                        "classifier_dict": self.classifier.state_dict(),
                        "motion_pattern_state": self.train_dataset.clusterer.state_dict(),
                        "motion_pattern_semantics": self.train_dataset.clusterer.semantic_descriptions(),
                    },
                    self.checkpoint_path,
                )
            self.scheduler.step()


class Trainer:
    """Stage two: end-to-end trajectory generation with frozen classifier guidance."""

    def __init__(
        self,
        cfg,
        model,
        classifier,
        dct_m,
        idct_m,
        log,
        checkpoint_path,
        clusterer_state,
        evaluation_only=False,
    ):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.classifier = classifier.to(self.device).eval()
        self.classifier.requires_grad_(False)
        self.dct_m = dct_m.float().to(self.device)
        self.idct_m = idct_m.float().to(self.device)
        self.log = log
        self.checkpoint_path = checkpoint_path
        if evaluation_only:
            test_dataset = build_test_dataset(cfg, clusterer_state)
            self.train_loader = None
            self.test_loader = DataLoader(
                test_dataset,
                batch_size=cfg.batchsize,
                shuffle=False,
                num_workers=cfg.num_workers,
            )
            self.optimizer = None
            self.scheduler = None
        else:
            datasets = build_datasets(cfg, clusterer_state=clusterer_state)
            self.train_loader, self.test_loader = build_loaders(
                cfg, *datasets, test_batch_size=cfg.batchsize
            )
            self.optimizer = torch.optim.AdamW(
                model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
            )
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.cfg.decay_step, gamma=self.cfg.decay_gamma)

    def cond_fn(self, x, timesteps, labels, exp):
        with torch.enable_grad():
            x_input = x if exp == "Train" else x.detach().requires_grad_(True)
            emissions = self.classifier(x_input, timesteps)
            log_probability = self.classifier.sequence_log_probability(emissions, labels)
            return torch.autograd.grad(
                log_probability.sum(),
                x_input,
                retain_graph=exp == "Test",
            )[0] * self.cfg.scale

    def generate(self, trajectory, padded, exp):
        dct = dct_transform(trajectory, self.dct_m, self.cfg.n_pre)
        history_dct = dct_transform(padded, self.dct_m, self.cfg.n_pre)
        timesteps = sample_timesteps(
            len(dct), self.cfg.noise_steps, device=self.device
        )
        c_d, _ = noise_motion(dct, timesteps, self.cfg)
        samples, mean, log_variance = self.model(
            c_d,
            timesteps,
            mod=history_dct,
        )
        candidates = normalize_traj(c_d, samples, mean, log_variance)

        predictions = []
        for index in range(self.cfg.num_sample):
            with torch.no_grad():
                emissions = self.classifier(
                    candidates[:, index], timesteps, mod=history_dct
                )
                pattern_labels = self.classifier.decode(emissions)
            prediction = ddpm_sample_loop_accelerate(
                self.cfg,
                self.model,
                self.cond_fn,
                history_dct,
                candidates[:, index],
                self.dct_m,
                self.idct_m,
                pattern_labels,
                exp,
            )
            predictions.append(prediction[:, None])
        prediction_dct = torch.cat(predictions, dim=1)
        prediction_time = idct_transform(prediction_dct, self.idct_m, self.cfg.n_pre)
        return prediction_time[:, :, self.cfg.obs_len:]

    @staticmethod
    def marginal_loss(predictions, future):
        per_sample = torch.linalg.vector_norm(predictions - future[:, None], dim=-1).sum(dim=-1)
        return per_sample.min(dim=1).values.mean()

    @staticmethod
    def joint_loss(predictions, future):
        per_sample = torch.linalg.vector_norm(
            predictions - future[:, None], dim=-1
        ).sum(dim=-1)
        return per_sample.mean(dim=0).min()

    @staticmethod
    def average_pairwise_distance(predictions):
        if predictions.shape[1] <= 1:
            return 0.0
        flattened = predictions.reshape(predictions.shape[0], predictions.shape[1], -1)
        return sum(torch.pdist(batch).mean().item() for batch in flattened)

    def _train_epoch(self):
        self.model.train()
        total = 0.0
        total_trajectories = 0
        for data in tqdm(self.train_loader, leave=False):
            trajectory, padded, future, _ = process_batch(self.cfg, data)
            predictions = self.generate(trajectory, padded, "Train")
            marginal = self.marginal_loss(predictions, future)
            joint = self.joint_loss(predictions, future)
            marginal_weight = float(self.cfg.get("hyper_param1", 1.0))
            joint_weight = float(self.cfg.get("hyper_param2", 0.0))
            loss = marginal * marginal_weight + joint * joint_weight
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total += loss.item() * len(trajectory)
            total_trajectories += len(trajectory)
        return total / total_trajectories

    def _validate(self):
        self.model.eval()
        total_apd = 0.0
        total_ade = 0.0
        total_fde = 0.0
        samples = 0
        for data in tqdm(self.test_loader, leave=False):
            trajectory, padded, future, _ = process_batch(self.cfg, data)
            with torch.enable_grad():
                predictions = self.generate(trajectory, padded, "Test")
            predictions = predictions.detach()
            distances = torch.linalg.vector_norm(predictions - future[:, None], dim=-1)
            total_apd += self.average_pairwise_distance(predictions)
            total_ade += distances.mean(dim=-1).min(dim=1).values.sum().item()
            total_fde += distances[:, :, -1].min(dim=1).values.sum().item()
            samples += len(trajectory)
        return (
            total_apd / samples,
            total_ade / samples * self.cfg.data_scale,
            total_fde / samples * self.cfg.data_scale,
        )

    def loop(self):
        best_ade = float("inf")
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        for epoch in range(1, self.cfg.num_epoch + 1):
            train_loss = self._train_epoch()
            self.scheduler.step()
            message = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] generator {epoch:03d} loss={train_loss:.4f}"
            if epoch % self.cfg.eval_iter == 0:
                apd, ade, fde = self._validate()
                message += f" APD={apd:.4f} minADE={ade:.4f} minFDE={fde:.4f}"
                if ade < best_ade:
                    best_ade = ade
                    torch.save({"model_dict": self.model.state_dict()}, self.checkpoint_path)
            print_log(message, self.log)
