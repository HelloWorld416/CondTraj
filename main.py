"""CondTraj two-stage training and best-of-20 evaluation entry point."""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
from omegaconf import OmegaConf

from build_trainer import ClassifierTrainer, Trainer
from model.classifier import Classifier
from model.denoising_model import CondTraj
from tool.tools import generate_pad, get_dct_matrix


def prepare_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_trainable_parameters(name: str, model: torch.nn.Module) -> int:
    parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f">>>>>> {name} trainable parameters: {parameters:,} ({parameters / 1_000_000:.2f}M)")
    return parameters


def reverse_encoder_for_dataset(dataset_type: str) -> str:
    return "mamba" if dataset_type.upper() == "NBA" else "transformer"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="./data")
    parser.add_argument(
        "--mode", choices=("train_classifier", "train", "eval"), default="eval"
    )
    parser.add_argument("--log-dir", default="./logs")
    parser.add_argument("--manual-seed", type=int, default=0)
    parser.add_argument("--eval-iter", type=int, default=2)
    parser.add_argument("--cfg", default="./configs/nba.yml")
    parser.add_argument("--classifier-ckpt")
    parser.add_argument("--model-ckpt")
    return parser.parse_args()


def main():
    args = parse_args()
    prepare_seed(args.manual_seed)
    cfg = OmegaConf.load(args.cfg)
    cfg.data_path = args.data_path
    cfg.mode = args.mode
    cfg.log_dir = args.log_dir
    cfg.manual_seed = args.manual_seed
    cfg.eval_iter = args.eval_iter
    cfg.idx_pad, cfg.zero_index = generate_pad(cfg.padding, cfg.obs_len, cfg.pred_len)

    checkpoint_dir = os.path.join("checkpoint", cfg.dataset_type)
    classifier_path = args.classifier_ckpt or os.path.join(
        checkpoint_dir, "classifier.pt"
    )
    model_path = args.model_ckpt or os.path.join(
        checkpoint_dir, "condtraj.pt"
    )
    log_path = os.path.join(cfg.log_dir, cfg.dataset_type)
    os.makedirs(log_path, exist_ok=True)
    log = open(os.path.join(log_path, "log.txt"), "a", encoding="utf-8")
    OmegaConf.save(cfg, os.path.join(log_path, "config.yaml"))

    dct_m, idct_m = get_dct_matrix(cfg.obs_len + cfg.pred_len)
    classifier = Classifier(
        input_feats=2,
        num_frames=cfg.n_pre,
        label_frames=cfg.obs_len + cfg.pred_len,
        latent_dim=cfg.cls_latent_dims,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
        output_dim=cfg.pattern,
    )

    if args.mode == "train_classifier":
        print_trainable_parameters("Classifier", classifier)
        ClassifierTrainer(cfg, classifier, dct_m, log, classifier_path).loop()
        return

    if not os.path.isfile(classifier_path):
        raise FileNotFoundError(
            f"Classifier checkpoint not found: {classifier_path}. "
            "Run --mode train_classifier first."
        )
    classifier_checkpoint = torch.load(classifier_path, map_location="cpu")
    classifier.load_state_dict(classifier_checkpoint["classifier_dict"], strict=True)
    clusterer_state = classifier_checkpoint["motion_pattern_state"]

    model = CondTraj(
        input_feats=2,
        num_frames=cfg.n_pre,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        latent_dim=cfg.latent_dims,
        dropout=cfg.dropout,
        num_sample=cfg.num_sample,
        reverse_encoder=reverse_encoder_for_dataset(cfg.dataset_type),
    )
    if args.mode == "train":
        print_trainable_parameters("CondTraj", model)
    trainer = Trainer(
        cfg,
        model,
        classifier,
        dct_m,
        idct_m,
        log,
        model_path,
        clusterer_state,
        evaluation_only=args.mode == "eval",
    )
    if args.mode == "train":
        trainer.loop()
    else:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Generator checkpoint not found: {model_path}")
        checkpoint = torch.load(model_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_dict"], strict=True)
        apd, ade, fde = trainer._validate()
        print(f"best-of-{cfg.num_sample} {trainer.format_metrics(apd, ade, fde)}")


if __name__ == "__main__":
    main()
