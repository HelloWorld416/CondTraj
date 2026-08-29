"""Stepwise motion-pattern classifier with a linear-chain CRF."""

from __future__ import annotations

import math

import torch
from torch import nn


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    arguments = timesteps[:, None].float() * frequencies[None]
    embedding = torch.cat([torch.cos(arguments), torch.sin(arguments)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class LinearChainCRF(nn.Module):
    """Nominal-label CRF used by the stage-one classifier."""

    def __init__(self, num_labels: int):
        super().__init__()
        self.num_labels = num_labels
        self.start_transitions = nn.Parameter(torch.empty(num_labels))
        self.end_transitions = nn.Parameter(torch.empty(num_labels))
        self.transitions = nn.Parameter(torch.empty(num_labels, num_labels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def _validate(self, emissions: torch.Tensor, labels: torch.Tensor | None = None) -> None:
        if emissions.ndim != 3 or emissions.shape[-1] != self.num_labels:
            raise ValueError("emissions must have shape [B, T, K]")
        if labels is not None:
            if labels.shape != emissions.shape[:2]:
                raise ValueError("labels must have shape [B, T]")
            if labels.dtype != torch.long:
                raise ValueError("labels must use torch.long dtype")

    def log_partition(self, emissions: torch.Tensor) -> torch.Tensor:
        self._validate(emissions)
        score = self.start_transitions + emissions[:, 0]
        for index in range(1, emissions.shape[1]):
            candidates = score.unsqueeze(2) + self.transitions.unsqueeze(0)
            score = torch.logsumexp(candidates, dim=1) + emissions[:, index]
        return torch.logsumexp(score + self.end_transitions, dim=1)

    def sequence_score(self, emissions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        self._validate(emissions, labels)
        batch = torch.arange(emissions.shape[0], device=emissions.device)
        score = self.start_transitions[labels[:, 0]]
        score = score + emissions[batch, 0, labels[:, 0]]
        for index in range(1, emissions.shape[1]):
            score = score + self.transitions[labels[:, index - 1], labels[:, index]]
            score = score + emissions[batch, index, labels[:, index]]
        return score + self.end_transitions[labels[:, -1]]

    def log_probability(self, emissions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.sequence_score(emissions, labels) - self.log_partition(emissions)

    def neg_log_likelihood(self, emissions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return -self.log_probability(emissions, labels).mean()

    @torch.no_grad()
    def decode(self, emissions: torch.Tensor) -> torch.Tensor:
        self._validate(emissions)
        score = self.start_transitions + emissions[:, 0]
        backpointers = []
        for index in range(1, emissions.shape[1]):
            candidates = score.unsqueeze(2) + self.transitions.unsqueeze(0)
            best_score, best_previous = candidates.max(dim=1)
            score = best_score + emissions[:, index]
            backpointers.append(best_previous)
        current = (score + self.end_transitions).argmax(dim=1)
        decoded = [current]
        for best_previous in reversed(backpointers):
            current = best_previous.gather(1, current[:, None]).squeeze(1)
            decoded.append(current)
        return torch.stack(list(reversed(decoded)), dim=1)


class Classifier(nn.Module):
    """Map noisy truncated-DCT trajectories to stepwise pattern emissions."""

    def __init__(
        self,
        input_feats: int = 2,
        num_frames: int = 10,
        label_frames: int = 20,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.2,
        output_dim: int = 3,
        **_: object,
    ):
        super().__init__()
        self.input_feats = input_feats
        self.num_frames = num_frames
        self.label_frames = label_frames
        self.latent_dim = latent_dim

        self.spectral_embedding = nn.Linear(input_feats, latent_dim)
        self.spectral_to_temporal = nn.Linear(num_frames, label_frames)
        self.sequence_embedding = nn.Parameter(torch.randn(label_frames, latent_dim) * 0.02)
        self.time_embedding = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.SiLU(), nn.Linear(latent_dim, latent_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(latent_dim)
        self.emission_head = nn.Linear(latent_dim, output_dim)
        self.crf = LinearChainCRF(output_dim)

    def forward(
        self, x: torch.Tensor, timesteps: torch.Tensor, mod: torch.Tensor | None = None
    ) -> torch.Tensor:
        if (
            x.ndim != 3
            or x.shape[1] != self.num_frames
            or x.shape[2] != self.input_feats
        ):
            raise ValueError(
                f"x must have shape [B, {self.num_frames}, {self.input_feats}]"
            )
        hidden = self.spectral_embedding(x).transpose(1, 2)
        hidden = self.spectral_to_temporal(hidden).transpose(1, 2)
        if mod is not None:
            if mod.shape != x.shape:
                raise ValueError("history DCT context must match the noisy DCT shape")
            context = self.spectral_embedding(mod).transpose(1, 2)
            hidden = hidden + self.spectral_to_temporal(context).transpose(1, 2)
        time = self.time_embedding(timestep_embedding(timesteps, self.latent_dim))
        hidden = hidden + time[:, None] + self.sequence_embedding[None]
        hidden = self.temporal_encoder(hidden)
        return self.emission_head(self.output_norm(hidden))

    def crf_loss(self, emissions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.crf.neg_log_likelihood(emissions, labels)

    def sequence_log_probability(
        self, emissions: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        return self.crf.log_probability(emissions, labels)

    def decode(self, emissions: torch.Tensor) -> torch.Tensor:
        return self.crf.decode(emissions)
