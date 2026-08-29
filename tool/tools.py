"""Data, DCT, and diffusion utilities shared by training and evaluation."""

from __future__ import annotations

import math

import numpy as np
import torch


def _device_of(data) -> torch.device:
    for value in data.values():
        if torch.is_tensor(value):
            return value.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def NBAdata_process(cfg, data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    past = data["pre_motion_3D"].float().to(device)
    future = data["fut_motion_3D"].float().to(device)
    labels = data["cluster_label"].long().to(device)
    batch_size, actors = past.shape[:2]
    initial_pos = past[:, :, -1:]
    full = torch.cat([past, future], dim=2)
    trajectory = ((full - initial_pos) / cfg.data_scale).reshape(-1, cfg.obs_len + cfg.pred_len, 2)
    padded = padding_traj(trajectory, cfg.padding, cfg.idx_pad, cfg.zero_index)
    future = ((future - initial_pos) / cfg.data_scale).reshape(-1, cfg.pred_len, 2)
    mask = torch.zeros(batch_size * actors, batch_size * actors, device=device)
    for index in range(batch_size):
        start = index * actors
        mask[start:start + actors, start:start + actors] = 1
    return trajectory, padded, future, mask, labels.reshape(-1, labels.shape[-1]), initial_pos


def SDDdata_process(cfg, data, mode="train"):
    del mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    observed = data["past_traj"].float().to(device)
    future_abs = data["fut_traj"].float().to(device)
    labels = data["cluster_label"].long().to(device)
    initial_pos = observed[:, -1:]
    trajectory = (torch.cat([observed, future_abs], dim=1) - initial_pos) / cfg.data_scale
    padded = padding_traj(trajectory, cfg.padding, cfg.idx_pad, cfg.zero_index)
    future = (future_abs - initial_pos) / cfg.data_scale
    mask = torch.eye(len(trajectory), device=device)
    return trajectory, padded, future, mask, labels, initial_pos


def make_beta_schedule(
    schedule: str = "cosine",
    n_timesteps: int = 1000,
    start: float = 1e-4,
    end: float = 5e-2,
    s: float = 0.008,
) -> torch.Tensor:
    name = schedule.lower()
    if name == "linear":
        betas = torch.linspace(start, end, n_timesteps)
    elif name == "sqrt":
        betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
    elif name == "sigmoid":
        values = torch.linspace(-6, 6, n_timesteps)
        betas = torch.sigmoid(values) * (end - start) + start
    elif name == "cosine":
        steps = n_timesteps + 1
        values = torch.linspace(0, n_timesteps, steps) / n_timesteps
        alpha_bar = torch.cos((values + s) / (1 + s) * math.pi * 0.5) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    else:
        raise ValueError(f"unknown diffusion schedule: {schedule}")
    return betas.clamp(1e-8, 0.999)


def diffusion_schedule(cfg, device: torch.device):
    betas = make_beta_schedule(cfg.scheduler, cfg.noise_steps).to(device)
    alphas = 1 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bar


def sample_timesteps(n: int, noise_steps: int = 1000, device=None) -> torch.Tensor:
    return torch.randint(1, noise_steps, (n,), device=device)


def noise_motion(x: torch.Tensor, timesteps: torch.Tensor, cfg, noise=None):
    _, _, alpha_bar = diffusion_schedule(cfg, x.device)
    alpha = alpha_bar[timesteps].view(-1, 1, 1)
    noise = torch.randn_like(x) if noise is None else noise
    return alpha.sqrt() * x + (1 - alpha).sqrt() * noise, noise


def _extract(values: torch.Tensor, timesteps: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return values.gather(0, timesteps).reshape(timesteps.shape[0], *([1] * (x.ndim - 1)))


def condition_mean(cond_fn, mean, variance, x, timesteps, labels, exp):
    gradient = cond_fn(x, timesteps, labels, exp)
    return mean + variance * gradient


def ddpm_sample_iter(cfg, model, current, step, history_dct, labels, exp, cond_fn=None):
    # The accelerated sampler follows the original 100-step linear reverse process.
    betas = make_beta_schedule(
        "linear", n_timesteps=100, start=1e-4, end=5e-2
    ).to(current.device)
    alphas = 1 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    timesteps = torch.full((current.shape[0],), step, dtype=torch.long, device=current.device)
    beta = _extract(betas, timesteps, current)
    alpha = _extract(alphas, timesteps, current)
    cumulative = _extract(alpha_bar, timesteps, current)
    predicted_noise = model.generate_accelerate(current, beta, history_dct)
    mean = (current - beta / (1 - cumulative).sqrt() * predicted_noise) / alpha.sqrt()
    if cond_fn is not None:
        guidance_timesteps = sample_timesteps(
            len(current), noise_steps=100, device=current.device
        )
        mean = condition_mean(
            cond_fn, mean, beta.sqrt(), current, guidance_timesteps, labels, exp
        )
    return mean + beta.sqrt() * torch.randn_like(current) * 1e-5


def mask_complete(cfg, history_dct, predicted_dct, dct_m, idct_m):
    observed_time = idct_transform(history_dct, idct_m, cfg.n_pre)
    predicted_time = idct_transform(predicted_dct, idct_m, cfg.n_pre)
    predicted_time[:, :cfg.obs_len] = observed_time[:, :cfg.obs_len]
    return dct_transform(predicted_time, dct_m, cfg.n_pre)


def ddpm_sample_loop_accelerate(
    cfg,
    model,
    cond_fn,
    history_dct,
    initial_dct,
    dct_m,
    idct_m,
    pattern_labels,
    exp,
):
    current = initial_dct
    for step in reversed(range(cfg.denoising_step)):
        current = ddpm_sample_iter(
            cfg, model, current, step, history_dct, pattern_labels, exp, cond_fn=cond_fn
        )
        current = mask_complete(cfg, history_dct, current, dct_m, idct_m)
    return current


def print_log(message, log, same_line=False, display=True):
    if display:
        print(message, end="" if same_line else "\n")
    log.write(message if same_line else message + "\n")
    log.flush()


def generate_pad(padding, t_his, t_pred):
    zero_index = None
    if padding == "Zero":
        indices = list(range(t_his)) + [t_his - 1] * t_pred
        zero_index = t_his - 1
    elif padding == "Repeat":
        repeats = math.ceil((t_his + t_pred) / t_his)
        indices = (list(range(t_his)) * repeats)[:t_his + t_pred]
    elif padding == "LastFrame":
        indices = list(range(t_his)) + [t_his - 1] * t_pred
    else:
        raise ValueError(f"unknown padding method: {padding}")
    return indices, zero_index


def padding_traj(trajectory, padding, indices, zero_index):
    if padding == "Zero":
        trajectory = trajectory.clone()
        trajectory[..., zero_index, :] = 0
    return trajectory[..., indices, :]


def get_dct_matrix(length, is_torch=True):
    dct = np.eye(length)
    for frequency in range(length):
        for index in range(length):
            weight = np.sqrt(1 / length) if frequency == 0 else np.sqrt(2 / length)
            dct[frequency, index] = weight * np.cos(np.pi * (index + 0.5) * frequency / length)
    inverse = dct.T
    if is_torch:
        return torch.from_numpy(dct), torch.from_numpy(inverse)
    return dct, inverse


def dct_transform(time_sequence, dct_matrix, n_frame):
    return torch.matmul(dct_matrix[:n_frame], time_sequence)


def idct_transform(frequency_sequence, idct_matrix, n_frame):
    return torch.matmul(idct_matrix[:, :n_frame], frequency_sequence)


def normalize_traj(x_t, trajectory_samples, distribution_mean, distribution_log_variance):
    del x_t
    sample_std = trajectory_samples.std(dim=1).mean(dim=(1, 2))
    scaled = (
        torch.exp(distribution_log_variance / 2)[:, None]
        * trajectory_samples
        / sample_std[:, None, None, None]
    )
    return scaled + distribution_mean[:, None]
