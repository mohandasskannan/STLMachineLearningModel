"""Small float32 models intended for a 6 GB GPU."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class ShapeVAE(nn.Module):
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, 4, 2, 1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv3d(16, 32, 4, 2, 1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv3d(32, 64, 4, 2, 1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Flatten(),
        )
        self.mu = nn.Linear(64 * 4**3, latent_dim)
        self.logvar = nn.Linear(64 * 4**3, latent_dim)
        self.project = nn.Linear(latent_dim, 64 * 4**3)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 4, 2, 1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.ConvTranspose3d(32, 16, 4, 2, 1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.ConvTranspose3d(16, 1, 4, 2, 1),
        )

    def encode(self, x):
        hidden = self.encoder(x)
        return self.mu(hidden), self.logvar(hidden).clamp(-12, 8)

    def decode(self, z):
        return self.decoder(self.project(z).reshape(-1, 64, 4, 4, 4))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        return self.decode(z), mu, logvar


def vae_loss(logits, target, mu, logvar, beta: float):
    # Soft Dice discourages the trivial all-empty solution without changing BCE calibration.
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = logits.sigmoid().flatten(1)
    y = target.flatten(1)
    dice = 1 - ((2 * (p * y).sum(1) + 1) / (p.sum(1) + y.sum(1) + 1)).mean()
    kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).mean()
    loss = bce + dice + beta * kl
    return loss, {
        "loss": loss.detach(),
        "bce": bce.detach(),
        "dice_loss": dice.detach(),
        "kl": kl.detach(),
    }


def voxel_iou(logits, target):
    prediction = logits > 0
    truth = target > 0.5
    intersection = (prediction & truth).flatten(1).sum(1)
    union = (prediction | truth).flatten(1).sum(1)
    return ((intersection + 1e-6) / (union + 1e-6)).mean()


class Denoiser(nn.Module):
    def __init__(self, latent_dim: int = 256, text_dim: int = 384, width: int = 512):
        super().__init__()
        self.latent_dim, self.text_dim, self.width = latent_dim, text_dim, width
        self.input = nn.Linear(latent_dim + text_dim + 64, width)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(nn.LayerNorm(width), nn.SiLU(), nn.Linear(width, width))
                for _ in range(4)
            ]
        )
        self.output = nn.Linear(width, latent_dim)

    def forward(self, z, t, text):
        frequencies = torch.exp(torch.arange(32, device=z.device) * (-math.log(10000) / 31))
        angles = t.float()[:, None] * frequencies[None]
        time = torch.cat((angles.sin(), angles.cos()), dim=1)
        hidden = self.input(torch.cat((z, text, time), dim=1))
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return self.output(F.silu(hidden))


class Diffusion:
    def __init__(self, steps: int = 200, device="cpu"):
        if steps < 2:
            raise ValueError("Diffusion requires at least two timesteps")
        self.steps = steps
        # Cosine schedule reaches nearly pure noise even with only 200 steps.
        x = torch.linspace(0, steps, steps + 1, device=device, dtype=torch.float64)
        cumulative = torch.cos(((x / steps + 0.008) / 1.008) * math.pi / 2).square()
        cumulative = cumulative / cumulative[0]
        self.betas = (1 - cumulative[1:] / cumulative[:-1]).clamp(0.0001, 0.999).float()
        self.alpha_bar = torch.cumprod(1 - self.betas, dim=0)

    def noisy(self, z, t, noise):
        alpha = self.alpha_bar[t, None]
        return alpha.sqrt() * z + (1 - alpha).sqrt() * noise

    @torch.inference_mode()
    def sample(self, model, text, seed: int, sampling_steps: int = 50):
        if not 2 <= sampling_steps <= self.steps:
            raise ValueError(f"Sampling steps must be between 2 and {self.steps}")
        generator = torch.Generator(device=text.device).manual_seed(seed)
        z = torch.randn((len(text), model.latent_dim), generator=generator, device=text.device)
        times = torch.linspace(self.steps - 1, 0, sampling_steps).round().long().tolist()
        # Deterministic DDIM: the seed selects initial noise; no extra reverse-step noise.
        for i, time in enumerate(times):
            timestep = torch.full((len(text),), time, device=text.device, dtype=torch.long)
            noise = model(z, timestep, text)
            alpha = self.alpha_bar[time]
            previous = self.alpha_bar[times[i + 1]] if i + 1 < len(times) else z.new_tensor(1.0)
            clean = (z - (1 - alpha).sqrt() * noise) / alpha.sqrt()
            z = previous.sqrt() * clean + (1 - previous).sqrt() * noise
        return z
