import math
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


@torch.no_grad()
def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """
    pred and target are expected in [0, 1], shape (B, C, H, W).
    """
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3)).clamp_min(eps)
    psnr = 10.0 * torch.log10(1.0 / mse)
    return psnr.mean().item()


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def cosine_lr(step: int, total_steps: int, lr_max: float, lr_min: float = 1e-7) -> float:
    if total_steps <= 1:
        return lr_max
    ratio = step / (total_steps - 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * ratio))
