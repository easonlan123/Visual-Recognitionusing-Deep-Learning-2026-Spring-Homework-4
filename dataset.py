from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


@dataclass(frozen=True)
class PairItem:
    degraded_path: Path
    clean_path: Path


class RestorationTrainDataset(Dataset):
    def __init__(
        self,
        pairs: list[PairItem],
        patch_size: int = 256,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.2,
        rot_prob: float = 0.2,
    ) -> None:
        self.pairs = pairs
        self.patch_size = patch_size
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rot_prob = rot_prob

    def __len__(self) -> int:
        return len(self.pairs)

    @staticmethod
    def _read_rgb(path: Path) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        return np.array(img, dtype=np.uint8)

    def _random_crop(self, deg: np.ndarray, clean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w, _ = deg.shape
        ps = self.patch_size
        if h < ps or w < ps:
            pad_h = max(0, ps - h)
            pad_w = max(0, ps - w)
            deg = np.pad(deg, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            clean = np.pad(clean, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            h, w, _ = deg.shape

        top = random.randint(0, h - ps)
        left = random.randint(0, w - ps)
        return deg[top : top + ps, left : left + ps], clean[top : top + ps, left : left + ps]

    def _augment(self, deg: np.ndarray, clean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if random.random() < self.hflip_prob:
            deg = np.fliplr(deg)
            clean = np.fliplr(clean)
        if random.random() < self.vflip_prob:
            deg = np.flipud(deg)
            clean = np.flipud(clean)
        if random.random() < self.rot_prob:
            k = random.randint(1, 3)
            deg = np.rot90(deg, k)
            clean = np.rot90(clean, k)
        return deg.copy(), clean.copy()

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.pairs[idx]
        deg = self._read_rgb(item.degraded_path)
        clean = self._read_rgb(item.clean_path)

        deg, clean = self._random_crop(deg, clean)
        deg, clean = self._augment(deg, clean)

        deg_t = torch.from_numpy(deg).permute(2, 0, 1).float() / 255.0
        clean_t = torch.from_numpy(clean).permute(2, 0, 1).float() / 255.0
        return deg_t, clean_t


class RestorationValDataset(Dataset):
    def __init__(self, pairs: list[PairItem], patch_size: int | None = None) -> None:
        self.pairs = pairs
        self.patch_size = patch_size

    def __len__(self) -> int:
        return len(self.pairs)

    @staticmethod
    def _read_rgb(path: Path) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        return np.array(img, dtype=np.uint8)

    def _center_crop_or_pad(
        self, deg: np.ndarray, clean: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.patch_size is None:
            return deg, clean

        ps = self.patch_size
        h, w, _ = deg.shape
        if h < ps or w < ps:
            pad_h = max(0, ps - h)
            pad_w = max(0, ps - w)
            deg = np.pad(deg, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            clean = np.pad(clean, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            h, w, _ = deg.shape

        top = max(0, (h - ps) // 2)
        left = max(0, (w - ps) // 2)
        return deg[top : top + ps, left : left + ps], clean[top : top + ps, left : left + ps]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.pairs[idx]
        deg = self._read_rgb(item.degraded_path)
        clean = self._read_rgb(item.clean_path)
        deg, clean = self._center_crop_or_pad(deg, clean)
        deg_t = torch.from_numpy(deg).permute(2, 0, 1).float() / 255.0
        clean_t = torch.from_numpy(clean).permute(2, 0, 1).float() / 255.0
        return deg_t, clean_t


class RestorationTestDataset(Dataset):
    def __init__(self, degraded_dir: str | Path) -> None:
        self.degraded_dir = Path(degraded_dir)
        files = [
            p for p in self.degraded_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS
        ]

        # Test files are expected to be 0.png ... 99.png; sort numerically if possible.
        def _sort_key(p: Path):
            stem = p.stem
            return (0, int(stem)) if stem.isdigit() else (1, stem)

        self.files = sorted(files, key=_sort_key)

    def __len__(self) -> int:
        return len(self.files)

    @staticmethod
    def _read_rgb(path: Path) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        return np.array(img, dtype=np.uint8)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        path = self.files[idx]
        img = self._read_rgb(path)
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img_t, path.name


def resolve_dataset_root(root: str | Path = ".") -> Path:
    """
    Accept either the dataset root itself, or a parent folder containing it.
    The first folder that has train/degraded, train/clean, and test/degraded is used.
    """
    root = Path(root).resolve()
    candidates = [root]

    # Common case: user passes workspace root and dataset is one level below.
    for child in root.iterdir() if root.exists() else []:
        if child.is_dir():
            candidates.append(child)

    for c in candidates:
        if (c / "train" / "degraded").exists() and (c / "train" / "clean").exists() and (
            c / "test" / "degraded"
        ).exists():
            return c

    raise RuntimeError(
        "Could not find dataset root. Expected folders train/degraded, train/clean, test/degraded. "
        f"Searched under: {root}"
    )


def _find_clean_match(clean_dir: Path, prefix: str, idx: str, ext: str) -> Path | None:
    # Try the most likely names first.
    candidates = [
        clean_dir / f"{prefix}_clean-{idx}{ext}",
        clean_dir / f"{prefix}_clean-{idx}.png",
        clean_dir / f"{prefix}-clean-{idx}{ext}",
        clean_dir / f"{prefix}-clean-{idx}.png",
    ]

    for c in candidates:
        if c.exists():
            return c

    # Fallback: any file starting with <prefix> and ending with -<idx><ext>
    for p in clean_dir.iterdir():
        if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
            continue
        stem = p.stem.lower()
        if stem.startswith(prefix) and stem.endswith(f"-{idx}"):
            return p

    return None


def _collect_train_pairs(root: Path) -> list[PairItem]:
    degraded_dir = root / "train" / "degraded"
    clean_dir = root / "train" / "clean"

    pairs: list[PairItem] = []
    for deg_path in sorted(degraded_dir.iterdir()):
        if not deg_path.is_file() or deg_path.suffix.lower() not in IMG_EXTS:
            continue

        name = deg_path.stem.lower()
        ext = deg_path.suffix.lower()

        if name.startswith("rain-"):
            idx = name.split("rain-")[-1]
            clean_path = _find_clean_match(clean_dir, "rain", idx, ext)
        elif name.startswith("snow-"):
            idx = name.split("snow-")[-1]
            clean_path = _find_clean_match(clean_dir, "snow", idx, ext)
        else:
            continue

        if clean_path is not None and clean_path.exists():
            pairs.append(PairItem(degraded_path=deg_path, clean_path=clean_path))

    if len(pairs) == 0:
        raise RuntimeError(
            f"No train pairs found under {root}. Check folder names and naming convention."
        )
    return pairs


def build_train_val_pairs(root: str | Path, val_ratio: float = 0.05, seed: int = 42):
    root = resolve_dataset_root(root)
    pairs = _collect_train_pairs(root)
    rng = random.Random(seed)
    rng.shuffle(pairs)

    if val_ratio <= 0:
        return pairs, []

    val_count = max(1, int(len(pairs) * val_ratio))
    val_pairs = pairs[:val_count]
    train_pairs = pairs[val_count:]
    return train_pairs, val_pairs
