from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import RestorationTestDataset, resolve_dataset_root
from model_promptir import PromptIRNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Inference for restoration test set")
    parser.add_argument(
        "--data-root",
        type=str,
        default=".",
        help="Dataset root or parent folder containing test/degraded",
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--out", type=str, default="pred.npz", help="Output npz path")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision for inference")
    parser.add_argument(
        "--tile-size",
        type=int,
        default=-1,
        help="Tile size for memory-safe inference; -1 disables tiling",
    )
    parser.add_argument(
        "--tile-overlap",
        type=int,
        default=32,
        help="Overlap in pixels between tiles (used when --tile-size > 0)",
    )
    return parser.parse_args()


def _forward_once(model: PromptIRNet, x: torch.Tensor, use_amp: bool) -> torch.Tensor:
    with torch.amp.autocast("cuda", enabled=(use_amp and x.device.type == "cuda")):
        y = model(x)
    return y.clamp(0.0, 1.0)


def _forward_tiled(
    model: PromptIRNet,
    x: torch.Tensor,
    tile_size: int,
    tile_overlap: int,
    use_amp: bool,
) -> torch.Tensor:
    _, _, h, w = x.shape
    if tile_size <= 0 or (h <= tile_size and w <= tile_size):
        return _forward_once(model, x, use_amp)

    tile_overlap = max(0, min(tile_overlap, tile_size - 1))
    stride = max(1, tile_size - tile_overlap)

    output = torch.zeros_like(x, dtype=torch.float32)
    weight = torch.zeros((1, 1, h, w), device=x.device, dtype=torch.float32)

    for top in range(0, h, stride):
        for left in range(0, w, stride):
            bottom = min(top + tile_size, h)
            right = min(left + tile_size, w)
            top = max(0, bottom - tile_size)
            left = max(0, right - tile_size)

            patch = x[:, :, top:bottom, left:right]
            pred_patch = _forward_once(model, patch, use_amp).to(torch.float32)

            output[:, :, top:bottom, left:right] += pred_patch
            weight[:, :, top:bottom, left:right] += 1.0

    return (output / weight.clamp_min(1e-6)).clamp(0.0, 1.0)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.tile_size > 0:
        print(f"Using tiled inference: tile_size={args.tile_size}, overlap={args.tile_overlap}")

    dataset_root = resolve_dataset_root(args.data_root)
    print(f"Using dataset root: {dataset_root}")
    ds = RestorationTestDataset(dataset_root / "test" / "degraded")
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    model = PromptIRNet().to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    outputs: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for img, name in tqdm(loader, desc="Predict", ncols=100):
            img = img.to(device, non_blocking=True)
            pred = _forward_tiled(model, img, args.tile_size, args.tile_overlap, args.amp)
            pred_u8 = (pred * 255.0).round().to(torch.uint8)

            # Required format: (3, H, W) uint8
            arr = pred_u8[0].cpu().numpy()
            outputs[name[0]] = arr

    np.savez(args.out, **outputs)
    print(f"Saved {len(outputs)} images to {args.out}")


if __name__ == "__main__":
    main()
