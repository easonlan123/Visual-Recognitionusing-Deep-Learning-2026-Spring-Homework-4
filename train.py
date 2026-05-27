from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import (
    RestorationTrainDataset,
    RestorationValDataset,
    build_train_val_pairs,
    resolve_dataset_root,
)
from model_promptir import PromptIRNet
from utils import calculate_psnr, count_parameters, cosine_lr, ensure_dir, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train PromptIR-style model from scratch")
    parser.add_argument(
        "--data-root",
        type=str,
        default=".",
        help="Dataset root or parent folder containing train/ and test/",
    )
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument(
        "--val-patch-size",
        type=int,
        default=0,
        help="Validation patch size: 0 uses --patch-size, -1 disables val cropping (full image)",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--init-ckpt",
        type=str,
        default="",
        help="Optional checkpoint path to initialize model for finetuning",
    )
    parser.add_argument(
        "--resume-ckpt",
        type=str,
        default="",
        help="Optional checkpoint path to resume model+optimizer state",
    )
    parser.add_argument("--amp", action="store_true", help="Use mixed precision")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
) -> tuple[float, float]:
    model.eval()
    l1 = nn.L1Loss(reduction="mean")
    total_loss = 0.0
    total_psnr = 0.0
    n = 0

    with torch.no_grad():
        for deg, clean in loader:
            deg = deg.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
                pred = model(deg)
            loss = l1(pred, clean)
            psnr = calculate_psnr(pred, clean)

            bs = deg.size(0)
            total_loss += loss.item() * bs
            total_psnr += psnr * bs
            n += bs

    return total_loss / max(n, 1), total_psnr / max(n, 1)


def main() -> None:
    args = parse_args()
    if args.init_ckpt and args.resume_ckpt:
        raise ValueError("Use only one of --init-ckpt or --resume-ckpt")

    seed_everything(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    save_dir = ensure_dir(args.save_dir)

    dataset_root = resolve_dataset_root(args.data_root)
    print(f"Using dataset root: {dataset_root}")
    train_pairs, val_pairs = build_train_val_pairs(
        dataset_root, val_ratio=args.val_ratio, seed=args.seed
    )
    print(f"Train pairs: {len(train_pairs)} | Val pairs: {len(val_pairs)}")

    train_ds = RestorationTrainDataset(train_pairs, patch_size=args.patch_size)
    val_patch_size = args.patch_size if args.val_patch_size == 0 else args.val_patch_size
    val_ds = None
    if len(val_pairs) > 0:
        val_ds = RestorationValDataset(
            val_pairs, patch_size=(None if val_patch_size < 0 else val_patch_size)
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=max(1, args.num_workers // 2),
            pin_memory=True,
        )

    model = PromptIRNet().to(device)
    print(f"Model params: {count_parameters(model) / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))

    l1 = nn.L1Loss(reduction="mean")
    steps_per_epoch = max(1, len(train_loader))
    global_step = 0
    start_epoch = 0

    best_psnr = -1.0
    best_train_l1 = float("inf")
    if args.resume_ckpt:
        resume_path = Path(args.resume_ckpt)
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume-ckpt not found: {resume_path}")

        resume_data = torch.load(resume_path, map_location=device)
        if not isinstance(resume_data, dict) or "model" not in resume_data:
            raise RuntimeError("--resume-ckpt must be a checkpoint dict containing key 'model'")

        model.load_state_dict(resume_data["model"], strict=True)
        if "optimizer" in resume_data:
            optimizer.load_state_dict(resume_data["optimizer"])
        if "scaler" in resume_data and args.amp and device.type == "cuda":
            scaler.load_state_dict(resume_data["scaler"])

        best_psnr = float(resume_data.get("best_psnr", best_psnr))
        best_train_l1 = float(resume_data.get("best_train_l1", best_train_l1))
        start_epoch = int(resume_data.get("epoch", 0))
        global_step = int(resume_data.get("global_step", start_epoch * steps_per_epoch))
        print(f"Resumed from: {resume_path}")
        print(
            f"Resume state: epoch={start_epoch}, global_step={global_step}, "
            f"best_psnr={best_psnr:.2f}, best_train_l1={best_train_l1:.4f}"
        )
    elif args.init_ckpt:
        init_path = Path(args.init_ckpt)
        if not init_path.exists():
            raise FileNotFoundError(f"--init-ckpt not found: {init_path}")

        init_data = torch.load(init_path, map_location=device)
        if isinstance(init_data, dict) and "model" in init_data:
            model.load_state_dict(init_data["model"], strict=True)
            best_psnr = float(init_data.get("best_psnr", best_psnr))
        else:
            model.load_state_dict(init_data, strict=True)
        print(f"Initialized model from: {init_path}")
        print(f"Starting best_psnr: {best_psnr:.2f}")

    total_steps = (start_epoch + args.epochs) * steps_per_epoch

    start = time.time()

    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{start_epoch + args.epochs}", ncols=110)
        for deg, clean in pbar:
            deg = deg.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)

            lr_now = cosine_lr(global_step, total_steps, args.lr, lr_min=args.lr * 0.05)
            for g in optimizer.param_groups:
                g["lr"] = lr_now

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(args.amp and device.type == "cuda")):
                pred = model(deg)
                loss = l1(pred, clean)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr_now:.2e}")

        train_loss = running_loss / max(len(train_loader), 1)
        if val_loader is not None:
            val_loss, val_psnr = evaluate(model, val_loader, device, use_amp=args.amp)
            is_best = val_psnr > best_psnr
            if is_best:
                best_psnr = val_psnr
            best_mode = "val_psnr"
        else:
            val_loss = float("nan")
            val_psnr = float("nan")
            is_best = train_loss < best_train_l1
            if is_best:
                best_train_l1 = train_loss
            best_mode = "train_l1"

        ckpt = {
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "best_psnr": best_psnr,
            "best_train_l1": best_train_l1,
            "best_mode": best_mode,
            "args": vars(args),
        }
        torch.save(ckpt, save_dir / "last.pt")

        if is_best:
            torch.save(ckpt, save_dir / "best.pt")

        metrics = {
            "epoch": epoch,
            "train_l1": train_loss,
            "val_l1": val_loss,
            "val_psnr": val_psnr,
            "best_psnr": best_psnr,
            "best_train_l1": best_train_l1,
            "best_mode": best_mode,
            "is_best": is_best,
            "saved_best": is_best,
            "saved_last": True,
        }
        with open(save_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        elapsed = time.time() - start
        save_note = "SAVED best.pt" if is_best else "kept current best.pt"
        if val_loader is not None:
            print(
                f"Epoch {epoch:03d} | train_l1={train_loss:.4f} | val_l1={val_loss:.4f} | "
                f"val_psnr={val_psnr:.2f} | best_psnr={best_psnr:.2f} | {save_note} | time={elapsed/60:.1f}m"
            )
        else:
            print(
                f"Epoch {epoch:03d} | train_l1={train_loss:.4f} | "
                f"best_train_l1={best_train_l1:.4f} | {save_note} | time={elapsed/60:.1f}m"
            )

    print("Training finished.")


if __name__ == "__main__":
    main()
