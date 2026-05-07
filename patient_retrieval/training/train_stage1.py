import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.dataset import load_datasets
from data.augmentation import TimeSeriesAugmentor, AugmentedDatasetWrapper
from models.encoder import PatientEncoder
from models.losses import HierarchicalContrastiveLoss

from tqdm import tqdm

def get_args():
    p = argparse.ArgumentParser(description="")
    p.add_argument("--data_dir",    type=str,   default="/path/to/output")
    p.add_argument("--out_dir",     type=str,   default="./checkpoints/stage1")
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--batch_size",  type=int,   default=256)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--hidden_dim",  type=int,   default=256)
    p.add_argument("--n_layers",    type=int,   default=6)
    p.add_argument("--embed_dim",   type=int,   default=256)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--patience",    type=int,   default=15)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--device",      type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n = 0

    for batch in tqdm(loader):
        v1 = batch["view1"].to(device)
        v2 = batch["view2"].to(device)

        z1 = model(v1, return_seq=True)
        z2 = model(v2, return_seq=True)

        loss = criterion(z1, z2)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * v1.size(0)
        n += v1.size(0)

    return total_loss / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n = 0

    for batch in tqdm(loader):
        v1 = batch["view1"].to(device)
        v2 = batch["view2"].to(device)
        z1 = model(v1, return_seq=True)
        z2 = model(v2, return_seq=True)
        loss = criterion(z1, z2)
        total_loss += loss.item() * v1.size(0)
        n += v1.size(0)

    return total_loss / n


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f"\n{'='*60}")
    print(f"  Stage 1: Self-supervised Pretraining")
    print(f"  Data: {args.data_dir}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    train_ds, val_ds, _ = load_datasets(args.data_dir, mode="stage1")

    augmentor = TimeSeriesAugmentor(
        crop_ratio=0.5, mask_ratio=0.15,
        scale_range=(0.8, 1.2), jitter_std=0.05,
    )
    train_aug = AugmentedDatasetWrapper(train_ds, augmentor)
    val_aug   = AugmentedDatasetWrapper(val_ds,   augmentor)

    train_loader = DataLoader(train_aug, batch_size=args.batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_aug,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    model = PatientEncoder(
        ts_input_dim  = 183,
        ts_hidden_dim = args.hidden_dim,
        ts_n_layers   = args.n_layers,
        embed_dim     = args.embed_dim,
        use_static    = False,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")

    criterion = HierarchicalContrastiveLoss(temperature=args.temperature)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    patience_cnt  = 0
    history = []

    for epoch in tqdm(range(1, args.epochs + 1)):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss   = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"train={train_loss:.4f}  val={val_loss:.4f}  lr={lr_now:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt  = 0
            ckpt_path = os.path.join(args.out_dir, "best_stage1.pt")
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "val_loss"   : val_loss,
                "args"       : vars(args),
            }, ckpt_path)
            print(f"    → Saved best checkpoint (val_loss={val_loss:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"\n  Early stopping at epoch {epoch}.")
                break

    print(f"\n  Best val loss: {best_val_loss:.4f}")
    print(f"  Checkpoint: {args.out_dir}/best_stage1.pt")

    torch.save({
        "epoch"      : epoch,
        "model_state": model.state_dict(),
        "val_loss"   : val_loss,
        "args"       : vars(args),
    }, os.path.join(args.out_dir, "last_stage1.pt"))


if __name__ == "__main__":
    main()