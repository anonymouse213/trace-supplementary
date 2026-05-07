import os, json, pickle, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm


FORECAST_IDX   = [0, 3, 4]
FORECAST_NAMES = ["HR", "RespRate", "SpO2"]
N_VARS         = len(FORECAST_IDX)

DENORM_RANGE_FALLBACK = {
    "HR"       : (58.0,  119.0),
    "RespRate" : (11.0,  30.0),
    "SpO2"     : (92.0,  100.0),
}
DENORM_RANGE = DENORM_RANGE_FALLBACK.copy()

NORMAL_RANGE = {
    "HR"       : (60.0,  100.0),
    "RespRate" : (12.0,  18.0),
    "SpO2"     : (96.0,  None),
}


def load_denorm_range(data_dir: str) -> dict:
    meta_path = os.path.join(data_dir, "metadata.pkl")
    if not os.path.exists(meta_path):
        return DENORM_RANGE_FALLBACK.copy()
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    feat_min = meta.get("feat_min_map", {})
    feat_max = meta.get("feat_max_map", {})
    result = {}
    for fname in FORECAST_NAMES:
        lo = feat_min.get(fname, None)
        hi = feat_max.get(fname, None)
        if lo is not None and hi is not None:
            result[fname] = (float(lo), float(hi))
            print(f"  DENORM {fname}: [{lo:.2f}, {hi:.2f}]")
        else:
            result[fname] = DENORM_RANGE_FALLBACK[fname]
            print(f"  DENORM {fname}: fallback {DENORM_RANGE_FALLBACK[fname]}")
    return result


def _norm_threshold(feat_name, raw_val):
    lo, hi = DENORM_RANGE[feat_name]
    return (raw_val - lo) / max(hi - lo, 1e-8)

ABNORMAL_THRESHOLDS = {
    "HR"       : (_norm_threshold("HR", 60.0),   _norm_threshold("HR", 100.0)),
    "RespRate" : (_norm_threshold("RespRate", 12.0), _norm_threshold("RespRate", 18.0)),
    "SpO2"     : (None, _norm_threshold("SpO2", 96.0)),
}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="/path/to/processed_data")
    p.add_argument("--output_dir", default="/path/to/output")
    p.add_argument("--gpu",        default="0")
    p.add_argument("--phase",      default="predict", choices=["train", "predict", "all"])
    p.add_argument("--ckpt",       default="/path/to/output")

    p.add_argument("--past_hours",   default=36, type=int)
    p.add_argument("--future_hours", default=12, type=int)

    p.add_argument("--kernel_size",  default=25, type=int,
                   help="")
    p.add_argument("--individual",   action="store_true", default=True,
                   help="")

    p.add_argument("--quantiles",  default=[0.1, 0.25, 0.5, 0.75, 0.9],
                   nargs="+", type=float)

    p.add_argument("--deterio_weight", default=0.1, type=float)
    p.add_argument("--trend_weight",   default=1.0, type=float)

    p.add_argument("--epochs",       default=100,  type=int)
    p.add_argument("--batch_size",   default=2048, type=int)
    p.add_argument("--lr",           default=1e-3, type=float)
    p.add_argument("--weight_decay", default=1e-4, type=float)
    p.add_argument("--patience",     default=20,   type=int)
    p.add_argument("--warmup_epochs",default=5,    type=int)
    p.add_argument("--num_workers",  default=8,    type=int)

    return p.parse_args()


class ForecastDataset(Dataset):
    def __init__(self, pkl_path: str, past_hours: int = 36,
                 future_hours: int = 12, split: str = "train",
                 augment: bool = False):
        with open(pkl_path, "rb") as f:
            raw = pickle.load(f)

        self.past    = past_hours
        self.future  = future_hours
        self.augment = augment
        self.samples = []

        total = past_hours + future_hours

        for d in raw:
            ft = d.get("forecasting_targets")
            fm = d.get("forecast_masks")
            if ft is None:
                continue
            ft = ft.astype(np.float32)
            if ft.shape[0] < total:
                continue

            x    = ft[:past_hours, :]
            y    = ft[past_hours:total, :]
            mask = fm.astype(np.float32) if fm is not None else (y > 0).astype(np.float32)

            self.samples.append({
                "x"    : x,
                "y"    : y,
                "mask" : mask,
                "label": int(d["label"]),
                "sid"  : int(d["stay_id"]),
            })

        print(f"  {split}: {len(raw):,} stays → {len(self.samples):,} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        x = s["x"].copy()
        y = s["y"].copy()

        if self.augment:
            x += np.random.normal(0, 0.01, x.shape)
            x = np.clip(x, 0.0, 1.0)
            shift = np.random.randint(0, 4)
            if shift > 0 and x.shape[0] > shift:
                x = np.concatenate([x[shift:], x[-shift:]], axis=0)

        return (
            torch.tensor(x,         dtype=torch.float32),
            torch.tensor(y,         dtype=torch.float32),
            torch.tensor(s["mask"], dtype=torch.float32),
            s["label"],
            s["sid"],
        )


def collate_fn(batch):
    xs, ys, masks, labels, sids = zip(*batch)
    return (torch.stack(xs), torch.stack(ys), torch.stack(masks),
            torch.tensor(labels, dtype=torch.float32), list(sids))


class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end   = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size)

    def forward(self, x):
        trend    = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class DLinearForecaster(nn.Module):
    """
    DLinear + Quantile output head + Deterioration head

      preds:   (B, n_q, T_out, N_VARS)
      deterio: (B, N_VARS) sigmoid


                    → seasonal_q(B,N,T_out*n_q) + trend_q(B,N,T_out*n_q)

    Deterioration head:
    """

    def __init__(self, past_hours: int, future_hours: int, n_vars: int,
                 kernel_size: int = 25, individual: bool = False,
                 quantiles: list = None):
        super().__init__()
        self.past       = past_hours
        self.future     = future_hours
        self.n_vars     = n_vars
        self.individual = individual
        self.quantiles  = quantiles or [0.1, 0.25, 0.5, 0.75, 0.9]
        self.n_q        = len(self.quantiles)

        self.decomp = series_decomp(kernel_size)

        out_dim = future_hours * self.n_q

        if individual:
            self.Linear_Seasonal = nn.ModuleList(
                [nn.Linear(past_hours, out_dim) for _ in range(n_vars)])
            self.Linear_Trend = nn.ModuleList(
                [nn.Linear(past_hours, out_dim) for _ in range(n_vars)])
            for i in range(n_vars):
                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / past_hours) * torch.ones(out_dim, past_hours))
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / past_hours) * torch.ones(out_dim, past_hours))
        else:
            self.Linear_Seasonal = nn.Linear(past_hours, out_dim)
            self.Linear_Trend    = nn.Linear(past_hours, out_dim)
            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / past_hours) * torch.ones(out_dim, past_hours))
            self.Linear_Trend.weight = nn.Parameter(
                (1 / past_hours) * torch.ones(out_dim, past_hours))

        w = 6
        self.deterio_head = nn.Sequential(
            nn.Linear(n_vars * w, 64),
            nn.GELU(),
            nn.Linear(64, n_vars),
        )
        self._w = w

    def forward(self, x: torch.Tensor):
        """
        x: (B, T_in, N_VARS)
        return:
          preds:   (B, n_q, T_out, N_VARS)
          deterio: (B, N_VARS)
        """
        B, T, N = x.shape
        seasonal, trend = self.decomp(x)

        s = seasonal.permute(0, 2, 1)
        t = trend.permute(0, 2, 1)

        if self.individual:
            s_out = torch.stack([self.Linear_Seasonal[i](s[:, i, :])
                                 for i in range(N)], dim=1)
            t_out = torch.stack([self.Linear_Trend[i](t[:, i, :])
                                 for i in range(N)], dim=1)
        else:
            s_out = self.Linear_Seasonal(s)
            t_out = self.Linear_Trend(t)

        out = s_out + t_out

        out = out.view(B, N, self.future, self.n_q)
        out = out.permute(0, 3, 2, 1)

        trend_feat = trend[:, -self._w:, :].reshape(B, -1)
        deterio = torch.sigmoid(self.deterio_head(trend_feat))

        return out, deterio


def compute_deterio_target(y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    B, T, V = y.shape
    result = torch.zeros(B, V, device=y.device)
    thresholds = [
        (0, 0.30, 0.50),
        (1, 0.24, 0.36),
        (2, None, 0.90),
    ]
    for vi, lo, hi in thresholds:
        v_vals = y[:, :, vi]
        v_mask = mask[:, :, vi]
        valid  = v_mask > 0.5
        if lo is not None and hi is not None:
            abnormal = (v_vals < lo) | (v_vals > hi)
        elif lo is None:
            abnormal = v_vals < hi
        else:
            abnormal = (v_vals < lo) | (v_vals > hi)
        result[:, vi] = (abnormal & valid).any(dim=1).float()
    return result


def quantile_loss(pred: torch.Tensor, target: torch.Tensor,
                  mask: torch.Tensor, quantiles: list) -> torch.Tensor:
    total_loss = torch.tensor(0.0, device=pred.device)
    n_valid    = 0
    for qi, q in enumerate(quantiles):
        p   = pred[:, qi, :, :]
        err = target - p
        loss_q = torch.where(err >= 0, q * err, (q - 1) * err)
        valid_mask = mask > 0.5
        if valid_mask.any():
            total_loss = total_loss + loss_q[valid_mask].mean()
            n_valid   += 1
    return total_loss / max(n_valid, 1)


def trend_loss(pred: torch.Tensor, target: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    n_q   = pred.shape[1]
    med   = n_q // 2
    p_med = pred[:, med, :, :]

    mask_1 = (mask[:, :-1, :] > 0.5) & (mask[:, 1:, :] > 0.5)
    if not mask_1.any():
        return torch.tensor(0.0, device=pred.device)

    td1 = target[:, 1:, :] - target[:, :-1, :]
    pd1 = p_med[:, 1:, :]  - p_med[:, :-1, :]
    return ((pd1 - td1) ** 2)[mask_1].mean()


def run_epoch(model, loader, optimizer, args, device, train: bool):
    model.train() if train else model.eval()
    total_loss = total_ql = total_dl = total_tl = 0.0
    all_preds_per_var  = {vi: [] for vi in range(N_VARS)}
    all_targets_per_var = {vi: [] for vi in range(N_VARS)}
    n_batch = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y, mask, labels, _ in loader:
            x    = x.to(device)
            y    = y.to(device)
            mask = mask.to(device)

            preds, deterio = model(x)

            loss_q = quantile_loss(preds, y, mask, args.quantiles)
            loss_t = trend_loss(preds, y, mask) \
                     if getattr(args, 'trend_weight', 0) > 0 \
                     else torch.tensor(0.0, device=device)

            det_tgt = compute_deterio_target(y, mask)
            loss_d  = F.binary_cross_entropy(deterio, det_tgt)

            loss = (loss_q
                    + getattr(args, 'trend_weight', 1.0) * loss_t
                    + args.deterio_weight * loss_d)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            total_ql   += loss_q.item()
            total_dl   += loss_d.item()
            total_tl   += loss_t.item()
            n_batch    += 1

            median_idx = args.quantiles.index(0.5) \
                         if 0.5 in args.quantiles else len(args.quantiles) // 2
            pred_median = preds[:, median_idx, :, :].detach().cpu()
            y_cpu    = y.detach().cpu()
            mask_cpu = mask.detach().cpu()

            for vi in range(N_VARS):
                valid_vi = mask_cpu[:, :, vi] > 0.5
                if valid_vi.any():
                    all_preds_per_var[vi].append(
                        pred_median[:, :, vi][valid_vi].numpy())
                    all_targets_per_var[vi].append(
                        y_cpu[:, :, vi][valid_vi].numpy())

    n = max(n_batch, 1)
    metrics = {
        "loss"  : total_loss / n,
        "q_loss": total_ql   / n,
        "d_loss": total_dl   / n,
        "t_loss": total_tl   / n,
    }

    for vi, fname in enumerate(FORECAST_NAMES):
        lo, hi = DENORM_RANGE[fname]
        if all_preds_per_var[vi]:
            p_v = np.concatenate(all_preds_per_var[vi])
            t_v = np.concatenate(all_targets_per_var[vi])
            p_raw = p_v * (hi - lo) + lo
            t_raw = t_v * (hi - lo) + lo
            metrics[f"mae_{fname}"] = float(mean_absolute_error(t_raw, p_raw))
    return metrics


def run_predict(args, model, device, split_name: str, data_dir: str):
    print(f"\n  Predicting {split_name}...")
    ds = ForecastDataset(
        os.path.join(data_dir, f"{split_name}.pkl"),
        past_hours=args.past_hours, future_hours=args.future_hours,
        split=split_name, augment=False)
    loader = DataLoader(ds, batch_size=256, shuffle=False,
                        collate_fn=collate_fn, num_workers=2)

    model.eval()
    results = {}
    with torch.no_grad():
        for x, y, mask, labels, sids in tqdm(loader, desc=f"  {split_name}"):
            x    = x.to(device)
            preds, _ = model(x)
            for i, sid in enumerate(sids):
                results[sid] = {
                    "preds": preds[i].cpu().numpy(),
                    "x"    : x[i].cpu().numpy(),
                    "y"    : y[i].numpy(),
                    "mask" : mask[i].numpy(),
                    "label": int(labels[i].item()),
                }

    out_path = os.path.join(args.output_dir, f"{split_name}_predictions.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(results, f, protocol=4)
    print(f"  → {len(results):,} stays → {out_path}")
    return results


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Quantiles: {args.quantiles}")
    print(f"Model: DLinear + Quantile + Deterio (individual={args.individual}, kernel={args.kernel_size})")

    global DENORM_RANGE, ABNORMAL_THRESHOLDS
    DENORM_RANGE = load_denorm_range(args.data_dir)
    ABNORMAL_THRESHOLDS = {
        "HR"       : (_norm_threshold("HR", 60.0),      _norm_threshold("HR", 100.0)),
        "RespRate" : (_norm_threshold("RespRate", 12.0), _norm_threshold("RespRate", 18.0)),
        "SpO2"     : (None, _norm_threshold("SpO2", 96.0)),
    }

    train_ds = ForecastDataset(
        os.path.join(args.data_dir, "train.pkl"),
        args.past_hours, args.future_hours, split="train", augment=True)
    val_ds = ForecastDataset(
        os.path.join(args.data_dir, "val.pkl"),
        args.past_hours, args.future_hours, split="val", augment=False)

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate_fn, num_workers=args.num_workers,
                          pin_memory=True, persistent_workers=True)
    val_ld   = DataLoader(val_ds,   batch_size=2048, shuffle=False,
                          collate_fn=collate_fn, num_workers=args.num_workers,
                          pin_memory=True, persistent_workers=True)

    model = DLinearForecaster(
        past_hours   = args.past_hours,
        future_hours = args.future_hours,
        n_vars       = N_VARS,
        kernel_size  = args.kernel_size,
        individual   = args.individual,
        quantiles    = args.quantiles,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    if args.ckpt and os.path.exists(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded: {args.ckpt}")

    if args.phase in ("train", "all"):
        optimizer = optim.AdamW(model.parameters(),
                                lr=args.lr, weight_decay=args.weight_decay)

        wu = args.warmup_epochs
        def lr_lambda(ep):
            if ep < wu:
                return float(ep + 1) / float(wu)
            prog = (ep - wu) / max(args.epochs - wu, 1)
            return 0.05 + 0.95 * 0.5 * (1.0 + np.cos(np.pi * prog))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        best_val  = float("inf")
        patience  = 0
        ckpt_path = os.path.join(args.output_dir, "best.pt")
        history   = []

        print("\n" + "="*60)
        print("Training DLinear + Quantile + Deterioration")
        print("="*60)

        for ep in tqdm(range(1, args.epochs + 1)):
            tr = run_epoch(model, train_ld, optimizer, args, device, train=True)
            va = run_epoch(model, val_ld,   optimizer, args, device, train=False)
            scheduler.step()

            lr_cur = optimizer.param_groups[0]["lr"]
            mae_str = "  ".join(
                f"{n}={va.get(f'mae_{n}', float('nan')):.3f}"
                for n in FORECAST_NAMES)
            print(f"  Ep {ep:3d} | "
                  f"tr={tr['loss']:.4f}"
                  f"(q={tr['q_loss']:.3f} t={tr['t_loss']:.3f} d={tr['d_loss']:.3f}) | "
                  f"val={va['loss']:.4f} | "
                  f"MAE[{mae_str}] | "
                  f"lr={lr_cur:.2e}", end="")

            history.append({"epoch": ep,
                             **{f"tr_{k}": v for k, v in tr.items()},
                             **{f"va_{k}": v for k, v in va.items()}})

            if va["loss"] < best_val:
                best_val = va["loss"]
                patience = 0
                torch.save({"model_state_dict": model.state_dict(),
                            "epoch": ep, "val_loss": va["loss"],
                            "args": vars(args)}, ckpt_path)
                print(" ✓", end="")
            else:
                patience += 1
            print()

            if patience >= args.patience:
                print(f"  Early stopping @ ep {ep}")
                break

        with open(os.path.join(args.output_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
        print(f"\nBest → {ckpt_path}  (val_loss={best_val:.5f})")

    if args.phase in ("predict", "all"):
        ckpt_path = args.ckpt or os.path.join(args.output_dir, "best.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"\nLoaded best ckpt: {ckpt_path}")

        for split in ["train", "val", "test"]:
            run_predict(args, model, device, split, args.data_dir)


if __name__ == "__main__":
    main()