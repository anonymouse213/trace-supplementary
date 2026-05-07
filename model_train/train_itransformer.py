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


def load_denorm_range(data_dir: str) -> dict:
    """
    """
    import os
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

NORMAL_RANGE = {
    "HR"       : (60.0,  100.0),
    "RespRate" : (12.0,  18.0),
    "SpO2"     : (96.0,  None),
}

def _norm_threshold(feat_name, raw_val):
    lo, hi = DENORM_RANGE[feat_name]
    return (raw_val - lo) / max(hi - lo, 1e-8)

ABNORMAL_THRESHOLDS = {
    "HR"       : (_norm_threshold("HR", 60.0),
                  _norm_threshold("HR", 100.0)),
    "RespRate" : (_norm_threshold("RespRate", 12.0),
                  _norm_threshold("RespRate", 18.0)),
    "SpO2"     : (None, _norm_threshold("SpO2", 96.0)),
}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="/path/to/processed_data")
    p.add_argument("--output_dir", default="/path/to/output")
    p.add_argument("--gpu",        default="0")
    p.add_argument("--phase",      default="predict",
                   choices=["train", "predict", "all"])
    p.add_argument("--ckpt",       default="/path/to/output")

    p.add_argument("--past_hours",   default=36, type=int)
    p.add_argument("--future_hours", default=12, type=int)
    p.add_argument("--slide_step",   default=6,  type=int,
                   help="")

    p.add_argument("--d_model",    default=512, type=int,
                   help="")
    p.add_argument("--n_heads",    default=8,   type=int)
    p.add_argument("--n_layers",   default=4,   type=int,
                   help="")
    p.add_argument("--d_ffn",      default=2048, type=int,
                   help="")
    p.add_argument("--dropout",    default=0.1, type=float)

    p.add_argument("--quantiles",  default=[0.1, 0.25, 0.5, 0.75, 0.9],
                   nargs="+", type=float)

    p.add_argument("--deterio_weight", default=0.1, type=float,
                   help="")
    p.add_argument("--trend_weight",   default=1.0, type=float,
                   help="")

    p.add_argument("--epochs",      default=100,  type=int)
    p.add_argument("--batch_size",  default=2048, type=int,
                   help="")
    p.add_argument("--lr",          default=1e-3, type=float,
                   help="")
    p.add_argument("--weight_decay",default=1e-4, type=float)
    p.add_argument("--patience",    default=20,   type=int)
    p.add_argument("--warmup_epochs", default=5,  type=int,
                   help="")
    p.add_argument("--num_workers",   default=8,  type=int)

    return p.parse_args()


class SlidingWindowForecastDataset(Dataset):
    """


      x: (past_hours, N_VARS)  HR/RespRate/SpO2 historical
      y: (future_hours, N_VARS) HR/RespRate/SpO2 future
    """

    def __init__(self, pkl_path: str, past_hours: int = 36,
                 future_hours: int = 12, slide_step: int = 6,
                 split: str = "train", augment: bool = False):
        with open(pkl_path, "rb") as f:
            raw = pickle.load(f)

        self.past    = past_hours
        self.future  = future_hours
        self.step    = slide_step
        self.augment = augment
        self.samples = []

        total = past_hours + future_hours

        for d in raw:
            ft = d.get("forecasting_targets")
            fm = d.get("forecast_masks")

            if ft is None:
                ts_full = d["historical_ts_numeric"]
                continue

            ft = ft.astype(np.float32)
            T  = ft.shape[0]

            if T < total:
                continue

            for start in range(0, T - total + 1, slide_step):
                x    = ft[start          : start + past_hours,  :]
                y    = ft[start + past_hours : start + total,   :]

                if fm is not None and start == 0:
                    mask = fm.astype(np.float32)
                else:
                    mask = (y > 0).astype(np.float32)

                self.samples.append({
                    "x"    : x,
                    "y"    : y,
                    "mask" : mask,
                    "label": int(d["label"]),
                    "sid"  : int(d["stay_id"]),
                })

        print(f"  {split}: {len(raw):,} stays → {len(self.samples):,} windows "
              f"(step={slide_step}h, x{len(self.samples)//max(len(raw),1):.1f}x)")

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
            torch.tensor(labels, dtype=torch.float32),
            list(sids))


class iTransformerForecaster(nn.Module):
    """

      2. DataEmbedding_inverted: (B, T, N) → (B, N, d_model)
      4. Projector: (B, N, d_model) → (B, N, T_out)

      - Deterioration head
    """

    def __init__(self, past_hours: int, future_hours: int, n_vars: int,
                 d_model: int, n_heads: int, n_layers: int, d_ffn: int,
                 dropout: float, quantiles: list):
        super().__init__()
        self.past      = past_hours
        self.future    = future_hours
        self.n_vars    = n_vars
        self.n_q       = len(quantiles)
        self.quantiles = quantiles


        self.enc_embedding = nn.Linear(past_hours, d_model)
        self.enc_dropout   = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model    = d_model,
            nhead      = n_heads,
            dim_feedforward = d_ffn,
            dropout    = dropout,
            activation = "gelu",
            batch_first = True,
            norm_first  = False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers = n_layers,
            norm       = nn.LayerNorm(d_model),
        )

        self.projector = nn.Linear(d_model, future_hours * self.n_q)

        self.deterio_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, n_vars),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        """
        x: (B, T_in, N_VARS) normalized [0,1]

        """
        B, T, N = x.shape

        x_t     = x.permute(0, 2, 1)
        enc     = self.enc_embedding(x_t)
        enc     = self.enc_dropout(enc)

        enc_out = self.encoder(enc)

        out = self.projector(enc_out)
        out = out.view(B, N, self.future, self.n_q)
        out = out.permute(0, 3, 2, 1)

        deterio = self.deterio_head(enc_out)
        deterio = deterio.mean(dim=1)
        deterio = torch.sigmoid(deterio)

        return out, deterio


def compute_deterio_target(
    y: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    y: (B, T_out, N_VARS) normalized future vital sign

    RespRate: > 0.36 (tachypnea) or < 0.24 (bradypnea)
    SpO2:     < 0.90 (hypoxia)
    """
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

        any_abnormal = (abnormal & valid).any(dim=1).float()
        result[:, vi] = any_abnormal

    return result


def quantile_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    quantiles: list,
) -> torch.Tensor:
    """
    pred:   (B, n_q, T_out, N_VARS)
    target: (B, T_out, N_VARS)

    """
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


def trend_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """

    pred:   (B, n_q, T_out, N_VARS)
    target: (B, T_out, N_VARS)
    mask:   (B, T_out, N_VARS)
    """
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
    all_preds_median, all_targets, all_masks = [], [], []
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

            loss_t = trend_loss(preds, y, mask)                      if getattr(args, 'trend_weight', 0) > 0                      else torch.tensor(0.0, device=device)

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
                         if 0.5 in args.quantiles else len(args.quantiles)//2
            pred_median = preds[:, median_idx, :, :].detach().cpu()
            y_cpu       = y.detach().cpu()
            mask_cpu    = mask.detach().cpu()

            valid_all = mask_cpu > 0.5
            if valid_all.any():
                all_preds_median.append(pred_median[valid_all].numpy())
                all_targets.append(y_cpu[valid_all].numpy())

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
    if all_preds_median:
        p_arr = np.concatenate(all_preds_median)
        t_arr = np.concatenate(all_targets)
        metrics["mae_norm"] = float(mean_absolute_error(t_arr, p_arr))

    if all_preds_per_var and all_targets_per_var:
        for vi, fname in enumerate(FORECAST_NAMES):
            lo, hi = DENORM_RANGE[fname]
            p_v = np.concatenate(all_preds_per_var[vi])
            t_v = np.concatenate(all_targets_per_var[vi])
            p_raw = p_v * (hi - lo) + lo
            t_raw = t_v * (hi - lo) + lo
            metrics[f"mae_{fname}"] = float(mean_absolute_error(t_raw, p_raw))

    return metrics


def run_predict(args, model, device, split_name: str, data_dir: str):
    """

      {stay_id: {
          "preds": (n_q, 12, 3),
          "x"    : (36, 3),
          "y"    : (12, 3),
          "mask" : (12, 3),
          "label": int,
      }}
    """
    print(f"\n  Predicting {split_name}...")
    ds = SlidingWindowForecastDataset(
        os.path.join(data_dir, f"{split_name}.pkl"),
        past_hours=args.past_hours, future_hours=args.future_hours,
        slide_step=999, split=split_name)
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

    out_path = os.path.join(args.output_dir,
                            f"{split_name}_predictions.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(results, f, protocol=4)
    print(f"  → {len(results):,} stays → {out_path}")
    return results


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}"
                          if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Quantiles: {args.quantiles}")

    global DENORM_RANGE, ABNORMAL_THRESHOLDS
    DENORM_RANGE = load_denorm_range(args.data_dir)
    ABNORMAL_THRESHOLDS = {
        "HR"       : (_norm_threshold("HR", 60.0),
                      _norm_threshold("HR", 100.0)),
        "RespRate" : (_norm_threshold("RespRate", 12.0),
                      _norm_threshold("RespRate", 18.0)),
        "SpO2"     : (None, _norm_threshold("SpO2", 96.0)),
    }


    train_ds = SlidingWindowForecastDataset(
        os.path.join(args.data_dir, "train.pkl"),
        args.past_hours, args.future_hours,
        slide_step=999, split="train",
        augment=True)
    val_ds = SlidingWindowForecastDataset(
        os.path.join(args.data_dir, "val.pkl"),
        args.past_hours, args.future_hours,
        slide_step=999, split="val",
        augment=False)

    train_ld = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True,  collate_fn=collate_fn,
                          num_workers=args.num_workers,
                          pin_memory=True, persistent_workers=True)
    val_ld   = DataLoader(val_ds,   batch_size=2048,
                          shuffle=False, collate_fn=collate_fn,
                          num_workers=args.num_workers,
                          pin_memory=True, persistent_workers=True)

    model = iTransformerForecaster(
        past_hours  = args.past_hours,
        future_hours= args.future_hours,
        n_vars      = N_VARS,
        d_model     = args.d_model,
        n_heads     = args.n_heads,
        n_layers    = args.n_layers,
        d_ffn       = args.d_ffn,
        dropout     = args.dropout,
        quantiles   = args.quantiles,
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

        wu = getattr(args, 'warmup_epochs', 5)
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
        print("Training")
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

            history.append({"epoch": ep, **{f"tr_{k}": v for k,v in tr.items()},
                             **{f"va_{k}": v for k,v in va.items()}})

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