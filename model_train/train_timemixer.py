import os, sys, json, pickle, argparse, types
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

from models import TimeMixer


class TimeMixerQuantile(nn.Module):
    """
    TimeMixer backbone + Quantile head + Deterioration head.


      backbone(x) → (B, pred_len, N_VARS)
      quantile_proj: Linear(N_VARS → N_VARS × n_q) per timestep
      → (B, pred_len, N_VARS × n_q) → reshape → (B, n_q, pred_len, N_VARS)

    """
    def __init__(self, backbone, n_q: int, n_vars: int, future_hours: int):
        super().__init__()
        self.backbone     = backbone
        self.n_q          = n_q
        self.n_vars       = n_vars
        self.future_hours = future_hours

        self.quantile_proj = nn.Linear(n_vars, n_vars * n_q)
        self.deterio_head  = nn.Sequential(
            nn.Linear(n_vars, 32), nn.GELU(), nn.Linear(32, n_vars))

        nn.init.xavier_uniform_(self.quantile_proj.weight)
        nn.init.zeros_(self.quantile_proj.bias)

    def forward(self, x):
        """
        x: (B, T_in, N_VARS)
        """
        B = x.shape[0]
        base_pred = self.backbone(x, None, None, None)

        q_out = self.quantile_proj(base_pred)
        q_out = q_out.view(B, self.future_hours, self.n_q, self.n_vars)
        q_out = q_out.permute(0, 2, 1, 3)

        det = self.deterio_head(x[:, -1, :])
        det = torch.sigmoid(det)

        return q_out, det


FORECAST_NAMES = ["HR", "RespRate", "SpO2"]
N_VARS         = 3
QUANTILES      = [0.1, 0.25, 0.5, 0.75, 0.9]
N_Q            = len(QUANTILES)
MED_IDX        = N_Q // 2


def build_timemixer_args(past_hours, future_hours, n_vars,
                          d_model, d_ff, e_layers,
                          down_sampling_layers, down_sampling_window,
                          dropout, moving_avg=25):
    """
    """
    args = types.SimpleNamespace(
        task_name          = 'long_term_forecast',
        seq_len            = past_hours,
        label_len          = 0,
        pred_len           = future_hours,
        enc_in             = n_vars,
        dec_in             = n_vars,
        c_out              = n_vars,
        d_model            = d_model,
        d_ff               = d_ff,
        e_layers           = e_layers,
        n_heads            = 8,
        d_layers           = 1,
        factor             = 3,
        dropout            = dropout,
        embed              = 'timeF',
        freq               = 'h',
        output_attention   = False,
        moving_avg         = moving_avg,
        decomp_method      = 'moving_avg',
        down_sampling_layers = down_sampling_layers,
        down_sampling_window = down_sampling_window,
        down_sampling_method = 'avg',
        channel_independence = 1,
        use_norm           = 1,
        use_future_temporal_feature = 0,
        top_k              = 5,
        num_class          = 2,
    )
    return args


class SlidingWindowDataset(Dataset):
    def __init__(self, pkl_path, past_hours=36, future_hours=12,
                 slide_step=6, split="train", augment=False):
        with open(pkl_path, "rb") as f:
            raw = pickle.load(f)
        self.samples = []
        total = past_hours + future_hours
        for d in raw:
            ft = d.get("forecasting_targets")
            if ft is None: continue
            ft = ft.astype(np.float32)
            if ft.shape[0] < total: continue
            fm = d.get("forecast_masks")
            for start in range(0, ft.shape[0] - total + 1, slide_step):
                x    = ft[start            : start + past_hours, :]
                y    = ft[start + past_hours : start + total,    :]
                mask = fm.astype(np.float32) if (fm is not None and start == 0) \
                       else (y > 0).astype(np.float32)
                self.samples.append({
                    "x": x, "y": y, "mask": mask,
                    "label": int(d["label"]), "sid": int(d["stay_id"]),
                })
        print(f"  {split}: {len(raw):,} stays → {len(self.samples):,} windows")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (torch.tensor(s["x"].copy(), dtype=torch.float32),
                torch.tensor(s["y"],        dtype=torch.float32),
                torch.tensor(s["mask"],     dtype=torch.float32),
                s["label"], s["sid"])


def collate_fn(batch):
    xs, ys, masks, labels, sids = zip(*batch)
    return (torch.stack(xs), torch.stack(ys), torch.stack(masks),
            torch.tensor(labels, dtype=torch.float32), list(sids))


def timemixer_forward(model, x):
    """
    x: (B, T_in, N_VARS)
    """
    return model(x, None, None, None)


def masked_mse(pred, target, mask):
    valid = mask > 0.5
    if not valid.any(): return torch.tensor(0.0, device=pred.device)
    return ((pred - target) ** 2)[valid].mean()


def quantile_loss(pred, target, mask, quantiles=QUANTILES):
    total = torch.tensor(0.0, device=pred.device)
    valid = mask > 0.5
    if not valid.any(): return total
    for qi, q in enumerate(quantiles):
        p   = pred[:, qi, :, :]
        err = target - p
        lq  = torch.where(err >= 0, q * err, (q - 1) * err)
        total = total + lq[valid].mean()
    return total / len(quantiles)


def trend_loss(pred, target, mask):
    p_med  = pred[:, MED_IDX, :, :]
    mask_1 = (mask[:, :-1, :] > 0.5) & (mask[:, 1:, :] > 0.5)
    if not mask_1.any(): return torch.tensor(0.0, device=pred.device)
    td1 = target[:, 1:, :] - target[:, :-1, :]
    pd1 = p_med[:, 1:, :]  - p_med[:, :-1, :]
    return ((pd1 - td1) ** 2)[mask_1].mean()


def load_denorm(data_dir):
    meta = os.path.join(data_dir, "metadata.pkl")
    fb   = {"HR": (58., 119.), "RespRate": (11., 30.), "SpO2": (92., 100.)}
    if not os.path.exists(meta): return fb
    with open(meta, "rb") as f: m = pickle.load(f)
    mn = m.get("feat_min_map", {}); mx = m.get("feat_max_map", {})
    return {k: (float(mn.get(k, fb[k][0])), float(mx.get(k, fb[k][1])))
            for k in FORECAST_NAMES}


def run_epoch(model, loader, optimizer, args, device, train, denorm):
    model.train() if train else model.eval()
    total, n_batch = 0.0, 0
    all_pv = {vi: [] for vi in range(N_VARS)}
    all_tv = {vi: [] for vi in range(N_VARS)}

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y, mask, _, _ in loader:
            x = x.to(device); y = y.to(device); mask = mask.to(device)
            preds, det = model(x)
            lq = quantile_loss(preds, y, mask)
            lt = trend_loss(preds, y, mask)
            det_tgt = ((y > 0.8) | (y < 0.1)).any(dim=1).float()
            ld = F.binary_cross_entropy(det, det_tgt)
            loss = lq + args.trend_weight * lt + args.deterio_weight * ld
            if train:
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += loss.item(); n_batch += 1
            p_med = preds[:, MED_IDX, :, :]
            valid = mask > 0.5
            for vi in range(N_VARS):
                vm = valid[:, :, vi]
                if vm.any():
                    all_pv[vi].append(p_med[:, :, vi][vm].detach().cpu().numpy())
                    all_tv[vi].append(y[:, :, vi][vm].detach().cpu().numpy())

    metrics = {"loss": total / max(n_batch, 1)}
    for vi, fn in enumerate(FORECAST_NAMES):
        if all_pv[vi]:
            lo, hi = denorm[fn]
            pv = np.concatenate(all_pv[vi]) * (hi - lo) + lo
            tv = np.concatenate(all_tv[vi]) * (hi - lo) + lo
            metrics[f"mae_{fn}"] = float(mean_absolute_error(tv, pv))
    return metrics


def run_predict(model, args, device, split_name):
    print(f"\n  Predicting {split_name}...")
    ds = SlidingWindowDataset(
        os.path.join(args.data_dir, f"{split_name}.pkl"),
        args.past_hours, args.future_hours, 9999, split_name)
    loader = DataLoader(ds, 512, shuffle=False,
                        collate_fn=collate_fn, num_workers=4)
    model.eval()
    results = {}
    with torch.no_grad():
        for x, y, mask, labels, sids in tqdm(loader, desc=f"  {split_name}"):
            preds, _ = model(x.to(device))
            for i, sid in enumerate(sids):
                results[sid] = {
                    "preds": preds[i].cpu().numpy(),
                    "x"    : x[i].cpu().numpy(),
                    "y"    : y[i].numpy(),
                    "mask" : mask[i].numpy(),
                    "label": int(labels[i].item()),
                }
    out = os.path.join(args.output_dir, f"{split_name}_predictions.pkl")
    with open(out, "wb") as f: pickle.dump(results, f, protocol=4)
    print(f"  → {len(results):,} stays → {out}")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="/path/to/processed_data")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--gpu",   default="0")
    p.add_argument("--phase", default="predict", choices=["train","predict","all"])
    p.add_argument("--ckpt",  default=None)

    p.add_argument("--past_hours",   default=36,  type=int)
    p.add_argument("--future_hours", default=12,  type=int)
    p.add_argument("--slide_step",   default=6,   type=int)

    p.add_argument("--d_model",  default=128,  type=int,
        help="")
    p.add_argument("--d_ff",     default=256,  type=int)
    p.add_argument("--e_layers", default=3,   type=int)
    p.add_argument("--down_sampling_layers", default=2, type=int)
    p.add_argument("--down_sampling_window", default=2, type=int)
    p.add_argument("--dropout",  default=0.1, type=float)
    p.add_argument("--moving_avg", default=25, type=int)


    p.add_argument("--epochs",        default=100,  type=int)
    p.add_argument("--batch_size",    default=2048, type=int)
    p.add_argument("--lr",            default=1e-3, type=float,
        help="")
    p.add_argument("--weight_decay",  default=1e-4, type=float)
    p.add_argument("--patience",      default=20,   type=int)
    p.add_argument("--warmup_epochs", default=5,    type=int)
    p.add_argument("--num_workers",   default=8,    type=int)
    p.add_argument("--trend_weight",   default=1.0,  type=float)
    p.add_argument("--deterio_weight", default=0.1,  type=float)
    return p.parse_args()


def main():
    args   = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    denorm = load_denorm(args.data_dir)

    tm_args = build_timemixer_args(
        past_hours           = args.past_hours,
        future_hours         = args.future_hours,
        n_vars               = N_VARS,
        d_model              = args.d_model,
        d_ff                 = args.d_ff,
        e_layers             = args.e_layers,
        down_sampling_layers = args.down_sampling_layers,
        down_sampling_window = args.down_sampling_window,
        dropout              = args.dropout,
        moving_avg           = args.moving_avg,
    )

    print(f"Device: {device}")
    print(f"TimeMixer (ours): d={args.d_model} dsl={args.down_sampling_layers} "
          f"dsw={args.down_sampling_window} [Quantile 5 paths]")

    backbone = TimeMixer.Model(tm_args).float().to(device)
    model = TimeMixerQuantile(
        backbone=backbone, n_q=N_Q,
        n_vars=N_VARS, future_hours=args.future_hours,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.ckpt and os.path.exists(args.ckpt):
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        print(f"  Loaded: {args.ckpt}")

    if args.phase in ("train", "all"):
        train_ds = SlidingWindowDataset(
            os.path.join(args.data_dir, "train.pkl"),
            args.past_hours, args.future_hours, args.slide_step, "train")
        val_ds   = SlidingWindowDataset(
            os.path.join(args.data_dir, "val.pkl"),
            args.past_hours, args.future_hours, 9999, "val")
        train_ld = DataLoader(train_ds, args.batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)
        val_ld   = DataLoader(val_ds, 2048, shuffle=False,
            collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

        best_loss, pat_cnt = float("inf"), 0
        history = []
        ckpt = os.path.join(args.output_dir, "best_model.pt")

        print(f"\n{'='*55}\nTraining [TimeMixer ours: Quantile, 5 paths]\n{'='*55}")

        for ep in range(1, args.epochs + 1):
            if ep <= args.warmup_epochs:
                for pg in optimizer.param_groups:
                    pg["lr"] = args.lr * ep / args.warmup_epochs

            tr = run_epoch(model, train_ld, optimizer, args, device, True, denorm)
            va = run_epoch(model, val_ld, None, args, device, False, denorm)
            if ep > args.warmup_epochs: scheduler.step()

            hr = va.get("mae_HR",0); rr = va.get("mae_RespRate",0); sp = va.get("mae_SpO2",0)
            history.append({"epoch":ep, "tr_loss":tr["loss"], "va_loss":va["loss"],
                             "va_mae_HR":hr, "va_mae_RespRate":rr, "va_mae_SpO2":sp})
            print(f"Ep{ep:3d} | tr={tr['loss']:.4f} | "
                  f"va={va['loss']:.4f} HR={hr:.3f} RR={rr:.3f} SpO2={sp:.3f}")

            if va["loss"] < best_loss:
                best_loss, pat_cnt = va["loss"], 0
                torch.save(model.state_dict(), ckpt)
                print(f"  ✓ Best (loss={best_loss:.4f})")
            else:
                pat_cnt += 1
                if pat_cnt >= args.patience:
                    print(f"  Early stop @ep{ep}"); break

        model.load_state_dict(torch.load(ckpt, map_location=device))
        with open(os.path.join(args.output_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
        print(f"Best val_loss: {best_loss:.4f}")

    if args.phase in ("predict", "all"):
        ckpt = os.path.join(args.output_dir, "best_model.pt")
        model.load_state_dict(torch.load(ckpt, map_location=device))
        for split in ("train", "val", "test"):
            run_predict(model, args, device, split)

    print("\n✓ Done.")


if __name__ == "__main__":
    main()