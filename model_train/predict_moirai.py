import os, pickle, argparse
import numpy as np
import torch

import sys
sys.path.insert(0, "/path/to/uni2ts/src")

from uni2ts.model.moirai import MoiraiForecast, MoiraiModule


FORECAST_NAMES = ["HR", "RespRate", "SpO2"]
N_VARS         = 3
QUANTILES      = [0.1, 0.25, 0.5, 0.75, 0.9]
N_Q            = len(QUANTILES)


def load_split(pkl_path, past_hours=36, future_hours=12):
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)

    total = past_hours + future_hours
    xs, ys, masks, labels, sids = [], [], [], [], []

    for d in raw:
        ft = d.get("forecasting_targets")
        if ft is None or ft.shape[0] < total:
            continue
        ft = ft.astype(np.float32)
        fm = d.get("forecast_masks")

        x    = ft[:past_hours, :]
        y    = ft[past_hours:total, :]
        mask = fm.astype(np.float32) if fm is not None else (y > 0).astype(np.float32)

        xs.append(x)
        ys.append(y)
        masks.append(mask)
        labels.append(int(d["label"]))
        sids.append(int(d["stay_id"]))

    print(f"  → {len(sids):,} stays")
    return xs, ys, masks, labels, sids


@torch.no_grad()
def predict_moirai(model, xs, batch_size, device, num_samples):
    """
    """
    N = len(xs)
    all_preds = []
    quantile_levels = torch.tensor(QUANTILES, dtype=torch.float32)

    for start in range(0, N, batch_size):
        end      = min(start + batch_size, N)
        batch_xs = xs[start:end]
        B        = len(batch_xs)

        past_target = torch.tensor(
            np.stack(batch_xs, axis=0), dtype=torch.float32
        ).to(device)

        past_observed_target = (past_target != 0).to(device)

        past_is_pad = torch.zeros(
            B, past_target.shape[1], dtype=torch.bool
        ).to(device)

        preds_samples = model(
            past_target=past_target,
            past_observed_target=past_observed_target,
            past_is_pad=past_is_pad,
            num_samples=num_samples,
        )

        if preds_samples.ndim == 3:
            preds_samples = preds_samples.unsqueeze(-1)

        preds_samples = preds_samples.cpu().float()

        preds_q = torch.quantile(
            preds_samples, quantile_levels, dim=1
        ).permute(1, 0, 2, 3)

        for i in range(B):
            all_preds.append(preds_q[i].numpy())

    return all_preds


def save_predictions(preds, xs, ys, masks, labels, sids, output_dir, split_name):
    results = {}
    for i, sid in enumerate(sids):
        results[sid] = {
            "preds": preds[i],
            "x"    : xs[i],
            "y"    : ys[i],
            "mask" : masks[i],
            "label": labels[i],
        }
    out = os.path.join(output_dir, f"{split_name}_predictions.pkl")
    with open(out, "wb") as f:
        pickle.dump(results, f, protocol=4)
    print(f"  → {len(results):,} stays → {out}")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="/path/to/processed_data")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--gpu", default="0")

    p.add_argument("--model_size", default="large",
        choices=["small", "base", "large"],
        help="")
    p.add_argument("--model_version", default="1.1",
        choices=["1.0", "1.1"])

    p.add_argument("--past_hours",   default=36,  type=int)
    p.add_argument("--future_hours", default=12,  type=int)

    p.add_argument("--patch_size", default=4, type=int,
        choices=[4, 8, 16, 32],
        help="")

    p.add_argument("--num_samples", default=100, type=int,
        help="")
    p.add_argument("--batch_size",  default=64,  type=int,
        help="")

    p.add_argument("--splits", default="train,val,test")
    return p.parse_args()


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device   = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model_id = f"Salesforce/moirai-{args.model_version}-R-{args.model_size}"

    print(f"{'='*55}")
    print(f"Model      : {model_id}")
    print(f"Device     : {device}")
    print(f"Patch size : {args.patch_size}  (fixed, auto removed)")
    print(f"Num samples: {args.num_samples}")
    print(f"Batch size : {args.batch_size}")
    print(f"{'='*55}")
    print("Loading Moirai...")

    model = MoiraiForecast(
        module=MoiraiModule.from_pretrained(model_id),
        prediction_length=args.future_hours,
        context_length=args.past_hours,
        patch_size=args.patch_size,
        num_samples=args.num_samples,
        target_dim=N_VARS,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    ).to(device)
    model.eval()
    print(f"✓ Loaded: {model_id}\n")

    for split in args.splits.split(","):
        split    = split.strip()
        pkl_path = os.path.join(args.data_dir, f"{split}.pkl")
        if not os.path.exists(pkl_path):
            print(f"  Skipping {split} (not found)")
            continue

        print(f"[{split}] Loading...")
        xs, ys, masks, labels, sids = load_split(
            pkl_path, args.past_hours, args.future_hours)

        print(f"[{split}] Predicting...")
        preds = predict_moirai(
            model, xs, args.batch_size, device, args.num_samples)

        print(f"[{split}] Saving...")
        save_predictions(preds, xs, ys, masks, labels, sids,
                         args.output_dir, split)

    print("\n✓ Done.")
    print(f"  python classifier_baseline.py --forecast_dir {args.output_dir} ...")
    print(f"  python cr_d.py --mode retrieval --forecast_dir {args.output_dir} ...")


if __name__ == "__main__":
    main()