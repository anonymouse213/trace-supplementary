import os, pickle, argparse
import numpy as np
import torch
from tqdm import tqdm

try:
    from chronos import BaseChronosPipeline
except ImportError:
    raise ImportError("pip install chronos-forecasting")


FORECAST_NAMES = ["HR", "RespRate", "SpO2"]
N_VARS         = 3
QUANTILES      = [0.1, 0.25, 0.5, 0.75, 0.9]
N_Q            = len(QUANTILES)
MED_IDX        = N_Q // 2


def load_split(pkl_path, past_hours=36, future_hours=12):
    """
    """
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

        x = ft[-total       : -future_hours if future_hours else None, :]
        y = ft[-future_hours:, :]
        mask = fm.astype(np.float32) if fm is not None else (y > 0).astype(np.float32)

        xs.append(x)
        ys.append(y)
        masks.append(mask)
        labels.append(int(d["label"]))
        sids.append(int(d["stay_id"]))

    print(f"  → {len(sids):,} stays")
    return xs, ys, masks, labels, sids


@torch.no_grad()
def predict_chronos(pipeline, xs, future_hours, batch_size, device):
    """
    xs: list of (36, 3) arrays
    """
    N = len(xs)
    all_preds = []

    xs_arr = np.stack(xs, axis=0)

    var_quantiles = []
    for vi in range(N_VARS):
        ctx_vi = xs_arr[:, :, vi]

        ctx_vi = ctx_vi.copy()
        ctx_vi[ctx_vi == 0] = np.nan

        q_vi_all = []
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            ctx_batch = [torch.tensor(ctx_vi[i], dtype=torch.float32) for i in range(start, end)]

            quantiles, _ = pipeline.predict_quantiles(
                inputs=ctx_batch,
                prediction_length=future_hours,
                quantile_levels=QUANTILES,
            )
            if isinstance(quantiles, list):
                quantiles = torch.stack(quantiles)
            q_vi_all.append(quantiles.cpu().numpy())

        q_vi_all = np.concatenate(q_vi_all, axis=0)
        var_quantiles.append(q_vi_all)

    for i in range(N):
        pred_i = np.stack([var_quantiles[vi][i] for vi in range(N_VARS)], axis=-1)
        pred_i = pred_i.transpose(1, 0, 2)
        all_preds.append(pred_i)

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
    p.add_argument("--data_dir",
        default="/path/to/processed_data")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--gpu",   default="0")

    p.add_argument("--model_id", default="amazon/chronos-t5-large",
        help=""
             "baseline=amazon/chronos-bolt-tiny, "
             "ours=amazon/chronos-bolt-base")
    p.add_argument("--torch_dtype", default="float32",
        choices=["float32", "bfloat16"])

    p.add_argument("--past_hours",   default=36,   type=int)
    p.add_argument("--future_hours", default=12,   type=int)
    p.add_argument("--num_samples",  default=20,  type=int)
    p.add_argument("--batch_size",   default=512,  type=int,
        help="")

    p.add_argument("--splits", default="train,val,test",
        help="")
    return p.parse_args()


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float32

    print(f"Device: {device}")
    print(f"Model : {args.model_id}")
    print(f"Dtype : {dtype}")
    print(f"Splits: {args.splits}")
    print(f"\n{'='*55}")
    print("Loading Chronos pipeline (pretrained, no training needed)")
    print(f"{'='*55}")

    pipeline = BaseChronosPipeline.from_pretrained(
        args.model_id,
        device_map=device,
        torch_dtype=dtype,
    )
    print(f"✓ Loaded: {args.model_id}")
    print(f"  Quantiles: {QUANTILES}")
    print(f"  Prediction horizon: {args.future_hours}h")

    for split in args.splits.split(","):
        split = split.strip()
        pkl_path = os.path.join(args.data_dir, f"{split}.pkl")
        if not os.path.exists(pkl_path):
            print(f"  Skipping {split} (not found: {pkl_path})")
            continue

        print(f"\n[{split}] Loading...")
        xs, ys, masks, labels, sids = load_split(
            pkl_path, args.past_hours, args.future_hours)

        print(f"[{split}] Predicting with Chronos...")
        preds = predict_chronos(
            pipeline, xs, args.future_hours, args.batch_size, device)

        print(f"[{split}] Saving...")
        save_predictions(preds, xs, ys, masks, labels, sids,
                         args.output_dir, split)

    print("\n✓ Done.")
    print(f"  python classifier_baseline.py --forecast_dir {args.output_dir} ...")
    print(f"  python cr_d.py --mode retrieval --forecast_dir {args.output_dir} ...")


if __name__ == "__main__":
    main()