import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.dataset import load_datasets
from models.encoder import PatientEncoder


@torch.no_grad()
def extract(model, loader, device) -> dict:
    model.eval()
    embs, labels, sofas, sids = [], [], [], []
    for batch in loader:
        ts  = batch["ts_rich"].to(device)
        sn  = batch["static_num"].to(device)
        sc  = batch["static_cat"].to(device)
        emb = model(ts, static_num=sn, static_cat=sc)

        embs.append(emb.cpu().numpy())
        labels.append(batch["label"].numpy())
        sofas.append(batch["sofa_proxy"].numpy())
        sids.append(np.array(batch["stay_id"]))

    return {
        "embeddings": np.concatenate(embs).astype(np.float32),
        "labels"    : np.concatenate(labels).astype(np.int32),
        "sofa"      : np.concatenate(sofas).astype(np.float32),
        "stay_ids"  : np.concatenate(sids).astype(np.int64),
    }


def main():
    p = argparse.ArgumentParser(description="")
    p.add_argument("--data_dir",    type=str, default="/path/to/output")
    p.add_argument("--stage2_ckpt", type=str, default="/path/to/output")
    p.add_argument("--out_dir",     type=str, default="/path/to/output",
                   help="")
    p.add_argument("--splits",      nargs="+", default=["train","val","test"])
    p.add_argument("--batch_size",  type=int, default=512)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device",      type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device  = torch.device(args.device)
    out_dir = args.out_dir or str(Path(args.stage2_ckpt).parent)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nLoading model from {args.stage2_ckpt}")
    ckpt  = torch.load(args.stage2_ckpt, map_location=device, weights_only=False)
    saved = ckpt.get("args", {})

    model = PatientEncoder(
        ts_input_dim       = 183,
        ts_hidden_dim      = saved.get("hidden_dim", 256),
        ts_n_layers        = saved.get("n_layers",   6),
        static_numeric_dim = 33,
        static_out_dim     = 64,
        embed_dim          = saved.get("embed_dim",  256),
        use_static         = True,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  embed_dim={saved.get('embed_dim', 256)}, "
          f"sep_score={ckpt.get('sep_score', 'N/A')}")

    print(f"\nLoading datasets from {args.data_dir}")
    train_ds, val_ds, test_ds = load_datasets(args.data_dir, mode="both")
    ds_map = {"train": train_ds, "val": val_ds, "test": test_ds}

    for sp in args.splits:
        if sp not in ds_map:
            print(f"  [SKIP] unknown split: {sp}")
            continue

        loader = DataLoader(
            ds_map[sp], batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers, pin_memory=True,
        )

        print(f"\n  Extracting [{sp}] ({len(ds_map[sp]):,} samples)...",
              end=" ", flush=True)
        data = extract(model, loader, device)
        print("done")

        mort = data["labels"].mean()
        print(f"    embeddings : {data['embeddings'].shape}  "
              f"(L2-norm mean={np.linalg.norm(data['embeddings'], axis=1).mean():.4f})")
        print(f"    mortality  : {mort*100:.2f}%")
        print(f"    SOFA mean  : {data['sofa'].mean():.4f}")

        out_path = os.path.join(out_dir, f"{sp}_embeddings.pt")
        torch.save(data, out_path)
        print(f"    Saved → {out_path}")

    print(f"\n✓ Done. All embeddings saved to {out_dir}/")


if __name__ == "__main__":
    main()