import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


SOFA_IDX = {
    "respiratory" : 7,
    "renal"       : 21,
    "hepatic"     : 50,
    "coagulation" : 37,
    "cardiovascular": slice(56, 61),
}

def compute_sofa_proxy(ts_numeric: np.ndarray) -> float:
    """
    """
    def last_valid(arr):
        nonzero = arr[arr != 0]
        return float(nonzero[-1]) if len(nonzero) > 0 else 0.0

    resp = 1.0 - last_valid(ts_numeric[:, SOFA_IDX["respiratory"]])

    renal = last_valid(ts_numeric[:, SOFA_IDX["renal"]])

    hepatic = last_valid(ts_numeric[:, SOFA_IDX["hepatic"]])

    coag = 1.0 - last_valid(ts_numeric[:, SOFA_IDX["coagulation"]])

    vaso_cols = ts_numeric[:, SOFA_IDX["cardiovascular"]]
    cardio = float(np.max([last_valid(vaso_cols[:, i]) for i in range(5)]))

    sofa = resp + renal + hepatic + coag + cardio
    return float(np.clip(sofa, 0.0, 5.0))


class ICUPatientDataset(Dataset):
    """

    Args:
        mode      : 'stage1' | 'stage2' | 'both'
    """

    def __init__(
        self,
        pkl_path: str,
        mode: str = "both",
        normalize_sofa: bool = True,
    ):
        assert mode in ("stage1", "stage2", "both")
        self.mode = mode
        self.normalize_sofa = normalize_sofa

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        self.samples = []
        for d in data:
            ts_rich    = d["historical_ts_rich"].astype(np.float32)
            ts_numeric = d["historical_ts_numeric"].astype(np.float32)
            label      = int(d["label"])
            stay_id    = int(d["stay_id"])
            static_num = d["static_numeric"].astype(np.float32)
            static_cat = d["static_categoric"].astype(np.int32)
            sofa       = compute_sofa_proxy(ts_numeric)

            self.samples.append({
                "stay_id"       : stay_id,
                "ts_rich"       : ts_rich,
                "ts_numeric"    : ts_numeric,
                "static_num"    : static_num,
                "static_cat"    : static_cat,
                "label"         : label,
                "sofa_proxy"    : sofa,
            })

        all_sofa = np.array([s["sofa_proxy"] for s in self.samples])
        self.sofa_mean = float(all_sofa.mean())
        self.sofa_std  = float(all_sofa.std()) + 1e-8

        print(f"  Loaded {len(self.samples):,} samples from {pkl_path}")
        print(f"  SOFA proxy — mean: {self.sofa_mean:.3f}, std: {self.sofa_std:.3f}")
        print(f"  Mortality rate: {sum(s['label'] for s in self.samples)/len(self.samples)*100:.2f}%")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        sofa = s["sofa_proxy"]
        if self.normalize_sofa:
            sofa = (sofa - self.sofa_mean) / self.sofa_std

        out = {
            "stay_id"    : s["stay_id"],
            "label"      : torch.tensor(s["label"], dtype=torch.long),
            "sofa_proxy" : torch.tensor(sofa, dtype=torch.float32),
        }

        if self.mode in ("stage1", "both"):
            out["ts_rich"]    = torch.tensor(s["ts_rich"],    dtype=torch.float32)

        if self.mode in ("stage2", "both"):
            out["ts_numeric"] = torch.tensor(s["ts_numeric"], dtype=torch.float32)
            out["static_num"] = torch.tensor(s["static_num"], dtype=torch.float32)
            out["static_cat"] = torch.tensor(s["static_cat"], dtype=torch.long)

        return out


def load_datasets(
    data_dir: str,
    mode: str = "both",
    normalize_sofa: bool = True,
):
    root = Path(data_dir)
    train_ds = ICUPatientDataset(root / "train.pkl", mode=mode, normalize_sofa=normalize_sofa)
    val_ds   = ICUPatientDataset(root / "val.pkl",   mode=mode, normalize_sofa=normalize_sofa)
    test_ds  = ICUPatientDataset(root / "test.pkl",  mode=mode, normalize_sofa=normalize_sofa)

    for ds in (val_ds, test_ds):
        ds.sofa_mean = train_ds.sofa_mean
        ds.sofa_std  = train_ds.sofa_std

    return train_ds, val_ds, test_ds