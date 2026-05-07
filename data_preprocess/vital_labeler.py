import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import warnings


OUTLIER_RANGES = {
    "HR":          (30,  220),
    "SysABP":      (40,  250),
    "DiasABP":     (20,  180),
    "MAP":         (20,  200),
    "RespRate":    (5,   80),
    "SpO2":        (50,  100),
    "Temp_C":      (30,  42),
}

NORMAL_RANGES = {
    "HR":       (60,  100),
    "SysABP":   (90,  140),
    "DiasABP":  (60,  90),
    "MAP":      (70,  100),
    "RespRate": (12,  20),
    "SpO2":     (95,  100),
    "Temp_C":   (36.0, 38.0),
}

DELTA_THRESHOLD = {
    "HR":       8.0,
    "SysABP":   8.0,
    "DiasABP":  6.0,
    "MAP":      6.0,
    "RespRate": 3.0,
    "SpO2":     2.0,
    "Temp_C":   0.4,
}

IMPROVEMENT_DIRECTION = {
    "HR":       -1,
    "SysABP":   +1,
    "DiasABP":  +1,
    "MAP":      +1,
    "RespRate": -1,
    "SpO2":     +1,
    "Temp_C":    0,
}

MIN_OBS_FRACTION = 0.30

HIGH_VARIABILITY_MULTIPLIER = 2.0

BURDEN_CHANGE_THRESHOLD = 0.15


def _clean(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    arr = arr.astype(float).copy()
    arr[(arr < lo) | (arr > hi)] = np.nan
    return arr


def _obs_fraction(arr: np.ndarray) -> float:
    return np.sum(~np.isnan(arr)) / len(arr)


def _safe_mean(arr: np.ndarray) -> float:
    v = arr[~np.isnan(arr)]
    return float(np.mean(v)) if len(v) > 0 else np.nan


def _slope(arr: np.ndarray) -> float:
    idx = np.where(~np.isnan(arr))[0]
    if len(idx) < 2:
        return np.nan
    x = idx.astype(float)
    y = arr[idx]
    x -= x.mean()
    return float(np.dot(x, y) / np.dot(x, x)) if np.dot(x, x) != 0 else 0.0


def _mean_abs_diff(arr: np.ndarray) -> float:
    v = arr[~np.isnan(arr)]
    if len(v) < 2:
        return np.nan
    return float(np.mean(np.abs(np.diff(v))))


def _abnormal_burden(arr: np.ndarray, lo: float, hi: float) -> float:
    v = arr[~np.isnan(arr)]
    if len(v) == 0:
        return np.nan
    return float(np.mean((v < lo) | (v > hi)))


def extract_features(arr: np.ndarray, var: str) -> dict:
    lo_out, hi_out = OUTLIER_RANGES[var]
    lo_norm, hi_norm = NORMAL_RANGES[var]

    arr = _clean(arr, lo_out, hi_out)
    n = len(arr)
    mid = n // 2

    obs_frac = _obs_fraction(arr)

    q = max(1, n // 5)
    first_val = _safe_mean(arr[:q])
    last_val  = _safe_mean(arr[-q:])
    delta     = (last_val - first_val) if not (np.isnan(first_val) or np.isnan(last_val)) else np.nan

    sl            = _slope(arr)
    mad           = _mean_abs_diff(arr)
    burden_all    = _abnormal_burden(arr, lo_norm, hi_norm)
    burden_first  = _abnormal_burden(arr[:mid], lo_norm, hi_norm)
    burden_second = _abnormal_burden(arr[mid:], lo_norm, hi_norm)
    burden_change = (
        burden_second - burden_first
        if not (np.isnan(burden_first) or np.isnan(burden_second))
        else np.nan
    )

    delta_thr = DELTA_THRESHOLD[var]
    is_high_var = (mad > HIGH_VARIABILITY_MULTIPLIER * delta_thr) if not np.isnan(mad) else False

    return dict(
        obs_fraction   = obs_frac,
        first_val      = first_val,
        last_val       = last_val,
        delta          = delta,
        slope          = sl,
        mean_abs_diff  = mad,
        burden_overall = burden_all,
        burden_first   = burden_first,
        burden_second  = burden_second,
        burden_change  = burden_change,
        is_high_variability = is_high_var,
    )


def _score_direction(delta, slope, burden_change, var: str) -> float:
    direction = IMPROVEMENT_DIRECTION[var]
    delta_thr = DELTA_THRESHOLD[var]
    score = 0.0

    if var == "Temp_C":
        lo_n, hi_n = NORMAL_RANGES[var]
        mid_n = (lo_n + hi_n) / 2.0
        if not np.isnan(delta):
            score += 0.0
    else:
        if not np.isnan(delta):
            if abs(delta) >= delta_thr:
                score += direction * np.sign(delta)

        if not np.isnan(slope):
            if abs(slope) >= delta_thr / 10.0:
                score += direction * np.sign(slope)

        if not np.isnan(burden_change):
            if abs(burden_change) >= BURDEN_CHANGE_THRESHOLD:
                score += -1.0 * np.sign(burden_change)

    return score


def label_variable(arr: np.ndarray, var: str) -> tuple[int, dict]:
    feats = extract_features(arr, var)

    if feats["obs_fraction"] < MIN_OBS_FRACTION:
        return -1, feats

    delta         = feats["delta"]
    slope         = feats["slope"]
    burden_change = feats["burden_change"]
    first_val     = feats["first_val"]
    is_high_var   = feats["is_high_variability"]
    delta_thr     = DELTA_THRESHOLD[var]

    if var == "Temp_C":
        lo_n, hi_n = NORMAL_RANGES[var]
        mid_n = (lo_n + hi_n) / 2.0
        score = 0.0
        if not np.isnan(delta) and not np.isnan(first_val):
            side = np.sign(first_val - mid_n)
            if abs(delta) >= delta_thr:
                score += -side * np.sign(delta)
        if not np.isnan(slope) and not np.isnan(first_val):
            side = np.sign(first_val - mid_n)
            if abs(slope) >= delta_thr / 10.0:
                score += -side * np.sign(slope)
        if not np.isnan(burden_change):
            if abs(burden_change) >= BURDEN_CHANGE_THRESHOLD:
                score += -1.0 * np.sign(burden_change)
    else:
        score = _score_direction(delta, slope, burden_change, var)

    if is_high_var and abs(score) < 1.0:
        score -= 0.5

    if score >= 1.5:
        label = 0
    elif score <= -1.5:
        label = 2
    else:
        label = 1

    feats["score"] = score
    return label, feats


LABEL_NAMES = {0: "improving", 1: "stable", 2: "worsening", -1: "unknown"}

VARIABLES = ["HR", "SysABP", "DiasABP", "MAP", "RespRate", "SpO2", "Temp_C"]


def label_patient_window(
    post_window: dict[str, np.ndarray],
    variables: list[str] = None,
    return_features: bool = False,
) -> dict:
    if variables is None:
        variables = VARIABLES

    labels   = {}
    feat_out = {}

    for var in variables:
        if var not in post_window or post_window[var] is None:
            labels[var] = -1
            feat_out[var] = {}
            continue
        lbl, feats = label_variable(post_window[var], var)
        labels[var]   = lbl
        feat_out[var] = feats

    composite = _compute_composite(labels)

    result = {
        "labels":       labels,
        "label_names":  {v: LABEL_NAMES[l] for v, l in labels.items()},
        "composite":    composite,
    }
    if return_features:
        result["features"] = feat_out
    return result


def _composite_from_subset(labels: dict, keys: list[str]) -> int:
    valid = [labels[k] for k in keys if labels.get(k, -1) != -1]
    if not valid:
        return -1
    counts = {0: valid.count(0), 1: valid.count(1), 2: valid.count(2)}
    return max(counts, key=lambda x: (counts[x], -x))


def _compute_composite(labels: dict) -> dict:
    cardio = _composite_from_subset(labels, ["HR", "SysABP", "DiasABP", "MAP"])
    resp   = _composite_from_subset(labels, ["RespRate", "SpO2"])
    therm  = labels.get("Temp_C", -1)

    all_labels = (
        [labels.get("HR", -1), labels.get("SysABP", -1),
         labels.get("DiasABP", -1), labels.get("MAP", -1)] +
        [labels.get("RespRate", -1), labels.get("SpO2", -1)] +
        [labels.get("Temp_C", -1)]
    )
    valid_all = [l for l in all_labels if l != -1]
    if valid_all:
        counts = {0: valid_all.count(0), 1: valid_all.count(1), 2: valid_all.count(2)}
        overall = max(counts, key=lambda x: counts[x])
    else:
        overall = -1

    return {
        "cardiovascular": cardio,
        "respiratory":    resp,
        "thermal":        therm,
        "overall":        overall,
        "cardiovascular_name": LABEL_NAMES[cardio],
        "respiratory_name":    LABEL_NAMES[resp],
        "thermal_name":        LABEL_NAMES[therm],
        "overall_name":        LABEL_NAMES[overall],
    }


def label_dataset(
    post_windows: list[dict[str, np.ndarray]],
    variables: list[str] = None,
    return_features: bool = False,
) -> list[dict]:
    return [
        label_patient_window(pw, variables=variables, return_features=return_features)
        for pw in post_windows
    ]


def extract_label_matrix(
    results: list[dict],
    variables: list[str] = None,
    include_composite: bool = True,
) -> np.ndarray:
    if variables is None:
        variables = VARIABLES
    rows = []
    for r in results:
        row = [r["labels"].get(v, -1) for v in variables]
        if include_composite:
            c = r["composite"]
            row += [c["cardiovascular"], c["respiratory"], c["thermal"], c["overall"]]
        rows.append(row)
    return np.array(rows, dtype=np.int8)


def build_condition_vector(
    result: dict,
    variables: list[str] = None,
    mode: str = "onehot",
    include_composite: bool = True,
) -> np.ndarray:
    if variables is None:
        variables = VARIABLES

    labels = result["labels"]
    comp   = result["composite"]

    all_labels = [labels.get(v, -1) for v in variables]
    if include_composite:
        all_labels += [
            comp["cardiovascular"],
            comp["respiratory"],
            comp["thermal"],
            comp["overall"],
        ]

    if mode == "ordinal":
        return np.array(all_labels, dtype=np.float32)

    elif mode == "onehot":
        vec = []
        for lbl in all_labels:
            if lbl == -1:
                vec.extend([0, 0, 0])
            else:
                oh = [0, 0, 0]
                oh[lbl] = 1
                vec.extend(oh)
        return np.array(vec, dtype=np.float32)

    elif mode == "ordinal_with_mask":
        vec = []
        for lbl in all_labels:
            if lbl == -1:
                vec.extend([0, 0])
            else:
                vec.extend([lbl, 1])
        return np.array(vec, dtype=np.float32)

    elif mode == "onehot_with_mask":
        vec = []
        for lbl in all_labels:
            if lbl == -1:
                vec.extend([0, 0, 0, 0])
            else:
                oh = [0, 0, 0]
                oh[lbl] = 1
                vec.extend(oh + [1])
        return np.array(vec, dtype=np.float32)

    else:
        raise ValueError(
            f"Unknown mode '{mode}'. "
            "Use 'onehot', 'ordinal', 'onehot_with_mask', or 'ordinal_with_mask'."
        )


def summarize_label_distribution(
    results: list[dict],
    variables: list[str] = None,
) -> dict:
    if variables is None:
        variables = VARIABLES

    summary = {}
    n = len(results)
    for var in variables:
        counts = {0: 0, 1: 0, 2: 0, -1: 0}
        for r in results:
            counts[r["labels"].get(var, -1)] += 1
        summary[var] = {
            LABEL_NAMES[k]: f"{v} ({100*v/n:.1f}%)" for k, v in counts.items()
        }
    for sys_name in ["cardiovascular", "respiratory", "overall"]:
        counts = {0: 0, 1: 0, 2: 0, -1: 0}
        for r in results:
            counts[r["composite"].get(sys_name, -1)] += 1
        summary[f"[composite] {sys_name}"] = {
            LABEL_NAMES[k]: f"{v} ({100*v/n:.1f}%)" for k, v in counts.items()
        }
    return summary


if __name__ == "__main__":
    np.random.seed(42)
    n_bins = 24

    patient_A = {
        "HR":       np.linspace(130, 88, n_bins) + np.random.randn(n_bins) * 2,
        "SysABP":   np.linspace(78, 112, n_bins) + np.random.randn(n_bins) * 3,
        "DiasABP":  np.linspace(50, 72, n_bins)  + np.random.randn(n_bins) * 2,
        "MAP":      np.linspace(58, 85, n_bins)  + np.random.randn(n_bins) * 2,
        "RespRate": np.linspace(26, 16, n_bins)  + np.random.randn(n_bins) * 1,
        "SpO2":     np.linspace(90, 97, n_bins)  + np.random.randn(n_bins) * 0.5,
        "Temp_C":   np.linspace(38.8, 37.2, n_bins) + np.random.randn(n_bins) * 0.1,
    }

    patient_B = {
        "HR":       np.linspace(90, 128, n_bins) + np.random.randn(n_bins) * 2,
        "SysABP":   np.linspace(110, 78, n_bins) + np.random.randn(n_bins) * 4,
        "DiasABP":  np.linspace(72, 50, n_bins)  + np.random.randn(n_bins) * 2,
        "MAP":      np.linspace(85, 60, n_bins)  + np.random.randn(n_bins) * 3,
        "RespRate": np.linspace(16, 30, n_bins)  + np.random.randn(n_bins) * 1,
        "SpO2":     np.linspace(97, 89, n_bins)  + np.random.randn(n_bins) * 0.5,
        "Temp_C":   np.linspace(37.0, 39.2, n_bins) + np.random.randn(n_bins) * 0.1,
    }

    patient_C = {
        "HR":       np.full(n_bins, 82.0) + np.random.randn(n_bins) * 2,
        "SysABP":   np.full(n_bins, 118.0) + np.random.randn(n_bins) * 3,
        "DiasABP":  np.full(n_bins, 76.0) + np.random.randn(n_bins) * 2,
        "MAP":      np.full(n_bins, 83.0)  + np.random.randn(n_bins) * 2,
        "RespRate": np.full(n_bins, 17.0) + np.random.randn(n_bins) * 1,
        "SpO2":     np.full(n_bins, 93.0) + np.concatenate([
                        np.random.randn(n_bins // 2) * 5,
                        np.random.randn(n_bins // 2) * 5]),
        "Temp_C":   np.full(n_bins, 37.1) + np.random.randn(n_bins) * 0.05,
    }

    for name, pw in [("Patient A (recovering)", patient_A),
                     ("Patient B (deteriorating)", patient_B),
                     ("Patient C (stable/fluctuating)", patient_C)]:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print('='*60)
        result = label_patient_window(pw, return_features=True)

        for var in VARIABLES:
            lbl  = result["label_names"][var]
            feat = result["features"][var]
            score = feat.get("score", "n/a")
            print(f"  {var:<12} → {lbl:<12}  "
                  f"(delta={feat.get('delta', float('nan')):+.1f}, "
                  f"score={score})")

        print(f"\n  Composite:")
        c = result["composite"]
        print(f"    cardiovascular : {c['cardiovascular_name']}")
        print(f"    respiratory    : {c['respiratory_name']}")
        print(f"    thermal        : {c['thermal_name']}")
        print(f"    overall        : {c['overall_name']}")

        vec_onehot  = build_condition_vector(result, mode="onehot")
        vec_ordinal = build_condition_vector(result, mode="ordinal")
        print(f"\n  Condition vector (ordinal) : {vec_ordinal}")
        print(f"  Condition vector dim (onehot): {len(vec_onehot)}")