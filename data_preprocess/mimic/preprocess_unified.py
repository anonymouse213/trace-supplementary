import os
import pickle
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from vital_labeler import (
    label_patient_window,
    OUTLIER_RANGES,
    NORMAL_RANGES,
    MIN_OBS_FRACTION,
    LABEL_NAMES,
)
from variable_dict import (
    ALL_CHARTEVENTS_ITEMIDS,
    ALL_VASOPRESSOR_ITEMIDS,
    ITEMID_TO_FEATURE,
    FEATURE_ORDER,
    FEATURE_INDEX,
    HOURLY_FEATURES,
    DAILY_FEATURES,
    N_FEATURES,
)

BASE             = "/path/to/mimiciv"
CHARTEVENTS_PATH = f"{BASE}/icu/chartevents.csv"
INPUTEVENTS_PATH = f"{BASE}/icu/inputevents.csv"
ICUSTAYS_PATH    = f"{BASE}/icu/icustays.csv"
ADMISSIONS_PATH  = f"{BASE}/hosp/admissions.csv"
PATIENTS_PATH    = f"{BASE}/hosp/patients.csv"

CHUNK_SIZE   = 2_000_000
MIN_LOS_DAYS = 2.0
RANDOM_STATE = 42
TRAIN_RATIO  = 0.8
VAL_RATIO    = 0.1

ITEMID_WEIGHT_KG   = 226512
ITEMID_WEIGHT_LBS  = 226531
ITEMID_HEIGHT_INCH = 226707
ITEMID_HEIGHT_CM   = 226730
ITEMID_TEMP_F      = 223761
STATIC_ITEMIDS     = {ITEMID_WEIGHT_KG, ITEMID_WEIGHT_LBS,
                      ITEMID_HEIGHT_INCH, ITEMID_HEIGHT_CM}

TARGET_VITALS  = ["HR", "SysABP", "DiasABP", "SpO2", "RespRate", "Temp_C"]
TARGET_INDICES = [FEATURE_INDEX[v] for v in TARGET_VITALS]
N_TARGETS      = len(TARGET_VITALS)

FORECAST_VITALS   = ["HR", "RespRate", "SpO2"]
FORECAST_INDICES  = [FEATURE_INDEX[v] for v in FORECAST_VITALS]
N_FORECAST        = len(FORECAST_VITALS)

HOURS_SINCE_MAX   = 99.0

N_RICH_FEATURES   = N_FEATURES * 3

LABELER_VITALS     = ["HR", "SysABP", "DiasABP", "RespRate", "SpO2", "Temp_C"]
COMPOSITE_KEYS     = ["cardiovascular", "respiratory", "thermal", "overall"]

COND_VEC_VITAL_DIM     = 6
COND_VEC_COMPOSITE_DIM = 4
COND_VEC_DIM = (len(LABELER_VITALS) * COND_VEC_VITAL_DIM +
                len(COMPOSITE_KEYS) * COND_VEC_COMPOSITE_DIM)

COND_VEC_FORECAST_DIM = len(FORECAST_VITALS) * COND_VEC_VITAL_DIM

ICU_TYPE_MAP = {
    "Coronary Care Unit (CCU)":                              1,
    "Cardiac Vascular Intensive Care Unit (CVICU)":          2,
    "Medical Intensive Care Unit (MICU)":                    3,
    "Medical/Surgical Intensive Care Unit (MICU/SICU)":      3,
    "Surgical Intensive Care Unit (SICU)":                   4,
    "Trauma SICU (TSICU)":                                   4,
    "Neuro Surgical Intensive Care Unit (Neuro SICU)":       4,
    "Neuro Intermediate":                                    3,
    "Neuro Stepdown":                                        3,
}

OUTLIER_CLIP = {
    "HR":(30,220), "SysABP":(40,250), "DiasABP":(20,180),
    "RespRate":(5,80), "SpO2":(50,100), "Temp_C":(30,42), "Temp_F":(86,108),
    "PaO2":(20,500), "PaCO2":(10,150), "ArterialPH":(6.8,7.8),
    "VenousPH":(6.8,7.8), "SaO2_art":(50,100), "HCO3":(5,50), "AnionGap":(2,40),
    "LacticAcid":(0,20),
    "Creatinine":(0,66),
    "BUN":(0,275),
    "Sodium_serum":(100,180), "Sodium_whole":(100,180),
    "Potassium_serum":(1.5,8.0), "Potassium_whole":(1.5,8.0),
    "Magnesium":(0.5,6.0), "Phosphate":(0.5,10.0), "Calcium":(4,14),
    "Chloride":(70,130),
    "Glucose_serum":(20,1000), "Glucose_finger":(20,1000), "Glucose_whole":(20,1000),
    "WBC":(0,1100),
    "Platelets":(0,2200),
    "Hemoglobin":(2,25), "HCT_serum":(5,80), "HCT_calc":(5,80),
    "Albumin":(0.5,6.0),
    "TotalBili":(0,66),
    "AST":(0,22000),
    "ALT":(0,11000),
    "ALP":(0,4000),
    "TroponinT":(0,500),
    "Epinephrine":(0,200), "Norepinephrine":(0,200),
    "Dopamine":(0,200), "Phenylephrine":(0,200), "Vasopressin":(0,10),
}


def get_args():
    p = argparse.ArgumentParser(description="MIMIC-IV unified preprocessing")
    p.add_argument("--past_hours",    type=int, default=36,
                   help="")
    p.add_argument("--future_hours",  type=int, default=12,
                   help="")
    p.add_argument("--out_dir",       type=str, default="./processed_36_12",
                   help="")
    p.add_argument("--convert_duett", action="store_true",
                   help="")
    p.add_argument("--duett_dir",     type=str, default=None,
                   help=""
    p.add_argument("--min_los_days",  type=float, default=2.0)
    p.add_argument("--train_ratio",   type=float, default=0.8)
    p.add_argument("--val_ratio",     type=float, default=0.1)
    return p.parse_args()


def clip_value(var: str, val: float) -> float:
    if pd.isna(val): return val
    lo, hi = OUTLIER_CLIP.get(var, (None, None))
    if lo is not None:
        val = max(lo, min(val, hi))
    return val


def convert_obj(obj):
    if isinstance(obj, np.ndarray):
        return np.array(obj, dtype=obj.dtype)
    elif isinstance(obj, dict):
        return {k: convert_obj(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_obj(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_obj(i) for i in obj)
    return obj


def save_pickle(obj, path, duett_compat=False):
    if duett_compat:
        obj = convert_obj(obj)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    print(f"  → {path}  ({os.path.getsize(path)/1e6:.1f} MB)"
          + (" [duett-compat]" if duett_compat else ""))


def build_condition_vector(
    label_result: dict,
    post_window:  dict,
    feat_min_map: dict,
    feat_max_map: dict,
) -> np.ndarray:
    vec = np.zeros(COND_VEC_DIM, dtype=np.float32)
    labels    = label_result["labels"]
    features  = label_result.get("features", {})
    composite = label_result["composite"]

    for i, var in enumerate(LABELER_VITALS):
        base = i * COND_VEC_VITAL_DIM
        lbl  = labels.get(var, -1)
        if lbl == -1: continue
        vec[base + lbl] = 1.0
        vec[base + 3]   = 1.0
        feat = features.get(var, {})
        burden = feat.get("burden_overall", np.nan)
        if not np.isnan(burden):
            vec[base + 4] = float(np.clip(burden, 0.0, 1.0))
        arr = post_window.get(var)
        if arr is not None and feat_min_map is not None:
            valid = arr[~np.isnan(arr)]
            if len(valid) > 0:
                lv   = float(valid[-1])
                lo   = feat_min_map.get(var, 0.0)
                hi   = feat_max_map.get(var, 1.0)
                denom = hi - lo if (hi - lo) > 1e-8 else 1.0
                vec[base + 5] = float(np.clip((lv - lo) / denom, 0.0, 1.0))

    comp_base = len(LABELER_VITALS) * COND_VEC_VITAL_DIM
    for j, key in enumerate(COMPOSITE_KEYS):
        base = comp_base + j * COND_VEC_COMPOSITE_DIM
        lbl  = composite.get(key, -1)
        if lbl == -1: continue
        vec[base + lbl] = 1.0
        vec[base + 3]   = 1.0

    return vec


def build_cohort(args) -> pd.DataFrame:
    print("\n[1] Building cohort...")
    icu = pd.read_csv(ICUSTAYS_PATH, parse_dates=["intime","outtime"])
    adm = pd.read_csv(ADMISSIONS_PATH,
                      parse_dates=["admittime","dischtime","deathtime"],
                      usecols=["admittime","dischtime","hadm_id",
                               "hospital_expire_flag","deathtime"])
    pat = pd.read_csv(PATIENTS_PATH, usecols=["subject_id","anchor_age","gender"])

    icu = icu[icu["los"] >= args.min_los_days].copy()
    print(f"  los >= {args.min_los_days}d: {len(icu):,} stays")

    icu = icu.merge(adm, on="hadm_id", how="left")
    icu = icu.merge(pat, on="subject_id", how="left")

    icu["died_in_icu"] = (
        (icu["hospital_expire_flag"] == 1)
    ).astype(int)

    icu["icu_type_num"] = icu["last_careunit"].map(ICU_TYPE_MAP).fillna(0).astype(int)

    subjects = icu["subject_id"].unique()
    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(subjects)
    n       = len(subjects)
    n_train = int(n * args.train_ratio)
    n_val   = int(n * args.val_ratio)
    train_s = set(subjects[:n_train])
    val_s   = set(subjects[n_train:n_train+n_val])
    test_s  = set(subjects[n_train+n_val:])

    def assign(sid):
        if sid in test_s: return "test"
        if sid in val_s:  return "val"
        return "train"

    icu["split"] = icu["subject_id"].map(assign)
    for sp in ["train","val","test"]:
        sub = icu[icu["split"]==sp]
        print(f"  {sp}: {len(sub):,} stays | mort={sub['died_in_icu'].mean():.3f}")
    return icu.reset_index(drop=True)


def build_static_features(icu: pd.DataFrame, total_hours: int) -> dict:
    print("\n[2] Building static features...")
    stay_ids = set(icu["stay_id"])
    icu_time = icu[["stay_id","intime"]].set_index("stay_id")
    hw_first: dict = defaultdict(dict)

    for chunk in pd.read_csv(CHARTEVENTS_PATH, chunksize=CHUNK_SIZE,
                             usecols=["stay_id","charttime","itemid","valuenum"]):
        chunk = chunk[chunk["itemid"].isin(STATIC_ITEMIDS) &
                      chunk["stay_id"].isin(stay_ids) &
                      chunk["valuenum"].notna()]
        if chunk.empty: continue
        chunk["charttime"] = pd.to_datetime(chunk["charttime"])
        chunk = chunk.join(icu_time, on="stay_id", how="left")
        chunk["offset_h"] = (chunk["charttime"]-chunk["intime"]).dt.total_seconds()/3600
        chunk = chunk[(chunk["offset_h"]>=0)&(chunk["offset_h"]<=total_hours)]
        chunk = chunk.sort_values("charttime")
        for row in chunk.itertuples(index=False):
            sid, iid = int(row.stay_id), int(row.itemid)
            if iid not in hw_first[sid]:
                hw_first[sid][iid] = float(row.valuenum)

    static_map = {}
    for row in icu[["stay_id","anchor_age","gender","icu_type_num"]].itertuples(index=False):
        sid = int(row.stay_id)
        hw  = hw_first.get(sid, {})
        weight_kg = (hw.get(ITEMID_WEIGHT_KG) or
                     (hw[ITEMID_WEIGHT_LBS]/2.20462 if ITEMID_WEIGHT_LBS in hw else np.nan))
        height_cm = (hw.get(ITEMID_HEIGHT_CM) or
                     (hw[ITEMID_HEIGHT_INCH]*2.54 if ITEMID_HEIGHT_INCH in hw else np.nan))
        static_map[sid] = {
            "age":       float(row.anchor_age),
            "gender":    1 if row.gender == "M" else 0,
            "height_cm": float(height_cm) if height_cm and not np.isnan(height_cm) else np.nan,
            "weight_kg": float(weight_kg) if weight_kg and not np.isnan(weight_kg) else np.nan,
            "icu_type":  int(row.icu_type_num),
            "_lab_raw": {},
        }
    return static_map


def build_raw_grids(icu: pd.DataFrame, total_hours: int) -> dict:
    print("\n[3] Processing CHARTEVENTS → raw grids...")
    stay_ids   = set(icu["stay_id"])
    icu_time   = icu[["stay_id","intime"]].set_index("stay_id")
    target_ids = set(ALL_CHARTEVENTS_ITEMIDS) - STATIC_ITEMIDS
    raw: dict  = defaultdict(lambda: defaultdict(dict))
    n_rows = 0

    for chunk in tqdm(
        pd.read_csv(CHARTEVENTS_PATH, chunksize=CHUNK_SIZE,
                    usecols=["stay_id","charttime","itemid","valuenum"]),
        desc="  chartevents"):
        chunk = chunk[chunk["itemid"].isin(target_ids) &
                      chunk["stay_id"].isin(stay_ids) &
                      chunk["valuenum"].notna()]
        if chunk.empty: continue
        chunk["charttime"] = pd.to_datetime(chunk["charttime"])
        chunk = chunk.join(icu_time, on="stay_id", how="left")
        chunk["offset_h"] = (chunk["charttime"]-chunk["intime"]).dt.total_seconds()/3600
        chunk = chunk[(chunk["offset_h"]>=0)&(chunk["offset_h"]<total_hours)]
        if chunk.empty: continue
        chunk["hour_bin"] = chunk["offset_h"].astype(int).clip(0, total_hours-1)
        chunk["feature"]  = chunk["itemid"].map(ITEMID_TO_FEATURE)
        chunk = chunk.dropna(subset=["feature"])

        for row in chunk.itertuples(index=False):
            feat = row.feature
            val  = float(row.valuenum)
            if feat == "Temp_F":
                val  = (val - 32.0) * 5.0 / 9.0
                feat = "Temp_C"
            val = clip_value(feat, val)
            if not np.isnan(val):
                raw[int(row.stay_id)][feat][int(row.hour_bin)] = val
        n_rows += len(chunk)

    print(f"  → {n_rows:,} rows | {len(raw):,} stays")
    return raw


def build_vasopressor_grids(icu: pd.DataFrame, total_hours: int) -> dict:
    print("\n[4] Processing INPUTEVENTS → vasopressor grids...")
    stay_ids   = set(icu["stay_id"])
    icu_time   = icu[["stay_id","intime"]].set_index("stay_id")
    target_ids = set(ALL_VASOPRESSOR_ITEMIDS)
    vaso: dict = defaultdict(lambda: defaultdict(list))
    n_rows = 0

    for chunk in tqdm(
        pd.read_csv(INPUTEVENTS_PATH, chunksize=CHUNK_SIZE,
                    usecols=["stay_id","starttime","endtime","itemid","rate","rateuom"]),
        desc="  inputevents"):
        chunk = chunk[chunk["itemid"].isin(target_ids) &
                      chunk["stay_id"].isin(stay_ids) &
                      chunk["rate"].notna() & (chunk["rate"]>0)]
        if chunk.empty: continue
        chunk["starttime"] = pd.to_datetime(chunk["starttime"])
        chunk["endtime"]   = pd.to_datetime(chunk["endtime"])
        chunk = chunk.join(icu_time, on="stay_id", how="left")
        chunk["feature"] = chunk["itemid"].map(ITEMID_TO_FEATURE)

        for row in chunk.itertuples(index=False):
            start_h = max(0.0, (row.starttime-row.intime).total_seconds()/3600)
            end_h   = min(float(total_hours), (row.endtime-row.intime).total_seconds()/3600)
            if start_h >= end_h: continue
            rate = clip_value(row.feature, float(row.rate))
            for b in range(int(start_h), min(int(end_h), total_hours-1)+1):
                vaso[int(row.stay_id)][row.feature].append((b, rate))
        n_rows += len(chunk)

    print(f"  → {n_rows:,} records | {len(vaso):,} stays with pressors")
    return vaso


def _build_forecast_cond_vec(label_result, post_window,
                              feat_min_map, feat_max_map) -> np.ndarray:
    """
    """
    vec = np.zeros(COND_VEC_FORECAST_DIM, dtype=np.float32)
    for vi, v in enumerate(FORECAST_VITALS):
        base  = vi * COND_VEC_VITAL_DIM
        label = label_result["labels"].get(v, -1)
        if label == 0: vec[base + 0] = 1.0
        elif label == 1: vec[base + 1] = 1.0
        elif label == 2: vec[base + 2] = 1.0

        arr = post_window.get(v)
        if arr is not None and not np.all(np.isnan(arr)):
            vec[base + 3] = 1.0
            valid = arr[~np.isnan(arr)]
            if len(valid) > 0:
                lo = feat_min_map.get(v, 0.0) if feat_min_map else 0.0
                hi = feat_max_map.get(v, 1.0) if feat_max_map else 1.0
                denom = max(hi - lo, 1e-8)
                burden = float(np.mean(np.clip((valid - lo) / denom, 0, 1)))
                last_val = float(np.clip((valid[-1] - lo) / denom, 0, 1))
                vec[base + 4] = burden
                vec[base + 5] = last_val
    return vec


def build_hourly_arrays(
    icu: pd.DataFrame,
    raw: dict,
    vaso: dict,
    past_hours: int,
    future_hours: int,
    feat_min_map: dict,
    feat_max_map: dict,
    static_map: dict = None,
) -> tuple:
    """
      label_counts : dict
    """
    total_hours = past_hours + future_hours
    print(f"\n[5-6] Building hourly arrays "
          f"(past={past_hours}h / future={future_hours}h / total={total_hours}h)...")

    stay_ids      = icu["stay_id"].tolist()
    grids         = {}
    counts        = {}
    hours_since_d = {}
    masks_vital   = {}
    forecast_masks= {}
    cond_vecs     = {}
    cond_vecs_fc  = {}
    skipped       = 0
    label_counts  = {v: {0:0,1:0,2:0,-1:0} for v in LABELER_VITALS}
    for ck in COMPOSITE_KEYS:
        label_counts[f"comp_{ck}"] = {0:0,1:0,2:0,-1:0}

    for sid in tqdm(stay_ids, desc="  assembling"):
        sid = int(sid)

        grid  = np.full((total_hours, N_FEATURES), np.nan,  dtype=np.float32)
        count = np.zeros((total_hours, N_FEATURES),          dtype=np.float32)

        for feat_name, hour_dict in raw.get(sid, {}).items():
            fi = FEATURE_INDEX.get(feat_name)
            if fi is None: continue
            if feat_name in HOURLY_FEATURES:
                for hbin, val in hour_dict.items():
                    if hbin < total_hours:
                        grid[hbin, fi]  = val
                        count[hbin, fi] += 1
            else:
                day_last = {}
                for hbin, val in hour_dict.items():
                    day = hbin // 24
                    if day not in day_last or hbin > day_last[day][0]:
                        day_last[day] = (hbin, val)
                for _, (hbin, val) in day_last.items():
                    if hbin < total_hours:
                        grid[hbin, fi]  = val
                        count[hbin, fi] = 1

        for feat_name, events in vaso.get(sid, {}).items():
            fi = FEATURE_INDEX.get(feat_name)
            if fi is None: continue
            bin_last = {}
            for (hbin, val) in events:
                if hbin < total_hours:
                    bin_last[hbin] = val
            for hbin, val in bin_last.items():
                grid[hbin, fi]  = val
                count[hbin, fi] = 1
        for feat_name in vaso.get(sid, {}):
            fi = FEATURE_INDEX.get(feat_name)
            if fi is None: continue
            for h in range(total_hours):
                if np.isnan(grid[h, fi]):
                    grid[h, fi] = 0.0

        hs = np.full((total_hours, N_FEATURES), HOURS_SINCE_MAX, dtype=np.float32)
        for fi in range(N_FEATURES):
            last_obs_h = None
            for h in range(total_hours):
                if count[h, fi] > 0:
                    last_obs_h = h
                if last_obs_h is not None:
                    hs[h, fi] = float(h - last_obs_h)

        future_raw  = grid[past_hours:, :]
        vital_mask  = np.zeros((future_hours, N_TARGETS),   dtype=np.int32)
        fc_mask     = np.zeros((future_hours, N_FORECAST),  dtype=np.int32)
        for j, fi in enumerate(TARGET_INDICES):
            vital_mask[:, j] = (~np.isnan(future_raw[:, fi])).astype(np.int32)
        for j, fi in enumerate(FORECAST_INDICES):
            fc_mask[:, j]    = (~np.isnan(future_raw[:, fi])).astype(np.int32)

        if vital_mask.sum() == 0:
            skipped += 1
            continue

        post_window = {}
        for v in LABELER_VITALS:
            fi = FEATURE_INDEX.get(v)
            if fi is None:
                post_window[v] = None; continue
            arr = future_raw[:, fi].copy()
            post_window[v] = None if np.all(np.isnan(arr)) else arr

        label_result = label_patient_window(
            post_window, variables=LABELER_VITALS, return_features=True)

        cond_vec = build_condition_vector(
            label_result, post_window, feat_min_map, feat_max_map)

        cond_vec_fc = _build_forecast_cond_vec(
            label_result, post_window, feat_min_map, feat_max_map)

        cond_vecs[sid]     = cond_vec
        cond_vecs_fc[sid]  = cond_vec_fc
        masks_vital[sid]   = vital_mask
        forecast_masks[sid]= fc_mask

        for v in LABELER_VITALS:
            label_counts[v][label_result["labels"].get(v, -1)] += 1
        for ck in COMPOSITE_KEYS:
            label_counts[f"comp_{ck}"][label_result["composite"].get(ck, -1)] += 1

        df = pd.DataFrame(grid)
        df = df.ffill(axis=0).bfill(axis=0).fillna(0.0)
        grids[sid]         = df.values.astype(np.float32)
        counts[sid]        = count
        hours_since_d[sid] = hs

        if static_map is not None and sid in static_map:
            lab_raw = {}
            lab_hs  = {}
            for feat_name in LAB_STATIC_FEATURES:
                fi = FEATURE_INDEX.get(feat_name)
                if fi is None: continue
                col_count = count[:past_hours, fi]
                obs_hours = np.where(col_count > 0)[0]
                if len(obs_hours) > 0:
                    last_h   = int(obs_hours[-1])
                    lab_raw[feat_name] = float(grid[last_h, fi])
                    lab_hs[feat_name]  = float(past_hours - 1 - last_h)
            static_map[sid]["_lab_raw"]         = lab_raw
            static_map[sid]["_lab_hours_since"] = lab_hs

    print("\n  ── Future label distribution ──────────────────────────────")
    n_total = max(len(grids), 1)
    for var, cnts in label_counts.items():
        s = "  ".join(f"{LABEL_NAMES.get(k,str(k))}={v}({100*v/n_total:.0f}%)"
                      for k, v in cnts.items())
        print(f"    {var:<25}: {s}")
    return grids, counts, hours_since_d, masks_vital, forecast_masks, cond_vecs, cond_vecs_fc, label_counts


def compute_norm_stats(grids: dict, train_sids: set):
    print("\n[7a] Computing normalization stats from train set...")
    train_arrays = [grids[sid] for sid in train_sids if sid in grids]
    stacked = np.stack(train_arrays, axis=0).reshape(-1, N_FEATURES)
    feat_min = np.zeros(N_FEATURES, dtype=np.float32)
    feat_max = np.ones(N_FEATURES,  dtype=np.float32)
    for fi in tqdm(range(N_FEATURES), desc="  stats"):
        col = stacked[:, fi]
        p5, p95 = np.percentile(col, 5), np.percentile(col, 95)
        col_w = np.clip(col, p5, p95)
        feat_min[fi] = col_w.min()
        feat_max[fi] = col_w.max()
    return feat_min, feat_max


def apply_norm(grids: dict, feat_min, feat_max) -> dict:
    print("\n[7b] Applying normalization...")
    denom = feat_max - feat_min
    denom[denom < 1e-8] = 1.0
    normed = {}
    for sid, grid in tqdm(grids.items(), desc="  normalizing"):
        g = np.clip(grid, feat_min, feat_max)
        normed[sid] = ((g - feat_min) / denom).astype(np.float32)
    return normed


def make_feat_minmax_map(feat_min, feat_max):
    fmin = {v: float(feat_min[FEATURE_INDEX[v]]) for v in LABELER_VITALS if v in FEATURE_INDEX}
    fmax = {v: float(feat_max[FEATURE_INDEX[v]]) for v in LABELER_VITALS if v in FEATURE_INDEX}
    return fmin, fmax


LAB_STATIC_FEATURES = [
    "Creatinine", "BUN", "WBC", "Hemoglobin", "Platelets",
    "LacticAcid", "Sodium_serum", "Potassium_serum", "Glucose_serum", "INR",
]
N_LAB_STATIC = len(LAB_STATIC_FEATURES)
LAB_HOURS_SINCE_MAX = 99.0


def normalize_static(static_map: dict, train_sids: set):
    """

      [age_norm, height_norm, weight_norm]              (3)
      + lab × (last_value_norm, observed, hours_since)  (10 × 3 = 30)

    static_categoric:

    """
    print("\n[8] Normalizing static features (+ lab last values)...")
    numeric_keys = ["age", "height_cm", "weight_kg"]
    train_vals = {k: [] for k in numeric_keys}
    lab_vals   = {k: [] for k in LAB_STATIC_FEATURES}

    for sid in train_sids:
        sv = static_map.get(sid, {})
        for k in numeric_keys:
            v = sv.get(k, np.nan)
            if not np.isnan(v): train_vals[k].append(v)
        lab_raw = sv.get("_lab_raw", {})
        for k in LAB_STATIC_FEATURES:
            if k in lab_raw:
                lab_vals[k].append(lab_raw[k])

    medians  = {k: float(np.median(v)) if v else 0.0 for k,v in train_vals.items()}
    minimums = {k: float(np.min(v))    if v else 0.0 for k,v in train_vals.items()}
    maximums = {k: float(np.max(v))    if v else 1.0 for k,v in train_vals.items()}

    lab_p5   = {k: float(np.percentile(v,  5)) if v else 0.0 for k,v in lab_vals.items()}
    lab_p95  = {k: float(np.percentile(v, 95)) if v else 1.0 for k,v in lab_vals.items()}

    normed_static = {}
    for sid, sv in static_map.items():
        num_vec = []
        for k in numeric_keys:
            v = sv.get(k, np.nan)
            if np.isnan(v): v = medians[k]
            d = maximums[k] - minimums[k]
            d = d if d > 1e-8 else 1.0
            num_vec.append(float(np.clip((v - minimums[k]) / d, 0.0, 1.0)))

        lab_raw = sv.get("_lab_raw", {})
        lab_hs  = sv.get("_lab_hours_since", {})
        for k in LAB_STATIC_FEATURES:
            if k in lab_raw:
                raw_v = lab_raw[k]
                lo = lab_p5[k]; hi = lab_p95[k]
                d  = hi - lo if hi - lo > 1e-8 else 1.0
                val_norm  = float(np.clip((raw_v - lo) / d, 0.0, 1.0))
                observed  = 1.0
                hs_norm   = float(np.clip(lab_hs.get(k, 0.0) / LAB_HOURS_SINCE_MAX, 0.0, 1.0))
            else:
                val_norm  = 0.5
                observed  = 0.0
                hs_norm   = 1.0
            num_vec.extend([val_norm, observed, hs_norm])

        normed_static[sid] = {
            "static_numeric"  : np.array(num_vec, dtype=np.float32),
            "static_categoric": np.array([sv["gender"], sv["icu_type"]], dtype=np.int32),
        }
    n_static = 3 + N_LAB_STATIC * 3
    print(f"  static_numeric dim: {n_static} = 3 + {N_LAB_STATIC}×3(lab)")
    return normed_static, medians, minimums, maximums


def assemble_dataset(
    icu:            pd.DataFrame,
    normed_grids:   dict,
    normed_static:  dict,
    masks_vital:    dict,
    forecast_masks: dict,
    cond_vecs:      dict,
    cond_vecs_fc:   dict,
    counts:         dict,
    hours_since_d:  dict,
    past_hours:     int,
    future_hours:   int,
) -> list:
    print("\n[9] Assembling final dataset...")
    stay_to_label = dict(zip(icu["stay_id"].astype(int), icu["died_in_icu"]))
    stay_to_split = dict(zip(icu["stay_id"].astype(int), icu["split"]))

    dataset = []
    skipped = 0
    for sid in tqdm(normed_grids.keys(), desc="  assembling"):
        sid = int(sid)
        if sid not in stay_to_label or sid not in normed_static: skipped+=1; continue
        if sid not in masks_vital   or sid not in cond_vecs:     skipped+=1; continue

        grid        = normed_grids[sid]
        mask        = masks_vital[sid]
        fc_mask     = forecast_masks.get(sid, np.zeros((future_hours, N_FORECAST), dtype=np.int32))
        stat        = normed_static[sid]
        cond_vec    = cond_vecs[sid]
        cond_vec_fc = cond_vecs_fc.get(sid, np.zeros(COND_VEC_FORECAST_DIM, dtype=np.float32))
        cnt         = counts.get(sid, np.zeros_like(grid))
        hs          = hours_since_d.get(sid, np.full_like(grid, HOURS_SINCE_MAX))

        hs_norm  = np.clip(hs / HOURS_SINCE_MAX, 0.0, 1.0).astype(np.float32)
        cnt_clip = np.clip(cnt, 0, 10).astype(np.float32) / 10.0
        rich_ts  = np.concatenate([grid, cnt_clip, hs_norm], axis=1)

        future_ts_full = np.tile(cond_vec,    (future_hours, 1))
        future_ts_fc   = np.tile(cond_vec_fc, (future_hours, 1))

        dataset.append({
            "stay_id"              : sid,
            "historical_ts_numeric": grid[:past_hours, :],
            "future_ts_numeric"    : future_ts_full,
            "targets"              : grid[:, TARGET_INDICES],
            "targets_masks"        : mask,
            "static_numeric"       : stat["static_numeric"],
            "static_categoric"     : stat["static_categoric"],
            "cond_vec"             : cond_vec,
            "label"                : int(stay_to_label[sid]),
            "split"                : stay_to_split.get(sid, "unknown"),
            "historical_ts_rich"   : rich_ts[:past_hours, :],
            "obs_count"            : cnt[:past_hours, :],
            "hours_since"          : hs[:past_hours, :],
            "forecasting_targets"  : grid[:, FORECAST_INDICES],
            "forecast_masks"       : fc_mask,
            "future_ts_forecast"   : future_ts_fc,
            "cond_vec_forecast"    : cond_vec_fc,
        })

    print(f"  → {len(dataset):,} assembled (skipped: {skipped:,})")
    return dataset


def main():
    args = get_args()

    PAST_HOURS   = args.past_hours
    FUTURE_HOURS = args.future_hours
    TOTAL_HOURS  = PAST_HOURS + FUTURE_HOURS

    print("="*60)
    print(f"  PAST={PAST_HOURS}h  FUTURE={FUTURE_HOURS}h  TOTAL={TOTAL_HOURS}h")
    print(f"  OUTPUT: {args.out_dir}")
    if args.convert_duett:
        duett_dir = args.duett_dir or (args.out_dir + "_duett")
        print(f"  DUETT:  {duett_dir}")
    print("="*60)

    os.makedirs(args.out_dir, exist_ok=True)
    if args.convert_duett:
        os.makedirs(duett_dir, exist_ok=True)

    icu = build_cohort(args)
    train_sids = set(icu[icu["split"]=="train"]["stay_id"].astype(int))

    static_map = build_static_features(icu, TOTAL_HOURS)

    raw = build_raw_grids(icu, TOTAL_HOURS)

    vaso = build_vasopressor_grids(icu, TOTAL_HOURS)

    print("\n[5-6 Pre-pass] Building raw grids for norm stats...")
    (grids_pre, counts, hours_since_d,
     masks_vital, forecast_masks,
     cond_vecs_pre, cond_vecs_fc_pre, _) = build_hourly_arrays(
        icu, raw, vaso, PAST_HOURS, FUTURE_HOURS,
        feat_min_map=None, feat_max_map=None,
        static_map=static_map,
    )

    feat_min, feat_max = compute_norm_stats(grids_pre, train_sids)
    feat_min_map, feat_max_map = make_feat_minmax_map(feat_min, feat_max)

    print("\n[5e Re-pass] Recomputing condition vectors with proper last_val_norm...")
    cond_vecs    = {}
    cond_vecs_fc = {}
    for sid_key in tqdm(cond_vecs_pre.keys(), desc="  re-labeling"):
        sid        = int(sid_key)
        grid_raw   = grids_pre[sid]
        future_raw = grid_raw[PAST_HOURS:, :]
        post_window = {}
        for v in LABELER_VITALS:
            fi = FEATURE_INDEX.get(v)
            if fi is None: post_window[v] = None; continue
            arr = future_raw[:, fi].copy()
            post_window[v] = arr if not np.all(arr == 0) else None
        label_result = label_patient_window(
            post_window, variables=LABELER_VITALS, return_features=True)
        cond_vecs[sid]    = build_condition_vector(
            label_result, post_window, feat_min_map, feat_max_map)
        cond_vecs_fc[sid] = _build_forecast_cond_vec(
            label_result, post_window, feat_min_map, feat_max_map)

    normed_grids = apply_norm(grids_pre, feat_min, feat_max)
    del grids_pre, raw, vaso

    normed_static, stat_medians, stat_mins, stat_maxs = normalize_static(
        static_map, train_sids)
    del static_map

    dataset = assemble_dataset(
        icu, normed_grids, normed_static,
        masks_vital, forecast_masks,
        cond_vecs, cond_vecs_fc,
        counts, hours_since_d,
        PAST_HOURS, FUTURE_HOURS,
    )
    del normed_grids, normed_static

    train_set = [d for d in dataset if d["split"]=="train"]
    val_set   = [d for d in dataset if d["split"]=="val"]
    test_set  = [d for d in dataset if d["split"]=="test"]
    pos = sum(d["label"] for d in train_set)
    neg = len(train_set) - pos
    print(f"\n  train={len(train_set):,} | val={len(val_set):,} | test={len(test_set):,}")
    print(f"  Mortality rate: {pos/len(train_set)*100:.2f}%")
    print(f"  Suggested BCE pos_weight: {neg/max(pos,1):.2f}")

    metadata = {
        "feature_order"    : FEATURE_ORDER,
        "feature_index"    : FEATURE_INDEX,
        "n_features"       : N_FEATURES,
        "target_vitals"    : TARGET_VITALS,
        "target_indices"   : TARGET_INDICES,
        "n_targets"        : N_TARGETS,
        "forecast_vitals"  : FORECAST_VITALS,
        "forecast_indices" : FORECAST_INDICES,
        "n_forecast"       : N_FORECAST,
        "n_rich_features"  : N_RICH_FEATURES,
        "hours_since_max"  : HOURS_SINCE_MAX,
        "cond_vec_forecast_dim": COND_VEC_FORECAST_DIM,
        "total_hours"      : TOTAL_HOURS,
        "past_hours"       : PAST_HOURS,
        "future_hours"     : FUTURE_HOURS,
        "cond_vec_dim"     : COND_VEC_DIM,
        "labeler_vitals"   : LABELER_VITALS,
        "composite_keys"   : COMPOSITE_KEYS,
        "feat_min"         : feat_min,
        "feat_max"         : feat_max,
        "feat_min_map"     : feat_min_map,
        "feat_max_map"     : feat_max_map,
        "stat_medians"     : stat_medians,
        "stat_mins"        : stat_mins,
        "stat_maxs"        : stat_maxs,
        "data_props": {
            "num_historical_numeric"             : N_FEATURES,
            "num_historical_rich"                : N_RICH_FEATURES,
            "num_historical_categorical"         : 0,
            "historical_categorical_cardinalities": [],
            "num_static_numeric"                 : 3,
            "num_static_categorical"             : 2,
            "static_categorical_cardinalities"   : [2, 5],
            "num_future_numeric"                 : COND_VEC_DIM,
            "num_future_numeric_forecast"        : COND_VEC_FORECAST_DIM,
            "num_future_categorical"             : 0,
            "future_categorical_cardinalities"   : [],
            "num_feature_predicted"              : N_TARGETS,
            "num_feature_forecast"               : N_FORECAST,
        },
        "n_train"    : len(train_set),
        "n_pos_train": pos,
        "n_neg_train": neg,
    }

    print("\n[10] Saving...")
    for name, data in [("train", train_set), ("val", val_set),
                       ("test", test_set), ("metadata", metadata)]:
        path = os.path.join(args.out_dir, f"{name}.pkl")
        save_pickle(data, path, duett_compat=False)

    if args.convert_duett:
        for name, data in [("train", train_set), ("val", val_set),
                           ("test", test_set), ("metadata", metadata)]:
            path = os.path.join(duett_dir, f"{name}.pkl")
            save_pickle(data, path, duett_compat=True)

    if args.convert_duett:
    if args.convert_duett:
        print(f"  DuETT       : --data_dir {duett_dir}")


if __name__ == "__main__":
    main()