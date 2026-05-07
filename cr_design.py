import os, json, pickle, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, f1_score,
    precision_score, recall_score, confusion_matrix,
)
from tqdm import tqdm

PAST_HOURS   = 36
FUTURE_HOURS = 12
N_QUANTILES  = 5
N_VARS_FC    = 3
MED_IDX      = N_QUANTILES // 2
D_EMBED      = 256

DIM_HIST     = N_VARS_FC * 4
DIM_PRED_MED = FUTURE_HOURS * N_VARS_FC
DIM_PRED_RNG = FUTURE_HOURS * N_VARS_FC
DIM_STATIC   = 33

DIM_SOLO = DIM_HIST + DIM_PRED_MED + DIM_PRED_RNG + DIM_STATIC

KNN_TEMP_DEFAULT = 0.1

CLINICAL_THRESHOLDS = {
    "HR":       {"low": 0.1, "high": 0.8},
    "RespRate": {"low": 0.1, "high": 0.7},
    "SpO2":     {"low": 0.2, "high": 1.0},
}
VAR_NAMES = ["HR", "RespRate", "SpO2"]

N_TRAJ_STATS_PER_VAR = 6
N_TRAJ_META          = 3
DIM_TRAJ_GROUP       = N_VARS_FC * N_TRAJ_STATS_PER_VAR + N_TRAJ_META
DIM_TRAJ_GROUP_PAD   = 24

DIM_GAP = FUTURE_HOURS * N_VARS_FC

DIM_RETRIEVAL = 1 + DIM_TRAJ_GROUP_PAD * 2 + DIM_GAP * 2

DIM_OPT_C = DIM_SOLO + DIM_RETRIEVAL


def extract_feature_q(d: dict, pred_pq: np.ndarray,
                       z_embed: np.ndarray = None,
                       mode: str = "solo") -> np.ndarray:
    ft      = d["forecasting_targets"].astype(np.float32)
    hist_ts = ft[:PAST_HOURS]
    w = 6
    hist = np.concatenate([
        hist_ts[-1],
        hist_ts.mean(0),
        hist_ts.std(0),
        hist_ts[-w:].mean(0) - hist_ts[:w].mean(0),
    ]).astype(np.float32)
    pred_med = pred_pq[MED_IDX].flatten().astype(np.float32)
    pred_rng = (pred_pq[-1] - pred_pq[0]).flatten().astype(np.float32)
    static   = d["static_numeric"].astype(np.float32)
    return np.concatenate([hist, pred_med, pred_rng, static])


def compute_p_knn(
        labels_ri: np.ndarray,
        sims     : np.ndarray,
        temperature: float = KNN_TEMP_DEFAULT,
) -> tuple:
    dist = 1.0 - sims.astype(np.float64)

    logits = -dist / (temperature + 1e-12)
    logits = logits - logits.max()
    w = np.exp(logits)
    w = w / (w.sum() + 1e-12)

    p_knn     = float(np.dot(w, labels_ri))
    p_uniform = float(labels_ri.mean())

    p_knn_c = float(np.clip(p_knn, 1e-7, 1 - 1e-7))
    entropy  = float(-p_knn_c * np.log(p_knn_c)
                     - (1 - p_knn_c) * np.log(1 - p_knn_c))

    return p_knn, entropy, p_uniform


def _traj_stats(trajs: np.ndarray, sims: np.ndarray) -> np.ndarray:
    if len(trajs) == 0:
        return np.zeros(DIM_TRAJ_GROUP_PAD, np.float32)

    n, H, C = trajs.shape
    stats = []

    for vi, vname in enumerate(VAR_NAMES):
        v = trajs[:, :, vi]
        thr_low  = CLINICAL_THRESHOLDS[vname]["low"]
        thr_high = CLINICAL_THRESHOLDS[vname]["high"]

        flat = v.reshape(-1)
        stats.append(float(flat.mean()))
        stats.append(float(flat.std() + 1e-8))
        stats.append(float(flat.min()))
        stats.append(float(flat.max()))

        t = np.arange(H, dtype=np.float32)
        slopes = []
        for pi in range(n):
            vp = v[pi]
            if vp.std() < 1e-6:
                slopes.append(0.0)
            else:
                slope = np.polyfit(t, vp, 1)[0]
                slopes.append(float(slope))
        stats.append(float(np.mean(slopes)))

        if vname == "SpO2":
            abnormal = (v < thr_low).mean()
        else:
            abnormal = ((v < thr_low) | (v > thr_high)).mean()
        stats.append(float(abnormal))

    stats.append(0.0)
    stats.append(float(sims.mean()))
    stats.append(float(sims.std() + 1e-8))

    arr = np.array(stats, dtype=np.float32)
    padded = np.zeros(DIM_TRAJ_GROUP_PAD, dtype=np.float32)
    padded[:len(arr)] = arr
    return padded


def compute_retrieval_feature(
        labels_ri    : np.ndarray,
        sims         : np.ndarray,
        neighbor_trajs: np.ndarray,
        pred_med     : np.ndarray,
        temperature  : float = KNN_TEMP_DEFAULT,
) -> np.ndarray:
    k = len(labels_ri)
    H, C = pred_med.shape

    dist   = 1.0 - sims.astype(np.float64)
    logits = -dist / (temperature + 1e-12)
    logits = logits - logits.max()
    w      = np.exp(logits)
    w      = w / (w.sum() + 1e-12)
    w      = w.astype(np.float32)

    p_knn = float(np.dot(w, labels_ri))

    dead_mask  = labels_ri.astype(bool)
    alive_mask = ~dead_mask

    if dead_mask.sum() > 0:
        dead_trajs = neighbor_trajs[dead_mask]
        dead_sims  = sims[dead_mask]
        T_plus     = _traj_stats(dead_trajs, dead_sims)
        T_plus[18] = float(dead_mask.sum()) / k
    else:
        T_plus = np.zeros(DIM_TRAJ_GROUP_PAD, np.float32)

    if alive_mask.sum() > 0:
        alive_trajs = neighbor_trajs[alive_mask]
        alive_sims  = sims[alive_mask]
        T_minus     = _traj_stats(alive_trajs, alive_sims)
        T_minus[18] = float(alive_mask.sum()) / k
    else:
        T_minus = np.zeros(DIM_TRAJ_GROUP_PAD, np.float32)

    if dead_mask.sum() > 0:
        dead_mean  = neighbor_trajs[dead_mask].mean(axis=0)
        gap_dead   = np.abs(pred_med - dead_mean).flatten()
    else:
        gap_dead = np.abs(pred_med).flatten()

    if alive_mask.sum() > 0:
        alive_mean = neighbor_trajs[alive_mask].mean(axis=0)
        gap_alive  = np.abs(pred_med - alive_mean).flatten()
    else:
        gap_alive  = np.abs(pred_med).flatten()

    feat = np.concatenate([
        np.array([p_knn], dtype=np.float32),
        T_plus,
        T_minus,
        gap_dead.astype(np.float32),
        gap_alive.astype(np.float32),
    ])
    assert len(feat) == DIM_RETRIEVAL, f"dim mismatch: {len(feat)} != {DIM_RETRIEVAL}"
    return feat


def interpolate(
        p_model  : float,
        p_knn    : float,
        entropy  : float,
        mode     : str   = "fixed",
        lam      : float = 0.3,
        lam_max  : float = 0.4,
        lam_learned: float = None,
) -> float:
    if mode == "adaptive":
        lam = lam_max * (1.0 - entropy)
    elif mode == "learned" and lam_learned is not None:
        lam = float(lam_learned)

    lam = float(np.clip(lam, 0.0, 1.0))
    return (1.0 - lam) * p_model + lam * p_knn


def gpu_batch_similarity(query_embs: np.ndarray,
                          pool_embs: np.ndarray,
                          device: torch.device,
                          batch_size: int = 2048) -> np.ndarray:
    q = torch.tensor(query_embs, dtype=torch.float32)
    p = torch.tensor(pool_embs,  dtype=torch.float32).to(device)
    q = q / (q.norm(dim=1, keepdim=True) + 1e-8)
    p = p / (p.norm(dim=1, keepdim=True) + 1e-8)
    N = q.shape[0]
    sims_all = np.zeros((N, pool_embs.shape[0]), dtype=np.float32)
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        sims_all[s:e] = (q[s:e].to(device) @ p.T).cpu().numpy()
    return np.clip(sims_all, 0, 1)


def build_retrieval_cache(
        sids, data_map, fc_map, embed_map,
        pool_sids, pool_embs_norm, pool_labels, pool_fc_map,
        pool_sid_to_idx,
        topk, split, device, cache_path,
) -> dict:
    if os.path.exists(cache_path):
        print(f"  [{split}] Cache hit: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"  [{split}] Building Option-C cache (max_k={topk})...")
    q_embs = np.stack([embed_map[sid] for sid in sids])
    print(f"    GPU sim: {len(sids):,} × {len(pool_sids):,}...", end=" ", flush=True)
    sims_full = gpu_batch_similarity(q_embs, pool_embs_norm, device)
    print("done")

    zero_pred = np.zeros((N_QUANTILES, FUTURE_HOURS, N_VARS_FC), np.float32)
    cache = {}

    for qi, sid in enumerate(tqdm(sids, desc=f"  [{split}] cache")):
        d      = data_map[sid]
        pred_q = fc_map.get(sid, zero_pred)
        feat_q = extract_feature_q(d, pred_q)
        pred_med = pred_q[MED_IDX]

        pool_idx = np.arange(len(pool_sids))
        if sid in pool_sid_to_idx:
            pool_idx = pool_idx[pool_idx != pool_sid_to_idx[sid]]

        sims_q    = sims_full[qi, pool_idx]
        k         = min(topk, len(pool_idx))
        top_local = np.argsort(sims_q)[::-1][:k]
        top_idx   = pool_idx[top_local]
        sims      = sims_q[top_local].astype(np.float32)

        neighbor_trajs = np.stack([
            pool_fc_map.get(pool_sids[idx], zero_pred)[MED_IDX]
            for idx in top_idx
        ]).astype(np.float32)

        cache[sid] = {
            "feat_q"         : feat_q,
            "pred_med"       : pred_med.astype(np.float32),
            "neighbor_labels": pool_labels[top_idx].astype(np.float32),
            "neighbor_sims"  : sims,
            "neighbor_trajs" : neighbor_trajs,
            "label"          : float(d["label"]),
        }

    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f, protocol=4)
    print(f"  [{split}] Cache saved → {cache_path}")
    return cache


def build_solo_cache(sids, data_map, fc_map, embed_map, cache_path) -> dict:
    if os.path.exists(cache_path):
        print(f"  Solo cache hit: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    cache = {}
    zero = np.zeros((N_QUANTILES, FUTURE_HOURS, N_VARS_FC), np.float32)
    for sid in tqdm(sids, desc="  solo cache"):
        d = data_map[sid]
        cache[sid] = {
            "feat_q": extract_feature_q(d, fc_map.get(sid, zero)),
            "label" : float(d["label"]),
        }
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f, protocol=4)
    print(f"  Solo cache saved → {cache_path}")
    return cache


class SoloDataset(Dataset):
    def __init__(self, cache: dict):
        self.sids  = list(cache.keys())
        self.cache = cache

    def __len__(self): return len(self.sids)

    def __getitem__(self, idx):
        c = self.cache[self.sids[idx]]
        return torch.tensor(c["feat_q"], dtype=torch.float32), float(c["label"])


class KNNInterpDataset(Dataset):
    def __init__(self, cache: dict, topk: int,
                 knn_temp: float = KNN_TEMP_DEFAULT):
        self.sids     = list(cache.keys())
        self.cache    = cache
        self.topk     = topk
        self.knn_temp = knn_temp

    def __len__(self): return len(self.sids)

    def __getitem__(self, idx):
        c    = self.cache[self.sids[idx]]
        k    = min(self.topk, len(c["neighbor_labels"]))

        labels = c["neighbor_labels"][:k]
        sims   = c["neighbor_sims"][:k]
        trajs  = c["neighbor_trajs"][:k]
        pred_med = c["pred_med"]

        rf = compute_retrieval_feature(
            labels, sims, trajs, pred_med, self.knn_temp)

        feat = np.concatenate([c["feat_q"], rf]).astype(np.float32)
        return torch.tensor(feat, dtype=torch.float32), float(c["label"])


def collate_fn(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.tensor(ys, dtype=torch.float32)


collate_fn_knn = collate_fn


def load_data(path: str) -> dict:
    with open(path, "rb") as f:
        raw = pickle.load(f)
    return {int(d["stay_id"]): d for d in raw}


def load_embeddings(embed_dir: str, split: str) -> tuple:
    pt_path = os.path.join(embed_dir, f"{split}_embeddings.pt")
    if os.path.exists(pt_path):
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        embed_map, label_map = {}, {}
        embeddings = data["embeddings"]
        if hasattr(embeddings, "numpy"): embeddings = embeddings.numpy()
        labels = data["labels"]
        if hasattr(labels, "numpy"): labels = labels.numpy()
        for i, sid in enumerate(data["stay_ids"]):
            sid = int(sid)
            embed_map[sid] = embeddings[i].astype(np.float32)
            label_map[sid] = int(labels[i])
        return embed_map, label_map

    pkl_path = os.path.join(embed_dir, f"{split}_embeddings.pkl")
    with open(pkl_path, "rb") as f:
        emb = pickle.load(f)
    embed_map, label_map = {}, {}
    for i, sid in enumerate(emb["stay_ids"]):
        sid = int(sid)
        embed_map[sid] = emb["embeddings"][i].astype(np.float32)
        label_map[sid] = int(emb["labels"][i])
    return embed_map, label_map


def load_forecasts(path: str) -> dict:
    with open(path, "rb") as f:
        raw = pickle.load(f)
    result = {}
    for sid, v in raw.items():
        sid = int(sid)
        if isinstance(v, dict):
            result[sid] = v["preds"].astype(np.float32)
        else:
            result[sid] = np.asarray(v, dtype=np.float32)
    return result


class ClassifierMLP(nn.Module):
    def __init__(self, d_in: int = DIM_OPT_C,
                 hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, hidden // 4),
            nn.LayerNorm(hidden // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden // 4, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def run_train_epoch(model, loader, optimizer, device, pw):
    model.train()
    total, n = 0.0, 0
    for x, y in tqdm(loader, desc="  train", leave=False):
        x, y = x.to(device), y.to(device)
        loss = F.binary_cross_entropy_with_logits(
            model(x), y, pos_weight=pw.expand_as(y))
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item(); n += 1
    return total / max(n, 1)


@torch.no_grad()
def run_eval(model, loader, device, pw):
    model.eval()
    all_p, all_l, total, n = [], [], 0.0, 0
    for x, y in tqdm(loader, desc="  eval", leave=False):
        x, y = x.to(device), y.to(device)
        logit = model(x)
        total += F.binary_cross_entropy_with_logits(
            logit, y, pos_weight=pw.expand_as(y)).item()
        n += 1
        all_p.extend(torch.sigmoid(logit).cpu().numpy())
        all_l.extend(y.cpu().int().numpy())
    p = np.array(all_p); l = np.array(all_l)
    return total/max(n,1), roc_auc_score(l,p), average_precision_score(l,p), p, l


def find_threshold(preds, labels):
    prec, rec, thrs = precision_recall_curve(labels, preds)
    f1s = 2*prec*rec/(prec+rec+1e-8)
    idx = np.argmax(f1s[:-1])
    return float(thrs[idx]), float(f1s[idx])


def eval_at_thr(preds, labels, thr, split=""):
    b    = (preds >= thr).astype(int)
    f1   = f1_score(labels, b)
    prec = precision_score(labels, b, zero_division=0)
    rec  = recall_score(labels, b, zero_division=0)
    cm   = confusion_matrix(labels, b)
    tn,fp,fn,tp = cm.ravel() if cm.shape==(2,2) else (0,0,0,cm[0,0])
    spec = tn/(tn+fp+1e-8)
    print(f"  [{split}] thr={thr:.4f} | F1={f1:.4f} Prec={prec:.4f} "
          f"Rec={rec:.4f} Spec={spec:.4f} | TP={tp} FP={fp} FN={fn} TN={tn}")
    return {"f1":f1,"precision":prec,"recall":rec,"specificity":spec,
            "tp":int(tp),"fp":int(fp),"fn":int(fn),"tn":int(tn)}


def get_args():
    p = argparse.ArgumentParser(description="CR Option-C: Deceased/Alive Trajectory Split")
    p.add_argument("--data_dir",     default="/path/to/processed_data")
    p.add_argument("--embed_dir",    default="/path/to/embeddings")
    p.add_argument("--forecast_dir", default="/path/to/forecasts")
    p.add_argument("--output_dir",   default="./results_option_c")
    p.add_argument("--cache_dir",    default=None)
    p.add_argument("--no_cache",     action="store_true")

    p.add_argument("--topk_list",  nargs="+", type=int, default=[5, 10, 20, 30])
    p.add_argument("--max_k",      type=int,   default=30)
    p.add_argument("--knn_temp",   type=float, default=0.05)

    p.add_argument("--hidden",        type=int,   default=256)
    p.add_argument("--dropout",       type=float, default=0.1)
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--batch_size",    type=int,   default=512)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--patience",      type=int,   default=10)
    p.add_argument("--warmup_epochs", type=int,   default=3)
    p.add_argument("--num_workers",   type=int,   default=8)
    p.add_argument("--gpu",           default="0")
    p.add_argument("--seed",          type=int,   default=42)
    return p.parse_args()


def train_option_c_model(train_cache, val_cache, args, device):
    print(f"\n{'='*60}")
    print(f"  [Option-C] Training MLP d_in={DIM_OPT_C} (max_k={args.max_k})")
    print(f"  retrieval_feature: {DIM_RETRIEVAL}d")
    print(f"    p_kNN(1) + T^+(24) + T^-(24) + gap^dead(36) + gap^alive(36)")
    print(f"{'='*60}")

    train_ds = KNNInterpDataset(train_cache, args.max_k, args.knn_temp)
    val_ds   = KNNInterpDataset(val_cache,   args.max_k, args.knn_temp)

    pos_frac   = float(np.mean([c["label"] for c in train_cache.values()]))
    pos_weight = (1.0 - pos_frac) / max(pos_frac, 1e-8)
    pw         = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    print(f"  pos_frac={pos_frac:.4f}  pos_weight={pos_weight:.2f}")
    print(f"  train={len(train_ds):,}  val={len(val_ds):,}")

    nw = args.num_workers
    tr_ld = DataLoader(train_ds, args.batch_size, shuffle=True,
                       collate_fn=collate_fn, num_workers=nw, pin_memory=True)
    va_ld = DataLoader(val_ds,   args.batch_size, shuffle=False,
                       collate_fn=collate_fn, num_workers=nw, pin_memory=True)

    model     = ClassifierMLP(DIM_OPT_C, args.hidden, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    best_auroc, pat = 0.0, 0
    ckpt = os.path.join(args.output_dir, "option_c_best.pt")

    for ep in tqdm(range(1, args.epochs + 1), desc="  [Option-C] epochs"):
        if ep <= args.warmup_epochs:
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * ep / args.warmup_epochs

        tr_loss = run_train_epoch(model, tr_ld, optimizer, device, pw)
        _, va_auroc, va_auprc, _, _ = run_eval(model, va_ld, device, pw)

        if ep > args.warmup_epochs:
            scheduler.step()

        print(f"  Ep{ep:3d} | tr={tr_loss:.4f} | "
              f"AUROC={va_auroc:.4f}  AUPRC={va_auprc:.4f}")

        if va_auroc > best_auroc:
            best_auroc, pat = va_auroc, 0
            torch.save(model.state_dict(), ckpt)
            print(f"    ✓ Best (AUROC={best_auroc:.4f})")
        else:
            pat += 1
            if pat >= args.patience:
                print(f"    Early stop @ep{ep}"); break

    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
    print(f"  Best val AUROC: {best_auroc:.4f}")
    return model, pw


def main():
    args      = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    cache_dir = args.cache_dir or os.path.join(args.output_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    device    = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    print(f"\n{'='*60}")
    print(f"  CR Option-C  —  Deceased/Alive Trajectory Split")
    print(f"  Device: {device}  topk: {args.topk_list}  max_k={args.max_k}")
    print(f"  τ={args.knn_temp}  d_in={DIM_OPT_C}  retrieval={DIM_RETRIEVAL}d")
    print(f"{'='*60}\n")

    print("Loading data...")
    splits = {}
    for sp in ("train", "val", "test"):
        dm      = load_data(os.path.join(args.data_dir, f"{sp}.pkl"))
        em, lm  = load_embeddings(args.embed_dir, sp)
        fc_path = os.path.join(args.forecast_dir, f"{sp}_predictions.pkl")
        fm      = load_forecasts(fc_path)
        sids    = [s for s in em if s in dm and s in fm]
        splits[sp] = dict(data_map=dm, embed_map=em, label_map=lm,
                          fc_map=fm, sids=sids)
        print(f"  {sp}: {len(sids):,}  (dead={sum(lm[s] for s in sids):,})")

    tr            = splits["train"]
    pool_embs     = np.stack([tr["embed_map"][s] for s in tr["sids"]])
    pool_embs_nrm = pool_embs / (np.linalg.norm(pool_embs, axis=1, keepdims=True) + 1e-8)
    pool = {
        "sids"      : tr["sids"],
        "embs_norm" : pool_embs_nrm,
        "labels"    : np.array([tr["label_map"][s] for s in tr["sids"]], np.float32),
        "fc_map"    : tr["fc_map"],
        "sid_to_idx": {s: i for i, s in enumerate(tr["sids"])},
    }

    caches = {}
    for sp in ("train", "val", "test"):
        sp_d = splits[sp]
        cp   = os.path.join(cache_dir, f"optc_{sp}_maxk{args.max_k}.pkl")
        if args.no_cache and os.path.exists(cp): os.remove(cp)
        caches[sp] = build_retrieval_cache(
            sids=sp_d["sids"], data_map=sp_d["data_map"],
            fc_map=sp_d["fc_map"], embed_map=sp_d["embed_map"],
            pool_sids=pool["sids"], pool_embs_norm=pool["embs_norm"],
            pool_labels=pool["labels"], pool_fc_map=pool["fc_map"],
            pool_sid_to_idx=pool["sid_to_idx"],
            topk=args.max_k, split=sp,
            device=device, cache_path=cp)

    print(f"\n{'='*60}")
    print(f"  [Solo baseline] Training MLP d_in={DIM_SOLO}")
    print(f"{'='*60}")
    solo_ds_tr = SoloDataset(caches["train"])
    solo_ds_va = SoloDataset(caches["val"])
    solo_ds_te = SoloDataset(caches["test"])

    pos_frac   = float(np.mean([c["label"] for c in caches["train"].values()]))
    pos_weight = (1.0 - pos_frac) / max(pos_frac, 1e-8)
    pw_solo    = torch.tensor([pos_weight], dtype=torch.float32, device=device)

    nw = args.num_workers
    solo_tr_ld = DataLoader(solo_ds_tr, args.batch_size, shuffle=True,
                            collate_fn=collate_fn, num_workers=nw, pin_memory=True)
    solo_va_ld = DataLoader(solo_ds_va, args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=nw, pin_memory=True)
    solo_te_ld = DataLoader(solo_ds_te, args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=nw, pin_memory=True)

    solo_model = ClassifierMLP(DIM_SOLO, args.hidden, args.dropout).to(device)
    solo_opt   = torch.optim.Adam(solo_model.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)
    solo_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        solo_opt, T_max=args.epochs, eta_min=args.lr * 0.01)
    best_solo, pat = 0.0, 0
    ckpt_solo  = os.path.join(args.output_dir, "solo_best.pt")

    for ep in tqdm(range(1, args.epochs + 1), desc="  [solo] epochs"):
        if ep <= args.warmup_epochs:
            for pg in solo_opt.param_groups:
                pg["lr"] = args.lr * ep / args.warmup_epochs
        run_train_epoch(solo_model, solo_tr_ld, solo_opt, device, pw_solo)
        _, va_auroc, _, _, _ = run_eval(solo_model, solo_va_ld, device, pw_solo)
        if ep > args.warmup_epochs: solo_sched.step()
        if va_auroc > best_solo:
            best_solo, pat = va_auroc, 0
            torch.save(solo_model.state_dict(), ckpt_solo)
        else:
            pat += 1
            if pat >= args.patience: break

    solo_model.load_state_dict(
        torch.load(ckpt_solo, map_location=device, weights_only=False))
    _, solo_auroc, solo_auprc, sp_, sl = run_eval(
        solo_model, solo_te_ld, device, pw_solo)
    best_thr, _ = find_threshold(sp_, sl)
    solo_m = eval_at_thr(sp_, sl, best_thr, "Test-solo")
    print(f"\n  [Solo] Test AUROC={solo_auroc:.4f}  AUPRC={solo_auprc:.4f}")

    model_c, pw_c = train_option_c_model(
        caches["train"], caches["val"], args, device)

    all_results = [
        {"tag": "solo", "topk": 0,
         "test": {"auroc": solo_auroc, "auprc": solo_auprc, **solo_m}},
    ]

    for topk in args.topk_list:
        tag = f"optc_k{topk}"
        print(f"\n  [{tag}]  k={topk}")

        va_ds = KNNInterpDataset(caches["val"],  topk, args.knn_temp)
        te_ds = KNNInterpDataset(caches["test"], topk, args.knn_temp)
        va_ld = DataLoader(va_ds, args.batch_size, shuffle=False,
                           collate_fn=collate_fn, num_workers=nw, pin_memory=True)
        te_ld = DataLoader(te_ds, args.batch_size, shuffle=False,
                           collate_fn=collate_fn, num_workers=nw, pin_memory=True)

        _, va_auroc, va_auprc, vp, vl = run_eval(model_c, va_ld, device, pw_c)
        thr, _ = find_threshold(vp, vl)
        va_m   = eval_at_thr(vp, vl, thr, "Val")
        print(f"  Val  AUROC={va_auroc:.4f}  AUPRC={va_auprc:.4f}")

        _, te_auroc, te_auprc, tp_, tl = run_eval(model_c, te_ld, device, pw_c)
        te_m = eval_at_thr(tp_, tl, thr, "Test")
        print(f"  Test AUROC={te_auroc:.4f}  AUPRC={te_auprc:.4f}")

        all_results.append({
            "tag": tag, "topk": topk,
            "val" : {"auroc": va_auroc, "auprc": va_auprc, **va_m},
            "test": {"auroc": te_auroc, "auprc": te_auprc, **te_m},
        })

    print(f"\n{'='*70}")
    print(f"  SWEEP SUMMARY  ({args.output_dir})")
    print(f"{'='*70}")
    hdr = f"  {'tag':<20} {'k':>4} {'Tst AUROC':>10} {'Tst AUPRC':>10} {'F1':>7}"
    print(hdr); print("  " + "-"*(len(hdr)-2))
    for r in all_results:
        print(f"  {r['tag']:<20} {r['topk']:>4} "
              f"{r['test']['auroc']:>10.4f} "
              f"{r['test']['auprc']:>10.4f} "
              f"{r['test'].get('f1',0):>7.4f}")
    print(f"{'='*70}")

    with open(os.path.join(args.output_dir, "sweep_summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Saved → {args.output_dir}/sweep_summary.json")


if __name__ == "__main__":
    main()