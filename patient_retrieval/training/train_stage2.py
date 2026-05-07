import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.dataset import load_datasets
from models.encoder import PatientEncoder
from models.losses import SoftSupConLoss, MortalityClassificationLoss, IntraSOFAOutcomeLoss, UniformityLoss
from utils.metrics import (
    evaluate_sofa_retrieval,
    evaluate_separation,
    evaluate_mortality_retrieval,
)


def get_args():
    p = argparse.ArgumentParser(
        description="")
    p.add_argument("--data_dir",      type=str,   default="/path/to/output")
    p.add_argument("--stage1_ckpt",   type=str,   default="/path/to/output")
    p.add_argument("--out_dir",       type=str,   default="./checkpoints/stage2_v6")

    p.add_argument("--epochs",        type=int,   default=60)
    p.add_argument("--warmup_epochs", type=int,   default=10)
    p.add_argument("--eval_interval", type=int,   default=5,
                   help="")

    p.add_argument("--batch_size",    type=int,   default=128)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--warmup_lr",     type=float, default=1e-3)
    p.add_argument("--weight_decay",  type=float, default=1e-4)

    p.add_argument("--hidden_dim",    type=int,   default=256)
    p.add_argument("--n_layers",      type=int,   default=6)
    p.add_argument("--embed_dim",     type=int,   default=256)

    p.add_argument("--alpha_bce_p1",  type=float, default=0.5,
                   help="")
    p.add_argument("--alpha_bce_p2",  type=float, default=0.1,
                   help="")
    p.add_argument("--alpha_sofa",    type=float, default=0.6,
                   help=""
                        "alpha_outcome + alpha_uniform = 1 - alpha_sofa")
    p.add_argument("--alpha_outcome", type=float, default=0.2,
                   help="")
    p.add_argument("--alpha_uniform", type=float, default=0.2,
                   help=""

    p.add_argument("--temperature",        type=float, default=0.2,
                   help="")
    p.add_argument("--sofa_sigma",         type=float, default=0.3)
    p.add_argument("--outcome_temperature",type=float, default=0.1,
                   help="")
    p.add_argument("--outcome_n_bins",     type=int,   default=5,
                   help="")
    p.add_argument("--uniform_t",          type=float, default=4.0,
                   help=""

    p.add_argument("--patience",      type=int,   default=10)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--device",        type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def set_phase(model, phase: int):
    if phase == 1:
        for name, param in model.named_parameters():
            param.requires_grad = (name.startswith("static_encoder")
                                   or name.startswith("final_proj"))
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  [Phase 1] ts_encoder frozen. Trainable: {trainable:,} params")
    else:
        for param in model.parameters():
            param.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  [Phase 2] All params unfrozen. Trainable: {trainable:,} params")


def train_epoch(
    model, loader, optimizer,
    supcon_loss, outcome_loss, bce_loss, uniform_loss,
    alpha_bce, alpha_sofa, alpha_outcome, alpha_uniform,
    device,
):
    model.train()
    total_loss = total_sofa = total_outcome = total_uniform = total_bce = 0.0
    n = 0

    for batch in loader:
        ts       = batch["ts_rich"].to(device)
        stat_num = batch["static_num"].to(device)
        stat_cat = batch["static_cat"].to(device)
        labels   = batch["label"].to(device)
        sofa     = batch["sofa_proxy"].to(device)

        emb = model(ts, static_num=stat_num, static_cat=stat_cat)

        l_sofa    = supcon_loss(emb, labels, sofa)
        l_outcome = outcome_loss(emb, labels, sofa)
        l_uniform = uniform_loss(emb)
        l_bce     = bce_loss(emb, labels)

        l_contrastive = (alpha_sofa    * l_sofa
                         + alpha_outcome * l_outcome
                         + alpha_uniform * l_uniform)
        loss = (1 - alpha_bce) * l_contrastive + alpha_bce * l_bce

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        B = ts.size(0)
        total_loss    += loss.item()      * B
        total_sofa    += l_sofa.item()    * B
        total_outcome += l_outcome.item() * B
        total_uniform += l_uniform.item() * B
        total_bce     += l_bce.item()     * B
        n += B

    return {
        "loss"   : total_loss    / n,
        "sofa"   : total_sofa    / n,
        "outcome": total_outcome / n,
        "uniform": total_uniform / n,
        "bce"    : total_bce     / n,
    }


@torch.no_grad()
def extract_embeddings(model, loader, device) -> dict:
    model.eval()
    all_emb, all_label, all_sofa, all_sid = [], [], [], []
    for batch in loader:
        ts  = batch["ts_rich"].to(device)
        sn  = batch["static_num"].to(device)
        sc  = batch["static_cat"].to(device)
        emb = model(ts, static_num=sn, static_cat=sc)
        all_emb.append(emb.cpu())
        all_label.append(batch["label"])
        all_sofa.append(batch["sofa_proxy"])
        all_sid.append(batch["stay_id"])
    return {
        "embeddings": torch.cat(all_emb),
        "labels"    : torch.cat(all_label),
        "sofa"      : torch.cat(all_sofa),
        "stay_ids"  : torch.cat([
            torch.tensor(s) if not isinstance(s, torch.Tensor) else s
            for s in all_sid
        ]),
    }


def evaluate_k_sensitivity(
    emb:    np.ndarray,
    labels: np.ndarray,
    sofa:   np.ndarray,
    k_list: list = [5, 10, 20, 30],
    verbose: bool = True,
) -> dict:
    """

      label_prior AUROC @ k

      dead_ratio @ k

      sim_spread @ k

      feat_cosim(k, k=5)
    """
    N = len(labels)
    sim_mat = cosine_similarity(emb)
    np.fill_diagonal(sim_mat, -np.inf)

    results = {}

    label_priors = {}
    dead_ratios  = {}
    sim_means    = {}

    for k in k_list:
        k_eff  = min(k, N - 1)
        top_idx = np.argsort(sim_mat, axis=1)[:, -k_eff:]
        sims    = np.sort(sim_mat, axis=1)[:, -k_eff:]

        w = sims / (sims.sum(axis=1, keepdims=True) + 1e-8)

        lp = (w * labels[top_idx]).sum(axis=1)
        label_priors[k] = lp

        dr = labels[top_idx].mean(axis=1)
        dead_ratios[k] = dr

        sim_means[k] = sims.mean(axis=1)

        if len(np.unique(labels)) > 1:
            lp_auroc = float(roc_auc_score(labels, lp))
        else:
            lp_auroc = 0.0

        results[f"lp_auroc_k{k}"]     = lp_auroc
        results[f"dead_ratio_k{k}"]   = float(dr.mean())
        results[f"dead_ratio_dead_k{k}"] = float(dr[labels == 1].mean()) \
                                           if (labels == 1).any() else 0.0
        results[f"sim_mean_k{k}"]     = float(sim_means[k].mean())

    lp_auroces = [results[f"lp_auroc_k{k}"] for k in k_list]
    n_pairs    = len(k_list) - 1
    n_mono_inc = sum(1 for i in range(n_pairs)
                     if lp_auroces[i + 1] > lp_auroces[i])
    results["lp_auroc_monotone_ratio"] = n_mono_inc / max(n_pairs, 1)

    if 5 in k_list and 30 in k_list:
        results["sim_spread_k5_k30"] = float(
            sim_means[5].mean() - sim_means[30].mean())
    elif len(k_list) >= 2:
        results["sim_spread_k5_k30"] = float(
            sim_means[k_list[0]].mean() - sim_means[k_list[-1]].mean())

    if 5 in label_priors and 30 in label_priors:
        lp5  = label_priors[5]
        lp30 = label_priors[30]
        if lp5.std() > 1e-8 and lp30.std() > 1e-8:
            results["lp_corr_k5_k30"] = float(
                np.corrcoef(lp5, lp30)[0, 1])
        else:
            results["lp_corr_k5_k30"] = 1.0

    if verbose:
        print(f"\n  [k-Sensitivity Evaluation]")
        print(f"  {'k':>5} | {'LP_AUROC':>10} | {'dead_ratio':>11} | "
              f"{'dead_ratio(dead)':>17} | {'sim_mean':>10}")
        print(f"  {'─'*62}")
        for k in k_list:
            print(f"  {k:>5} | "
                  f"{results[f'lp_auroc_k{k}']:>10.4f} | "
                  f"{results[f'dead_ratio_k{k}']:>11.4f} | "
                  f"{results[f'dead_ratio_dead_k{k}']:>17.4f} | "
                  f"{results[f'sim_mean_k{k}']:>10.4f}")
        print(f"\n  LP AUROC monotone ratio  : "
              f"{results['lp_auroc_monotone_ratio']:.2f}  "
        print(f"  sim spread (k5→k30)      : "
              f"{results.get('sim_spread_k5_k30', 0):.6f}  "
        if "lp_corr_k5_k30" in results:
            print(f"  LP corr (k5, k30)        : "
                  f"{results['lp_corr_k5_k30']:.4f}  "

    return results


def _compute_sim_spread(emb_np: np.ndarray,
                        k_lo: int = 5, k_hi: int = 30) -> float:
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    N    = len(emb_np)
    s    = cos_sim(emb_np)
    np.fill_diagonal(s, -np.inf)
    s_lo = np.sort(s, axis=1)[:, -min(k_lo, N - 1):]
    s_hi = np.sort(s, axis=1)[:, -min(k_hi, N - 1):]
    return float(s_lo.mean() - s_hi.mean())


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    phase2_epochs = args.epochs - args.warmup_epochs

    total = args.alpha_sofa + args.alpha_outcome + args.alpha_uniform
    alpha_sofa    = args.alpha_sofa    / total
    alpha_outcome = args.alpha_outcome / total
    alpha_uniform = args.alpha_uniform / total

    print(f"\n{'='*60}")
    print(f"  Stage 2: SOFA SupCon + IntraSOFA Outcome + Uniformity")
    print(f"  Data: {args.data_dir}")
    print(f"  Stage1 ckpt: {args.stage1_ckpt or 'random init'}")
    print(f"  Phase 1: {args.warmup_epochs}ep  "
          f"(static warm-up, lr={args.warmup_lr})")
    print(f"  Phase 2: {phase2_epochs}ep  "
          f"(full fine-tune, lr={args.lr})")
    print(f"\n  Loss weights (Phase 2, normalized):")
    print(f"    alpha_bce     = {args.alpha_bce_p2}")
    print(f"    alpha_sofa    = {alpha_sofa:.3f}")
    print(f"    alpha_outcome = {alpha_outcome:.3f}")
    print(f"    alpha_uniform = {alpha_uniform:.3f}")
    print(f"    → L = {1-args.alpha_bce_p2:.1f}×[{alpha_sofa:.2f}×L_sofa "
          f"+ {alpha_outcome:.2f}×L_outcome "
          f"+ {alpha_uniform:.2f}×L_uniform] "
          f"+ {args.alpha_bce_p2:.1f}×L_bce")
    print(f"\n  uniform_t={args.uniform_t}  "
          f"outcome_temp={args.outcome_temperature}  "
          f"n_bins={args.outcome_n_bins}")
    print(f"{'='*60}\n")

    train_ds, val_ds, test_ds = load_datasets(args.data_dir, mode="both")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    model = PatientEncoder(
        ts_input_dim       = 183,
        ts_hidden_dim      = args.hidden_dim,
        ts_n_layers        = args.n_layers,
        static_numeric_dim = 33,
        static_out_dim     = 64,
        embed_dim          = args.embed_dim,
        use_static         = True,
    ).to(device)

    if args.stage1_ckpt and os.path.exists(args.stage1_ckpt):
        ckpt  = torch.load(args.stage1_ckpt, map_location=device, weights_only=False)
        state = ckpt["model_state"]
        filtered = {k: v for k, v in state.items()
                    if not k.startswith("static_encoder")
                    and not k.startswith("final_proj")}
        missing, _ = model.load_state_dict(filtered, strict=False)
        print(f"  Loaded Stage 1 weights (ts_encoder + projection).")
        if missing:
            print(f"  New params (random init): {missing}")
    else:
        print("  No Stage 1 checkpoint. Training from random init.")

    supcon_loss  = SoftSupConLoss(
        temperature = args.temperature,
        sofa_sigma  = args.sofa_sigma,
    )
    outcome_loss = IntraSOFAOutcomeLoss(
        temperature = args.outcome_temperature,
        n_bins      = args.outcome_n_bins,
    )
    uniform_loss = UniformityLoss(
        t = args.uniform_t,
    )
    bce_loss = MortalityClassificationLoss(
        embed_dim      = args.embed_dim,
        pos_weight_val = 9.0,
    ).to(device)

    print(f"\n{'─'*60}")
    print(f"  Phase 1: Static Encoder Warm-up ({args.warmup_epochs} epochs)")
    print(f"{'─'*60}")

    set_phase(model, phase=1)
    p1_params = (
        [p for p in model.parameters() if p.requires_grad]
        + list(bce_loss.parameters())
    )
    optimizer_p1 = torch.optim.AdamW(
        p1_params, lr=args.warmup_lr, weight_decay=args.weight_decay)
    scheduler_p1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_p1, T_max=args.warmup_epochs, eta_min=1e-5)

    for epoch in range(1, args.warmup_epochs + 1):
        m = train_epoch(
            model, train_loader, optimizer_p1,
            supcon_loss, outcome_loss, bce_loss, uniform_loss,
            alpha_bce=args.alpha_bce_p1,
            alpha_sofa=1.0, alpha_outcome=0.0, alpha_uniform=0.0,
            device=device,
        )
        scheduler_p1.step()
        lr_now = optimizer_p1.param_groups[0]["lr"]

        if epoch % args.eval_interval == 0 or epoch == 1:
            val_d = extract_embeddings(model, val_loader, device)
            sep   = evaluate_separation(
                val_d["embeddings"].numpy(),
                val_d["labels"].numpy(),
                val_d["sofa"].numpy(),
            )
            sim_spread = _compute_sim_spread(val_d["embeddings"].numpy())
            print(f"  [P1] Ep{epoch:3d}/{args.warmup_epochs} | "
                  f"loss={m['loss']:.4f}  sofa={m['sofa']:.4f}  "
                  f"bce={m['bce']:.4f}  "
                  f"sep={sep:.4f}  spread={sim_spread:.6f}  lr={lr_now:.6f}")
        else:
            print(f"  [P1] Ep{epoch:3d}/{args.warmup_epochs} | "
                  f"loss={m['loss']:.4f}  sofa={m['sofa']:.4f}  "
                  f"bce={m['bce']:.4f}  lr={lr_now:.6f}")

    print(f"\n{'─'*60}")
    print(f"  Phase 2: Full Fine-tuning ({phase2_epochs} epochs)")
    print(f"  L = {1-args.alpha_bce_p2:.1f}×[{args.alpha_sofa:.1f}×L_sofa "
          f"+ {alpha_outcome:.1f}×L_outcome] + {args.alpha_bce_p2:.1f}×L_bce")
    print(f"{'─'*60}")

    set_phase(model, phase=2)
    optimizer_p2 = torch.optim.AdamW(
        list(model.parameters()) + list(bce_loss.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler_p2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_p2, T_max=phase2_epochs, eta_min=1e-6)

    best_val_sep = -float("inf")
    patience_cnt = 0
    history      = []
    K_EVAL       = [5, 10, 20, 30]

    for epoch in range(1, phase2_epochs + 1):
        m = train_epoch(
            model, train_loader, optimizer_p2,
            supcon_loss, outcome_loss, bce_loss, uniform_loss,
            alpha_bce     = args.alpha_bce_p2,
            alpha_sofa    = alpha_sofa,
            alpha_outcome = alpha_outcome,
            alpha_uniform = alpha_uniform,
            device        = device,
        )
        scheduler_p2.step()
        lr_now = optimizer_p2.param_groups[0]["lr"]

        if epoch % args.eval_interval == 0 or epoch == 1:
            val_d   = extract_embeddings(model, val_loader, device)
            emb_np  = val_d["embeddings"].numpy()
            lab_np  = val_d["labels"].numpy()
            sofa_np = val_d["sofa"].numpy()

            sep    = evaluate_separation(emb_np, lab_np, sofa_np)
            spread = _compute_sim_spread(emb_np)
            k_res  = evaluate_k_sensitivity(
                emb_np, lab_np, sofa_np, k_list=K_EVAL, verbose=True)

            print(f"\n  [P2] Ep{epoch:3d}/{phase2_epochs} | "
                  f"loss={m['loss']:.4f}  sofa={m['sofa']:.4f}  "
                  f"outcome={m['outcome']:.4f}  uniform={m['uniform']:.4f}  "
                  f"bce={m['bce']:.4f}")
            print(f"       sep={sep:.4f}  sim_spread={spread:.6f}  "

            row = {**m, "epoch": epoch, "sep_score": sep,
                   "sim_spread": spread, **k_res}
            history.append(row)

            if sep > best_val_sep:
                best_val_sep = sep
                patience_cnt = 0
                ckpt_path = os.path.join(args.out_dir, "best_stage2.pt")
                torch.save({
                    "epoch"       : epoch,
                    "model_state" : model.state_dict(),
                    "bce_state"   : bce_loss.state_dict(),
                    "sep_score"   : sep,
                    "sim_spread"  : spread,
                    "k_results"   : k_res,
                    "args"        : vars(args),
                    "sofa_mean"   : train_ds.sofa_mean,
                    "sofa_std"    : train_ds.sofa_std,
                }, ckpt_path)
                print(f"    → Saved best  sep={sep:.4f}  "
                      f"spread={spread:.6f}  "
                      f"lp_auroc_k10={k_res.get('lp_auroc_k10',0):.4f}  "
                      f"mono={k_res.get('lp_auroc_monotone_ratio',0):.2f}")
            else:
                patience_cnt += 1
                if patience_cnt >= args.patience:
                    print(f"\n  Early stopping at phase 2 epoch {epoch}.")
                    break
        else:
            print(f"  [P2] Ep{epoch:3d}/{phase2_epochs} | "
                  f"loss={m['loss']:.4f}  sofa={m['sofa']:.4f}  "
                  f"outcome={m['outcome']:.4f}  uniform={m['uniform']:.4f}  "
                  f"bce={m['bce']:.4f}  lr={lr_now:.6f}")

    print(f"\n  Loading best checkpoint for test evaluation...")
    best_ckpt = torch.load(
        os.path.join(args.out_dir, "best_stage2.pt"),
        map_location=device, weights_only=False,
    )
    model.load_state_dict(best_ckpt["model_state"])

    test_d  = extract_embeddings(model, test_loader, device)
    emb_np  = test_d["embeddings"].numpy()
    lab_np  = test_d["labels"].numpy()
    sofa_np = test_d["sofa"].numpy()

    print(f"\n  {'='*60}")
    print(f"  [Test Set Evaluation]")
    print(f"  {'='*60}")
    evaluate_separation(emb_np, lab_np, sofa_np, verbose=True)
    evaluate_sofa_retrieval(emb_np, sofa_np, k_list=K_EVAL, verbose=True)
    evaluate_mortality_retrieval(emb_np, lab_np, k_list=K_EVAL, verbose=True)
    evaluate_k_sensitivity(emb_np, lab_np, sofa_np, k_list=K_EVAL, verbose=True)

    emb_path = os.path.join(args.out_dir, "test_embeddings.pt")
    torch.save(test_d, emb_path)
    print(f"\n  Test embeddings saved : {emb_path}")
    print(f"  Best val sep score    : {best_val_sep:.4f}")


if __name__ == "__main__":
    main()