"""
utils/metrics.py
================


"""

import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity


def evaluate_sofa_retrieval(
    embeddings: np.ndarray,
    sofa:       np.ndarray,
    k_list:     list = [5, 10, 20],
    threshold:  float = 0.5,
    verbose:    bool = False,
) -> dict:
    """


    """
    N = len(sofa)
    sim_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(sim_matrix, -np.inf)

    results = {}

    for k in k_list:
        k_eff = min(k, N - 1)
        top_k_idx = np.argsort(sim_matrix, axis=1)[:, -k_eff:]

        sofa_prec_list = []
        sofa_ndcg_list = []
        sofa_diff_list = []

        for i in range(N):
            retrieved  = top_k_idx[i]
            sofa_diffs = np.abs(sofa[i] - sofa[retrieved])

            sofa_prec_list.append((sofa_diffs < threshold).mean())

            sofa_diff_list.append(sofa_diffs.mean())

            relevance = np.exp(-sofa_diffs ** 2)
            ranks     = np.arange(1, k_eff + 1)
            dcg       = (relevance / np.log2(ranks + 1)).sum()
            idcg      = (np.sort(relevance)[::-1] / np.log2(ranks + 1)).sum()
            sofa_ndcg_list.append(dcg / (idcg + 1e-8))

        results[f"SOFA_P@{k}"]    = float(np.mean(sofa_prec_list))
        results[f"SOFA_NDCG@{k}"] = float(np.mean(sofa_ndcg_list))
        results[f"SOFA_diff@{k}"] = float(np.mean(sofa_diff_list))

    triu_i, triu_j = np.triu_indices(N, k=1)
    if len(triu_i) > 5000:
        sel = np.random.choice(len(triu_i), 5000, replace=False)
        triu_i, triu_j = triu_i[sel], triu_j[sel]

    emb_dist  = 1 - cosine_similarity(embeddings)[triu_i, triu_j]
    sofa_diff = np.abs(sofa[triu_i] - sofa[triu_j])

    if emb_dist.std() > 1e-8 and sofa_diff.std() > 1e-8:
        results["SOFA_dist_corr"] = float(np.corrcoef(emb_dist, sofa_diff)[0, 1])
    else:
        results["SOFA_dist_corr"] = 0.0

    if verbose:
        print(f"\n  [SOFA-based Retrieval Evaluation]  (threshold={threshold})")
        print(f"  {'K':>4} | {'SOFA_P@K':>10} | {'SOFA_NDCG@K':>12} | {'Mean_SOFA_diff':>14}")
        print(f"  {'─'*48}")
        for k in k_list:
            k_eff = min(k, N - 1)
            print(f"  {k_eff:>4} | {results[f'SOFA_P@{k}']:>10.4f} | "
                  f"{results[f'SOFA_NDCG@{k}']:>12.4f} | "
                  f"{results[f'SOFA_diff@{k}']:>14.4f}")
        print(f"\n  SOFA-distance correlation : {results['SOFA_dist_corr']:.4f}")

    return results


def evaluate_separation(
    embeddings: np.ndarray,
    labels:     np.ndarray,
    sofa:       np.ndarray,
    verbose:    bool = False,
) -> float:
    """


    """
    N = len(labels)
    if N < 10:
        return 0.0

    sim_matrix = cosine_similarity(embeddings)

    triu_i, triu_j = np.triu_indices(N, k=1)
    if len(triu_i) > 5000:
        sel = np.random.choice(len(triu_i), 5000, replace=False)
        triu_i, triu_j = triu_i[sel], triu_j[sel]

    emb_dist  = 1 - sim_matrix[triu_i, triu_j]
    sofa_diff = np.abs(sofa[triu_i] - sofa[triu_j])

    if emb_dist.std() > 1e-8 and sofa_diff.std() > 1e-8:
        sofa_corr      = float(np.corrcoef(emb_dist, sofa_diff)[0, 1])
        sofa_corr_norm = (sofa_corr + 1) / 2
    else:
        sofa_corr, sofa_corr_norm = 0.0, 0.5

    sofa_bins = np.digitize(sofa, bins=np.percentile(sofa, [33, 66]))
    intra_sims, inter_sims = [], []
    for i in range(N):
        for j in range(i + 1, min(i + 50, N)):
            s = sim_matrix[i, j]
            if sofa_bins[i] == sofa_bins[j]:
                intra_sims.append(s)
            else:
                inter_sims.append(s)

    intra_mean = float(np.mean(intra_sims)) if intra_sims else 0.0
    inter_mean = float(np.mean(inter_sims)) if inter_sims else 0.0
    ratio_raw  = (intra_mean - inter_mean) / (abs(intra_mean) + abs(inter_mean) + 1e-8)
    ratio_norm = (ratio_raw + 1) / 2

    if len(np.unique(labels)) > 1:
        sil      = silhouette_score(embeddings, labels, metric="cosine")
        sil_norm = (sil + 1) / 2
    else:
        sil, sil_norm = 0.0, 0.5

    combined = 0.5 * sofa_corr_norm + 0.3 * ratio_norm + 0.2 * sil_norm

    if verbose:
        print(f"\n  [Embedding Separation Analysis]")
        print(f"    SOFA-distance correlation   : {sofa_corr:+.4f}  (norm: {sofa_corr_norm:.4f})")
        print(f"    SOFA intra/inter sim ratio  : {ratio_raw:+.4f}  (norm: {ratio_norm:.4f})")
        print(f"    Silhouette (mortality, ref) : {sil:+.4f}  (norm: {sil_norm:.4f})")
        print(f"    ─────────────────────────────────────")
        print(f"    Combined Score              : {combined:.4f}")

    return combined


def evaluate_mortality_retrieval(
    embeddings: np.ndarray,
    labels:     np.ndarray,
    k_list:     list = [5, 10, 20],
    verbose:    bool = False,
) -> dict:
    """

    """
    N = len(labels)
    sim_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(sim_matrix, -np.inf)

    results = {}
    for k in k_list:
        k_eff = min(k, N - 1)
        top_k_idx = np.argsort(sim_matrix, axis=1)[:, -k_eff:]

        prec_all, prec_mort, prec_surv = [], [], []
        for i in range(N):
            retrieved = labels[top_k_idx[i]]
            prec      = (retrieved == labels[i]).mean()
            prec_all.append(prec)
            (prec_mort if labels[i] == 1 else prec_surv).append(prec)

        results[f"P@{k}"]      = float(np.mean(prec_all))
        results[f"P@{k}_mort"] = float(np.mean(prec_mort)) if prec_mort else 0.0
        results[f"P@{k}_surv"] = float(np.mean(prec_surv)) if prec_surv else 0.0

    if verbose:
        mort_rate = labels.mean()
        print(f"\n  [Mortality Retrieval — Reference Only]")
        print(f"  {'K':>4} | {'Overall':>8} | {'Mortality':>10} | {'Survival':>9}")
        print(f"  {'─'*40}")
        for k in k_list:
            k_eff = min(k, N - 1)
            print(f"  {k_eff:>4} | {results[f'P@{k}']:>8.4f} | "
                  f"{results[f'P@{k}_mort']:>10.4f} | "
                  f"{results[f'P@{k}_surv']:>9.4f}")
        print(f"\n  Baseline (random): {mort_rate:.4f}")
        print(f"  Improvement      : {results.get('P@10', 0) / max(mort_rate, 1e-8):.2f}x")

    return results


def evaluate_ablation(
    embeddings_dict: dict,
    sofa:            np.ndarray,
    labels:          np.ndarray,
    k:               int   = 10,
    threshold:       float = 0.5,
    verbose:         bool  = True,
) -> dict:
    """

      {
        "Stage1"      : emb_stage1,
        "SOFA-SupCon" : emb_stage2,
      }
    """
    N = len(sofa)
    results = {}

    rand_sofa_prec, rand_mort_prec = [], []
    for i in range(N):
        others = np.delete(np.arange(N), i)
        rand_k = np.random.choice(others, min(k, N - 1), replace=False)
        rand_sofa_prec.append((np.abs(sofa[i] - sofa[rand_k]) < threshold).mean())
        rand_mort_prec.append((labels[rand_k] == labels[i]).mean())

    results["Random"] = {
        f"SOFA_P@{k}":    float(np.mean(rand_sofa_prec)),
        f"SOFA_NDCG@{k}": 0.0,
        f"SOFA_diff@{k}": float(np.mean(np.abs(sofa[:, None] - sofa[None, :]))),
        f"P@{k}":         float(np.mean(rand_mort_prec)),
        f"P@{k}_mort":    float(labels.mean()),
    }

    for name, emb in embeddings_dict.items():
        if emb is None:
            continue
        sofa_res = evaluate_sofa_retrieval(emb, sofa, k_list=[k], threshold=threshold)
        mort_res = evaluate_mortality_retrieval(emb, labels, k_list=[k])
        results[name] = {
            f"SOFA_P@{k}":    sofa_res[f"SOFA_P@{k}"],
            f"SOFA_NDCG@{k}": sofa_res[f"SOFA_NDCG@{k}"],
            f"SOFA_diff@{k}": sofa_res[f"SOFA_diff@{k}"],
            f"P@{k}":         mort_res[f"P@{k}"],
            f"P@{k}_mort":    mort_res[f"P@{k}_mort"],
        }

    if verbose:
        print(f"\n  [Ablation: Retrieval Method Comparison @ K={k}]")
        print(f"  {'Method':<16} | {'SOFA_P@K':>10} | {'SOFA_NDCG@K':>12} | "
              f"{'Mort_P@K':>10} | {'SOFA_diff':>10}")
        print(f"  {'─'*68}")
        for name, res in results.items():
            print(f"  {name:<16} | {res.get(f'SOFA_P@{k}', 0):>10.4f} | "
                  f"{res.get(f'SOFA_NDCG@{k}', 0):>12.4f} | "
                  f"{res.get(f'P@{k}', 0):>10.4f} | "
                  f"{res.get(f'SOFA_diff@{k}', 0):>10.4f}")

    return results


def compute_similarity_distribution(
    embeddings: np.ndarray,
    sofa:       np.ndarray,
    labels:     np.ndarray,
    threshold:  float = 0.5,
    n_sample:   int   = 5000,
) -> dict:
    """

    """
    N = len(sofa)
    sim_matrix = cosine_similarity(embeddings)
    triu_i, triu_j = np.triu_indices(N, k=1)

    if len(triu_i) > n_sample:
        sel = np.random.choice(len(triu_i), n_sample, replace=False)
        triu_i, triu_j = triu_i[sel], triu_j[sel]

    sims      = sim_matrix[triu_i, triu_j]
    sofa_diff = np.abs(sofa[triu_i] - sofa[triu_j])
    same_mort = (labels[triu_i] == labels[triu_j])

    sofa_pos = sims[sofa_diff < threshold]
    sofa_neg = sims[sofa_diff >= threshold]
    mort_pos = sims[same_mort]
    mort_neg = sims[~same_mort]

    return {
        "sofa_pos_mean": float(sofa_pos.mean()) if len(sofa_pos) else 0.0,
        "sofa_pos_std" : float(sofa_pos.std())  if len(sofa_pos) else 0.0,
        "sofa_neg_mean": float(sofa_neg.mean()) if len(sofa_neg) else 0.0,
        "sofa_neg_std" : float(sofa_neg.std())  if len(sofa_neg) else 0.0,
        "sofa_gap"     : float(sofa_pos.mean() - sofa_neg.mean())
                         if len(sofa_pos) and len(sofa_neg) else 0.0,
        "mort_pos_mean": float(mort_pos.mean()) if len(mort_pos) else 0.0,
        "mort_neg_mean": float(mort_neg.mean()) if len(mort_neg) else 0.0,
        "mort_gap"     : float(mort_pos.mean() - mort_neg.mean())
                         if len(mort_pos) and len(mort_neg) else 0.0,
    }

def evaluate_k_sensitivity(
    embeddings: np.ndarray,
    labels:     np.ndarray,
    sofa:       np.ndarray,
    k_list:     list  = [5, 10, 20, 30],
    verbose:    bool  = True,
) -> dict:
    """

    """
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    N = len(labels)
    sim_mat = cos_sim(embeddings)
    np.fill_diagonal(sim_mat, -np.inf)

    results       = {}
    label_priors  = {}
    sim_means_d   = {}

    for k in k_list:
        k_eff   = min(k, N - 1)
        top_idx = np.argsort(sim_mat, axis=1)[:, -k_eff:]
        sims    = np.sort(sim_mat, axis=1)[:, -k_eff:]
        w       = sims / (sims.sum(axis=1, keepdims=True) + 1e-8)
        lp      = (w * labels[top_idx]).sum(axis=1)
        dr      = labels[top_idx].mean(axis=1)

        label_priors[k]  = lp
        sim_means_d[k]   = sims.mean(axis=1)

        if len(np.unique(labels)) > 1:
            lp_auroc = float(roc_auc_score(labels, lp))
        else:
            lp_auroc = 0.0

        results[f"lp_auroc_k{k}"]        = lp_auroc
        results[f"dead_ratio_k{k}"]      = float(dr.mean())
        results[f"dead_ratio_dead_k{k}"] = float(dr[labels == 1].mean()) \
                                           if (labels == 1).any() else 0.0
        results[f"sim_mean_k{k}"]        = float(sim_means_d[k].mean())

    lp_auroces = [results[f"lp_auroc_k{k}"] for k in k_list]
    n_pairs    = len(k_list) - 1
    n_mono     = sum(1 for i in range(n_pairs)
                     if lp_auroces[i + 1] > lp_auroces[i])
    results["lp_auroc_monotone_ratio"] = n_mono / max(n_pairs, 1)

    if 5 in k_list and 30 in k_list:
        results["sim_spread_k5_k30"] = float(
            sim_means_d[5].mean() - sim_means_d[30].mean())
    elif len(k_list) >= 2:
        results["sim_spread_k5_k30"] = float(
            sim_means_d[k_list[0]].mean() - sim_means_d[k_list[-1]].mean())

    if 5 in label_priors and 30 in label_priors:
        lp5, lp30 = label_priors[5], label_priors[30]
        if lp5.std() > 1e-8 and lp30.std() > 1e-8:
            results["lp_corr_k5_k30"] = float(np.corrcoef(lp5, lp30)[0, 1])
        else:
            results["lp_corr_k5_k30"] = 1.0

    if verbose:
        print(f"\n  [k-Sensitivity Evaluation]")
        print(f"  {'k':>5} | {'LP_AUROC':>10} | {'dead_ratio':>11} | "
              f"{'dead(dead query)':>17} | {'sim_mean':>10}")
        print(f"  {'─'*62}")
        for k in k_list:
            print(f"  {k:>5} | "
                  f"{results[f'lp_auroc_k{k}']:>10.4f} | "
                  f"{results[f'dead_ratio_k{k}']:>11.4f} | "
                  f"{results[f'dead_ratio_dead_k{k}']:>17.4f} | "
                  f"{results[f'sim_mean_k{k}']:>10.4f}")
        print(f"\n  LP AUROC monotone ratio : "
        print(f"  sim spread (k5→k30)     : "
        if "lp_corr_k5_k30" in results:
            print(f"  LP corr (k5, k30)       : "

    return results