"""A1 v3: three follow-ups the user specified before the v2 adjudication counts as settled.

 1. Supervised test (the field-standard one v2 was missing): leave-one-out 1-NN and
    nearest-centroid classification, genus label as target, evaluated against a permutation
    null -- directly asks "can genus be predicted from shape" rather than "does genus fall out
    of the unsupervised top-2 PCs" (which a real signal on PC5/PC6 could evade entirely, since
    PCA maximizes total variance, not between-group separation, and v2 already showed PC1-2 are
    dominated by size/fit-quality).
 2. Fabian's original explicit ask, not yet tested: per-vertex mesh DEFORMATIONS
    (`deform_verts`), not just the global `betas` shape prior. Betas are a heavily-regularized
    global prior; deform_verts is the free-form residual that absorbs whatever the prior
    couldn't explain -- plausibly where specimen-specific morphological detail ends up.
 3. A robustness check the linear residualization in v2 doesn't cover: restrict to a
    high-fit-quality-only subsample (top-half by fscore@0.01) and rerun the UNRESIDUALIZED ratio
    test on that subset. Catches a nonlinear/heteroscedastic fit-quality relationship that OLS
    residualization would miss.

Reuses v2's data loading and statistical helpers (no duplication of the core logic).
"""
import json
import os
import sys

import numpy as np
from scipy.stats import pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from a1_pca_taxonomy_v2 import (  # noqa: E402
    load_all, parse_genus, centroid_size, residualize, genus_groups, ratio_stat,
    permutation_test, pca, D1DIR, OUT, SEEDS, N_PERM,
)

RNG_SEED = 0


def loo_1nn_accuracy(feat_mat, class_labels):
    """feat_mat: (N, D), class_labels: (N,) ints. Leave-one-out 1-nearest-neighbour accuracy."""
    n = feat_mat.shape[0]
    correct = 0
    for i in range(n):
        dists = np.linalg.norm(feat_mat - feat_mat[i], axis=1)
        dists[i] = np.inf
        pred = class_labels[np.argmin(dists)]
        correct += int(pred == class_labels[i])
    return correct / n


def loo_centroid_accuracy(feat_mat, class_labels):
    """Leave-one-out nearest-CLASS-CENTROID accuracy (centroid excludes the held-out point)."""
    n = feat_mat.shape[0]
    classes = np.unique(class_labels)
    correct = 0
    for i in range(n):
        best_c, best_d = None, np.inf
        for c in classes:
            members = np.where((class_labels == c) & (np.arange(n) != i))[0]
            if members.size == 0:
                continue
            centroid = feat_mat[members].mean(axis=0)
            d = np.linalg.norm(feat_mat[i] - centroid)
            if d < best_d:
                best_d, best_c = d, c
        correct += int(best_c == class_labels[i])
    return correct / n


def classification_permutation_test(feat_mat, class_labels, observed_acc, rng, n_perm=N_PERM, method="1nn"):
    fn = loo_1nn_accuracy if method == "1nn" else loo_centroid_accuracy
    n = feat_mat.shape[0]
    null_accs = np.empty(n_perm)
    for k in range(n_perm):
        perm_labels = rng.permutation(class_labels)
        null_accs[k] = fn(feat_mat, perm_labels)
    p_value = float((null_accs >= observed_acc).mean())
    return p_value, null_accs


def main():
    rng = np.random.default_rng(RNG_SEED)
    labels, per_seed_betas, per_seed_verts, per_seed_metrics = load_all()
    genus_by_label = {lb: parse_genus(lb) for lb in labels}
    labeled_idx = [i for i, lb in enumerate(labels) if genus_by_label[lb] is not None]
    idx_map = {orig: new for new, orig in enumerate(labeled_idx)}
    gg_full = genus_groups(labels, genus_by_label)
    gg_remapped = {g: [idx_map[i] for i in idxs] for g, idxs in gg_full.items()}
    group_sizes = [len(v) for v in gg_remapped.values()]
    print(f"multi-specimen genera: { {g: len(v) for g, v in gg_remapped.items()} }")

    # class-labeled subset: ONLY the 11 specimens across 4 multi-member genera (singleton-genus
    # specimens can't be classified in a leave-one-out task -- no other same-genus example exists)
    class_idx_in_labeled = []  # indices into the 60-labeled array
    class_labels_str = []
    for g, idxs in gg_remapped.items():
        for i in idxs:
            class_idx_in_labeled.append(i)
            class_labels_str.append(g)
    class_names = sorted(set(class_labels_str))
    class_to_int = {g: k for k, g in enumerate(class_names)}
    class_labels = np.array([class_to_int[g] for g in class_labels_str])
    print(f"classification subset: {len(class_idx_in_labeled)} specimens, {len(class_names)} classes "
          f"({dict(zip(*np.unique(class_labels_str, return_counts=True)))})")

    betas_stack = np.stack([per_seed_betas[s] for s in SEEDS], axis=0)
    mean_betas_all = betas_stack.mean(axis=0)
    mean_betas_labeled = mean_betas_all[labeled_idx]

    verts_stack = np.stack([per_seed_verts[s] for s in SEEDS], axis=0)
    cs_mean = np.stack([centroid_size(verts_stack[s]) for s in range(len(SEEDS))], axis=0).mean(axis=0)
    cs_labeled = cs_mean[labeled_idx]

    chamfer_mean_all = {}
    fscore_mean_all = {}
    for lb in labels:
        chamfer_mean_all[lb] = np.mean([per_seed_metrics[s][lb]["chamfer_l2"] for s in SEEDS])
        fscore_mean_all[lb] = np.mean([per_seed_metrics[s][lb]["fscore_01"] for s in SEEDS])
    chamfer_labeled = np.array([chamfer_mean_all[labels[i]] for i in labeled_idx])
    fscore_labeled = np.array([fscore_mean_all[labels[i]] for i in labeled_idx])

    X_size_chamfer = np.stack([np.ones_like(cs_labeled), cs_labeled, chamfer_labeled], axis=1)
    betas_resid_joint = np.zeros_like(mean_betas_labeled)
    for k in range(mean_betas_labeled.shape[1]):
        coef, *_ = np.linalg.lstsq(X_size_chamfer, mean_betas_labeled[:, k], rcond=None)
        betas_resid_joint[:, k] = mean_betas_labeled[:, k] - X_size_chamfer @ coef

    # ============================================================
    # PART 1: leave-one-out classification (supervised test)
    # ============================================================
    print(f"\n{'='*70}\nPART 1: Leave-one-out classification (supervised, field-standard test)\n{'='*70}")

    CLASSIF_N_PERM = 5000  # LOO loops are pure-python; keep this bounded, still ample resolution for p to 3dp

    def eval_classifier(name, feat_full_labeled, use_idx):
        feat = feat_full_labeled[use_idx]
        acc_1nn = loo_1nn_accuracy(feat, class_labels)
        acc_cent = loo_centroid_accuracy(feat, class_labels)
        p_1nn, null_1nn = classification_permutation_test(feat, class_labels, acc_1nn, rng, n_perm=CLASSIF_N_PERM, method="1nn")
        p_cent, null_cent = classification_permutation_test(feat, class_labels, acc_cent, rng, n_perm=CLASSIF_N_PERM, method="centroid")
        print(f"{name:<42} 1NN acc={acc_1nn:.3f} (null mean={null_1nn.mean():.3f}, p={p_1nn:.4f})   "
              f"centroid acc={acc_cent:.3f} (null mean={null_cent.mean():.3f}, p={p_cent:.4f})")
        return dict(acc_1nn=acc_1nn, p_1nn=p_1nn, null_mean_1nn=float(null_1nn.mean()),
                    acc_centroid=acc_cent, p_centroid=p_cent, null_mean_centroid=float(null_cent.mean()))

    classif_results = {}
    classif_results["betas_raw"] = eval_classifier("fitted betas (raw)", mean_betas_labeled, class_idx_in_labeled)
    classif_results["betas_size_chamfer_residualized"] = eval_classifier(
        "fitted betas (size+chamfer residualized)", betas_resid_joint, class_idx_in_labeled)

    # ============================================================
    # PART 2: deformation-space test (Fabian's original ask, not yet run)
    # ============================================================
    print(f"\n{'='*70}\nPART 2: Per-vertex deformation (deform_verts) as shape descriptor\n{'='*70}")
    deform_per_seed = {}
    for s in SEEDS:
        d = np.load(os.path.join(D1DIR, "runs_holdout", f"seed{s}", "d1", "Stage_3_deform_fine.npz"),
                    allow_pickle=True)
        deform_per_seed[s] = d["deform_verts"].reshape(d["deform_verts"].shape[0], -1).astype(np.float64)  # (80, V*3)

    deform_stack = np.stack([deform_per_seed[s] for s in SEEDS], axis=0)  # (3, 80, V*3)
    mean_deform_all = deform_stack.mean(axis=0)  # (80, V*3)
    mean_deform_labeled = mean_deform_all[labeled_idx]  # (60, V*3)

    # PCA basis fit on all 60 labeled specimens (rank-limited to 59 components anyway)
    K = 10
    proj_deform_labeled, var_ratio_deform = pca(mean_deform_labeled, n_components=K)
    print(f"deform PCA: top-{K} explained variance = {var_ratio_deform.sum():.3f} "
          f"(PC1={var_ratio_deform[0]:.3f}, PC2={var_ratio_deform[1]:.3f})")

    # precondition: cross-seed repeatability in this reduced space (project each seed onto the
    # SAME basis used above, i.e. mean-centered by the specimen-mean's own mean)
    mu_deform = mean_deform_labeled.mean(axis=0, keepdims=True)
    # recompute basis explicitly to reuse for per-seed projection
    Xc = mean_deform_labeled - mu_deform
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:K]
    deform_seed_proj = np.stack([
        (deform_stack[s][labeled_idx] - mu_deform) @ comps.T for s in range(len(SEEDS))
    ], axis=0)  # (3, 60, K)
    specimen_mean_proj = deform_seed_proj.mean(axis=0)
    intra_var = ((deform_seed_proj - specimen_mean_proj[None]) ** 2).mean()
    inter_var = specimen_mean_proj.var(axis=0).mean()
    deform_repeat_ratio = float(inter_var / max(intra_var, 1e-12))
    print(f"deform-PCA cross-seed repeatability (inter/intra variance ratio): {deform_repeat_ratio:.2f} "
          f"(betas' was 613 -- compare directly; if much lower, deform is noisier / less repeatable per seed)")

    r_pc1_chamfer, p_pc1_chamfer = pearsonr(proj_deform_labeled[:, 0], chamfer_labeled)
    r_pc1_cs, p_pc1_cs = pearsonr(proj_deform_labeled[:, 0], cs_labeled)
    print(f"deform PC1 vs chamfer_l2: r={r_pc1_chamfer:+.3f} (p={p_pc1_chamfer:.4f})   "
          f"vs centroid_size: r={r_pc1_cs:+.3f} (p={p_pc1_cs:.4f})")

    obs_within, obs_random, _ = ratio_stat(proj_deform_labeled, list(gg_remapped.values()), labeled_idx, rng)
    deform_ratio = obs_within / obs_random
    deform_p, deform_null = permutation_test(proj_deform_labeled, group_sizes, deform_ratio, rng)
    print(f"deform-PCA genus-consistency ratio={deform_ratio:.3f}  permutation p={deform_p:.4f}")

    classif_results["deform_pca10"] = eval_classifier("deform_verts (PCA top-10)", proj_deform_labeled, class_idx_in_labeled)

    # ============================================================
    # PART 3: high-fit-quality-only subsample (nonlinear-robust check)
    # ============================================================
    print(f"\n{'='*70}\nPART 3: High-fit-quality subsample (top-half by fscore@0.01), UNRESIDUALIZED\n{'='*70}")
    median_fscore = np.median(fscore_labeled)
    high_q_mask = fscore_labeled >= median_fscore
    high_q_idx_in_labeled = np.where(high_q_mask)[0]
    print(f"top-half by fscore@0.01: {high_q_mask.sum()}/{len(fscore_labeled)} specimens retained "
          f"(fscore threshold={median_fscore:.3f})")

    gg_high_q = {}
    for g, idxs in gg_remapped.items():
        kept = [i for i in idxs if i in set(high_q_idx_in_labeled.tolist())]
        if len(kept) >= 2:
            gg_high_q[g] = kept
    print(f"genera retaining >=2 specimens after fit-quality filter: "
          f"{ {g: len(v) for g, v in gg_high_q.items()} }")

    if gg_high_q:
        high_q_group_sizes = [len(v) for v in gg_high_q.values()]
        betas_high_q_subset = mean_betas_labeled[list(high_q_idx_in_labeled)]
        # need index remap: ratio_stat expects group indices INTO the passed feat_mat
        hq_pos_map = {orig: new for new, orig in enumerate(high_q_idx_in_labeled.tolist())}
        gg_high_q_remapped = {g: [hq_pos_map[i] for i in idxs] for g, idxs in gg_high_q.items()}
        obs_within_hq, obs_random_hq, _ = ratio_stat(betas_high_q_subset, list(gg_high_q_remapped.values()),
                                                       high_q_idx_in_labeled.tolist(), rng)
        ratio_hq = obs_within_hq / obs_random_hq
        p_hq, null_hq = permutation_test(betas_high_q_subset, high_q_group_sizes, ratio_hq, rng)
        print(f"high-fit-quality-only, UNRESIDUALIZED betas ratio={ratio_hq:.3f}  permutation p={p_hq:.4f} "
              f"(n_perm={len(null_hq)}, group_sizes={high_q_group_sizes})")
        part3_result = dict(n_retained=int(high_q_mask.sum()), genera_with_pairs=gg_high_q,
                             ratio=ratio_hq, permutation_p=p_hq)
    else:
        print("DEGENERATE: no genus retains >=2 specimens after the fit-quality filter -- "
              "this check cannot be run as designed on this dataset at this sample size.")
        part3_result = dict(n_retained=int(high_q_mask.sum()), genera_with_pairs={}, ratio=None, permutation_p=None,
                             note="degenerate -- filter removed all within-genus pairs")

    # ============================================================
    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    print("Classification (supervised, the actual field-standard test):")
    for k, v in classif_results.items():
        sig = []
        if v["p_1nn"] < 0.05:
            sig.append("1NN significant")
        if v["p_centroid"] < 0.05:
            sig.append("centroid significant")
        print(f"  {k:<42} 1NN={v['acc_1nn']:.3f}(p={v['p_1nn']:.3f})  centroid={v['acc_centroid']:.3f}"
              f"(p={v['p_centroid']:.3f})  {' & '.join(sig) if sig else 'not significant'}")
    print(f"\nDeformation-space unsupervised ratio: {deform_ratio:.3f} (p={deform_p:.4f})")
    print(f"Deformation cross-seed repeatability: {deform_repeat_ratio:.2f} (compare to betas' 613)")
    print(f"High-fit-quality-only subsample: {'see above' if gg_high_q else 'DEGENERATE, could not run'}")

    report = dict(
        classification=classif_results,
        deformation_space=dict(
            repeatability_ratio=deform_repeat_ratio,
            pc1_vs_chamfer=dict(r=float(r_pc1_chamfer), p=float(p_pc1_chamfer)),
            pc1_vs_centroid_size=dict(r=float(r_pc1_cs), p=float(p_pc1_cs)),
            genus_ratio=deform_ratio, genus_ratio_p=deform_p,
            explained_variance_top10=var_ratio_deform.tolist(),
        ),
        high_fit_quality_subsample=part3_result,
    )
    with open(os.path.join(OUT, "a1_v3_followups.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {os.path.join(OUT, 'a1_v3_followups.json')}")


if __name__ == "__main__":
    main()
