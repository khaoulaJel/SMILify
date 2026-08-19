"""A1 v2: adjudicating the v1 headline finding (fitted betas show no genus clustering while a
raw-mesh baseline does) before it goes to Fabian as anything more than a lead. Four checks,
all CPU, all cheap:

 1. Significance: n=4 genera (sizes 2,3,2,4) is too small to trust either ratio at face value.
    Permutation test both ratios against a null built from random equal-size groupings of the
    same 60 labeled specimens.
 2. Size confound: v1's "non-articulated baseline" was NOT actually scale-invariant despite the
    label -- it included absolute bbox extents and a volume proxy alongside the two aspect
    ratios, i.e. it could win purely because genera differ in raw size, not body-plan shape.
    Build a genuinely scale-invariant baseline (aspect ratios only) for a fair comparison, and
    separately test whether centroid-size-residualized betas (allometry-corrected) tell a
    different story than raw betas.
 3. Registration-artifact covariate: correlate the beta-space PCs with per-specimen fit quality
    (chamfer_l2, fscore@0.01). If PC1/PC2 track registration quality rather than anatomy, a null
    clustering result doesn't implicate the model's biological expressiveness at all.
 4. Confirm the 20/80 unlabeled specimens are DROPPED (not lumped) from every statistic below,
    including any normalization/standardization step (v1 z-scored the raw baseline using all 80
    specimens' mean/std, which is a data-leakage channel v1 didn't check) -- fixed here to
    standardize on labeled-only statistics.
"""
import csv
import json
import os
import re
import sys

import numpy as np
from pytorch3d.io import load_obj
from scipy.stats import pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
D1DIR = os.path.join(REPO, "diagnostics", "d1_n50_evidence")
HOLDOUT_MESH_DIR = os.path.join(D1DIR, "all_ants_clean_holdout")
OUT = os.path.join(HERE, "out")
SEEDS = [0, 1, 2]
N_PERM = 20000


def parse_genus(label):
    stem = label[:-4] if label.endswith(".obj") else label
    if re.fullmatch(r"\d+", stem):
        return None
    return stem.split("-")[0]


def pca(X, n_components=2):
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:n_components]
    proj = Xc @ comps.T
    var_ratio = (S ** 2) / (S ** 2).sum()
    return proj, var_ratio[:n_components]


def load_all():
    per_seed_betas, per_seed_verts, per_seed_metrics = {}, {}, {}
    labels_ref = None
    for s in SEEDS:
        d = np.load(os.path.join(D1DIR, "runs_holdout", f"seed{s}", "d1", "Stage_3_deform_fine.npz"),
                    allow_pickle=True)
        labels = [str(x) for x in d["labels"]]
        if labels_ref is None:
            labels_ref = labels
        else:
            assert labels == labels_ref
        per_seed_betas[s] = d["betas"]
        per_seed_verts[s] = d["verts"]

        mcsv = os.path.join(D1DIR, "runs_holdout", f"seed{s}", "d1", "metrics.csv")
        by_mesh = {}
        with open(mcsv) as fh:
            for row in csv.DictReader(fh):
                by_mesh[row["mesh"]] = dict(chamfer_l2=float(row["chamfer_l2"]),
                                             fscore_01=float(row["fscore@0.01"]))
        per_seed_metrics[s] = by_mesh
    return labels_ref, per_seed_betas, per_seed_verts, per_seed_metrics


def centroid_size(verts):
    """verts: (N, V, 3) -> (N,) RMS distance from each specimen's own centroid."""
    c = verts.mean(axis=1, keepdims=True)
    d2 = ((verts - c) ** 2).sum(axis=-1)
    return np.sqrt(d2.mean(axis=1))


def residualize(y, x):
    """OLS-residualize y (N,) against x (N,) [with intercept]; returns residuals (N,)."""
    X = np.stack([np.ones_like(x), x], axis=1)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    return y - pred


def genus_groups(labels, genus_by_label):
    g2idx = {}
    for i, lb in enumerate(labels):
        g = genus_by_label[lb]
        if g is not None:
            g2idx.setdefault(g, []).append(i)
    return {g: idxs for g, idxs in g2idx.items() if len(idxs) >= 2}


def ratio_stat(feat_mat, group_idx_lists, labeled_idx, rng, n_random_pairs=2000):
    """feat_mat: (N_labeled_subset, D) already indexed to labeled-only rows, with `labeled_idx`
    giving the ORIGINAL indices those rows correspond to (for bookkeeping only)."""
    def dist(i, j):
        return float(np.linalg.norm(feat_mat[i] - feat_mat[j]))
    within = [dist(i, j) for idxs in group_idx_lists for a, i in enumerate(idxs) for j in idxs[a + 1:]]
    n = feat_mat.shape[0]
    rp = []
    for _ in range(n_random_pairs):
        i, j = rng.choice(n, size=2, replace=False)
        rp.append(dist(i, j))
    return float(np.mean(within)), float(np.mean(rp)), within


def permutation_test(feat_mat, group_sizes, observed_ratio, rng, n_perm=N_PERM):
    """Null: random equal-size groupings (same sizes as observed) drawn from the same specimen
    pool, group membership shuffled, everything else identical. One-sided: P(null_ratio <=
    observed_ratio) -- is the observed clustering tighter than chance groupings of the same size?
    """
    n = feat_mat.shape[0]

    def dist(i, j):
        return float(np.linalg.norm(feat_mat[i] - feat_mat[j]))

    null_ratios = []
    for _ in range(n_perm):
        perm = rng.permutation(n)
        pos = 0
        groups = []
        for sz in group_sizes:
            groups.append(perm[pos:pos + sz])
            pos += sz
        within = [dist(i, j) for idxs in groups for a, i in enumerate(idxs) for j in idxs[a + 1:]]
        remaining = perm[pos:]
        if len(remaining) >= 2:
            rp_idx = rng.choice(remaining, size=(min(200, len(remaining) * (len(remaining) - 1) // 2), 2))
            rp = [dist(i, j) for i, j in rp_idx if i != j]
        else:
            rp = [dist(i, j) for i in range(n) for j in range(n) if i != j][:200]
        if not rp or not within:
            continue
        null_ratios.append(np.mean(within) / np.mean(rp))
    null_ratios = np.array(null_ratios)
    p_value = float((null_ratios <= observed_ratio).mean())
    return p_value, null_ratios


def main():
    rng = np.random.default_rng(0)
    labels, per_seed_betas, per_seed_verts, per_seed_metrics = load_all()
    genus_by_label = {lb: parse_genus(lb) for lb in labels}
    labeled_idx = [i for i, lb in enumerate(labels) if genus_by_label[lb] is not None]
    unlabeled_idx = [i for i, lb in enumerate(labels) if genus_by_label[lb] is None]
    print(f"{len(labels)} total, {len(labeled_idx)} labeled (used below), {len(unlabeled_idx)} "
          f"unlabeled numeric-ID specimens -- CONFIRMED DROPPED from every statistic below (not "
          f"lumped, not used in any normalization).")

    labels_labeled = [labels[i] for i in labeled_idx]
    genus_groups_full = genus_groups(labels, genus_by_label)  # indices into full 80-array
    # remap group indices onto the labeled-only 0..59 index space
    idx_map = {orig: new for new, orig in enumerate(labeled_idx)}
    genus_groups_remapped = {g: [idx_map[i] for i in idxs] for g, idxs in genus_groups_full.items()}
    group_sizes = [len(v) for v in genus_groups_remapped.values()]
    print(f"multi-specimen genera: { {g: len(v) for g, v in genus_groups_remapped.items()} }")

    # ---------------- betas, labeled-only ----------------
    betas_stack = np.stack([per_seed_betas[s] for s in SEEDS], axis=0)  # (3, 80, 25)
    mean_betas_all = betas_stack.mean(axis=0)  # (80, 25)
    mean_betas_labeled = mean_betas_all[labeled_idx]  # (60, 25), LABELED-ONLY, no normalization needed (raw units)

    # ---------------- centroid size (fitted mesh), labeled-only ----------------
    verts_stack = np.stack([per_seed_verts[s] for s in SEEDS], axis=0)  # (3, 80, V, 3)
    cs_per_seed = np.stack([centroid_size(verts_stack[s]) for s in range(len(SEEDS))], axis=0)  # (3, 80)
    cs_mean = cs_per_seed.mean(axis=0)  # (80,)
    cs_labeled = cs_mean[labeled_idx]

    # correlation of raw betas' PC1 with centroid size (labeled-only, descriptive)
    proj_betas_labeled, var_ratio_betas = pca(mean_betas_labeled, n_components=2)
    r_cs_pc1, p_cs_pc1 = pearsonr(proj_betas_labeled[:, 0], cs_labeled)
    print(f"\n=== Size entanglement check ===")
    print(f"corr(beta PC1, centroid size) on labeled specimens: r={r_cs_pc1:.3f}, p={p_cs_pc1:.4f}")

    # residualize each beta dim against centroid size (labeled-only fit)
    betas_resid = np.stack([residualize(mean_betas_labeled[:, k], cs_labeled)
                             for k in range(mean_betas_labeled.shape[1])], axis=1)  # (60, 25)

    # ---------------- raw mesh descriptors, labeled-only, two variants ----------------
    def raw_descriptors_for(lbls):
        feats_scaleinv, feats_sizeinclusive, used = [], [], []
        for lb in lbls:
            p = os.path.join(HOLDOUT_MESH_DIR, lb)
            if not os.path.isfile(p):
                continue
            v, _, _ = load_obj(p, load_textures=False)
            v = v.numpy()
            extent = np.sort(v.max(0) - v.min(0))[::-1]
            aspect1 = extent[0] / max(extent[1], 1e-9)
            aspect2 = extent[1] / max(extent[2], 1e-9)
            volume_proxy = float(np.prod(extent))
            feats_scaleinv.append([aspect1, aspect2])
            feats_sizeinclusive.append([extent[0], extent[1], extent[2], aspect1, aspect2, volume_proxy])
            used.append(lb)
        return used, np.array(feats_scaleinv), np.array(feats_sizeinclusive)

    raw_labels_used, raw_scaleinv, raw_sizeinclusive = raw_descriptors_for(labels_labeled)
    assert raw_labels_used == labels_labeled, "label order must match labeled_idx order"

    def zscore_labeled(X):
        mu, sd = X.mean(0), X.std(0).clip(min=1e-9)
        return (X - mu) / sd

    raw_scaleinv_z = zscore_labeled(raw_scaleinv)
    raw_sizeinclusive_z = zscore_labeled(raw_sizeinclusive)

    # ---------------- ratio + permutation test, for each feature space ----------------
    def evaluate(name, feat_mat):
        obs_within, obs_random, within_list = ratio_stat(feat_mat, list(genus_groups_remapped.values()),
                                                           labeled_idx, rng)
        obs_ratio = obs_within / obs_random
        p_value, null_ratios = permutation_test(feat_mat, group_sizes, obs_ratio, rng)
        print(f"{name:<32} ratio={obs_ratio:.3f}  permutation p={p_value:.4f}  "
              f"(null mean={null_ratios.mean():.3f}, null std={null_ratios.std():.3f}, n_perm={len(null_ratios)})")
        return dict(ratio=obs_ratio, within_mean=obs_within, random_mean=obs_random,
                    permutation_p=p_value, null_mean=float(null_ratios.mean()), null_std=float(null_ratios.std()))

    print(f"\n=== Ratio + permutation significance (n_perm={N_PERM}, one-sided: null_ratio <= observed) ===")
    results = {}
    results["betas_raw"] = evaluate("fitted betas (raw, v1 comparison)", mean_betas_labeled)
    results["betas_size_residualized"] = evaluate("fitted betas (centroid-size residualized)", betas_resid)
    results["raw_baseline_size_inclusive_v1"] = evaluate("raw baseline, SIZE-INCLUSIVE (v1's actual metric)", raw_sizeinclusive_z)
    results["raw_baseline_scale_invariant"] = evaluate("raw baseline, TRULY scale-invariant (aspect ratios only)", raw_scaleinv_z)

    # ---------------- registration-artifact covariate ----------------
    print(f"\n=== Registration-artifact covariate check ===")
    chamfer_mean = np.array([np.mean([per_seed_metrics[s][labels[i]]["chamfer_l2"] for s in SEEDS])
                              for i in labeled_idx])
    fscore_mean = np.array([np.mean([per_seed_metrics[s][labels[i]]["fscore_01"] for s in SEEDS])
                             for i in labeled_idx])
    for pc_idx in [0, 1]:
        r_ch, p_ch = pearsonr(proj_betas_labeled[:, pc_idx], chamfer_mean)
        r_fs, p_fs = pearsonr(proj_betas_labeled[:, pc_idx], fscore_mean)
        print(f"beta PC{pc_idx+1} vs chamfer_l2: r={r_ch:+.3f} (p={p_ch:.4f})   "
              f"vs fscore@0.01: r={r_fs:+.3f} (p={p_fs:.4f})")
    reg_artifact = dict(
        pc1_vs_chamfer=dict(r=float(pearsonr(proj_betas_labeled[:, 0], chamfer_mean)[0]),
                             p=float(pearsonr(proj_betas_labeled[:, 0], chamfer_mean)[1])),
        pc2_vs_chamfer=dict(r=float(pearsonr(proj_betas_labeled[:, 1], chamfer_mean)[0]),
                             p=float(pearsonr(proj_betas_labeled[:, 1], chamfer_mean)[1])),
        pc1_vs_fscore=dict(r=float(pearsonr(proj_betas_labeled[:, 0], fscore_mean)[0]),
                            p=float(pearsonr(proj_betas_labeled[:, 0], fscore_mean)[1])),
        pc2_vs_fscore=dict(r=float(pearsonr(proj_betas_labeled[:, 1], fscore_mean)[0]),
                            p=float(pearsonr(proj_betas_labeled[:, 1], fscore_mean)[1])),
    )

    print(f"\n=== SUMMARY ===")
    for k, v in results.items():
        sig = "SIGNIFICANT (p<0.05)" if v["permutation_p"] < 0.05 else "not distinguishable from chance"
        print(f"  {k:<32} ratio={v['ratio']:.3f}  p={v['permutation_p']:.4f}  -> {sig}")

    report = dict(
        n_labeled=len(labeled_idx), n_unlabeled=len(unlabeled_idx),
        group_sizes=group_sizes,
        size_entanglement=dict(r_pc1_vs_centroid_size=float(r_cs_pc1), p=float(p_cs_pc1)),
        ratio_results=results,
        registration_artifact_check=reg_artifact,
    )
    with open(os.path.join(OUT, "a1_v2_adjudication.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {os.path.join(OUT, 'a1_v2_adjudication.json')}")


if __name__ == "__main__":
    main()
