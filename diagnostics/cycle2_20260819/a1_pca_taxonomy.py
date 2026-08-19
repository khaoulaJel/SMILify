"""A1 (Track A, CPU-only, no dependency on Track B/GNC work): does fitted SMAL shape space
(betas + per-joint scale) cluster meaningfully by taxonomy? Fabian's original question.

Data source: `diagnostics/d1_n50_evidence/runs_holdout/seed{0,1,2}/d1/Stage_3_deform_fine.npz`
-- REAL specimen fits (not synthetic), 80 unique ALL_ANTS_CLEAN meshes, 3 seeds, production
D1_low_scalecap.yaml recipe (already run for the D1 promotion-bar work, reused here, zero new
fitting jobs). Taxonomy labels parsed from filenames (curated genus-representative corpus, see
diagnostics/d1_n50_evidence/scorecard.md's "Second follow-up" section) -- most files are
`<genus>[-<species>].obj`; 20/80 are numeric IDs (`01.obj`..`20.obj`) with NO taxonomy label
recoverable from any metadata currently on disk (checked: no accompanying csv/json/txt in the
holdout dir or elsewhere in the repo) -- excluded from genus-clustering analysis, retained in the
embedding for visualization only.

Dependency note: sklearn and umap-learn are NOT installed in the `pytorch3d` conda env (checked
directly, not assumed) -- environment is authoritative per CLAUDE.md, not modified without
asking. This script uses plain PCA (numpy SVD, no new dependency) instead of UMAP. If UMAP is
wanted, that's a one-line ask, not implemented here.

Two things this script checks, in order (probe before interpreting):
 1. Cross-seed repeatability: is the beta signal for a given specimen tighter across the 3 fitting
    seeds than the spread across DIFFERENT specimens? If not, "clusters by taxonomy" is not yet a
    meaningful question -- the embedding would be fitting noise, not signal.
 2. Genus-consistency: for the handful of genera with >=2 specimens (real replication, not
    curation artifacts), are same-genus specimens closer in PC space than random specimen pairs?
 3. Non-articulated baseline: PCA on simple raw-mesh descriptors (bounding-box extents/aspect
    ratios, no SMAL fitting at all) as a null-ish comparison -- does the fitted articulated shape
    space add clustering signal beyond crude size/aspect-ratio shape descriptors?
"""
import json
import os
import re
import sys

import numpy as np
from pytorch3d.io import load_obj

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
D1DIR = os.path.join(REPO, "diagnostics", "d1_n50_evidence")
HOLDOUT_MESH_DIR = os.path.join(D1DIR, "all_ants_clean_holdout")
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

SEEDS = [0, 1, 2]


def parse_genus(label):
    """label like 'acanthomyrmex.obj' or 'cephalotes-varians.obj' or '01.obj'."""
    stem = label[:-4] if label.endswith(".obj") else label
    if re.fullmatch(r"\d+", stem):
        return None  # numeric ID, no taxonomy recoverable from filename
    return stem.split("-")[0]


def pca(X, n_components=2):
    """Plain PCA via SVD, mean-centered, no whitening. X: (N, D)."""
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:n_components]
    proj = Xc @ comps.T
    var_ratio = (S ** 2) / (S ** 2).sum()
    return proj, var_ratio[:n_components], comps


def load_betas_all_seeds():
    """Returns dict: label -> (3, 25) betas array, plus (3, 80) trans-normalized scale if wanted."""
    per_seed = {}
    labels_ref = None
    for s in SEEDS:
        p = os.path.join(D1DIR, "runs_holdout", f"seed{s}", "d1", "Stage_3_deform_fine.npz")
        d = np.load(p, allow_pickle=True)
        labels = [str(x) for x in d["labels"]]
        if labels_ref is None:
            labels_ref = labels
        else:
            assert labels == labels_ref, f"label order mismatch seed{s}"
        per_seed[s] = dict(betas=d["betas"], log_beta_scales=d["log_beta_scales"])
    return labels_ref, per_seed


def cross_seed_repeatability(labels, per_seed):
    """Intra-specimen (across-seed) scatter vs total (across-specimen) scatter, in raw beta space."""
    betas_stack = np.stack([per_seed[s]["betas"] for s in SEEDS], axis=0)  # (3, 80, 25)
    specimen_mean = betas_stack.mean(axis=0)  # (80, 25)
    intra_var = ((betas_stack - specimen_mean[None]) ** 2).mean()  # scalar, avg over seeds/specimens/dims
    total_var = betas_stack.reshape(-1, betas_stack.shape[-1]).var(axis=0).mean()  # pooled variance across all (seed,specimen)
    # a cleaner, standard framing: mean intra-specimen variance (across seeds) vs mean inter-specimen
    # variance of the specimen means (across specimens) -- classic one-way ANOVA framing
    inter_specimen_var = specimen_mean.var(axis=0).mean()
    ratio = float(inter_specimen_var / max(intra_var, 1e-12))
    return dict(intra_specimen_var=float(intra_var), inter_specimen_var=float(inter_specimen_var),
                variance_ratio_inter_over_intra=ratio, total_var_pooled=float(total_var))


def genus_consistency_check(labels, mean_betas, genus_by_label, rng):
    """For genera with >=2 specimens, is within-genus pairwise distance smaller than random pairs?"""
    genus_to_idx = {}
    for i, lb in enumerate(labels):
        g = genus_by_label[lb]
        if g is None:
            continue
        genus_to_idx.setdefault(g, []).append(i)
    multi_genus = {g: idxs for g, idxs in genus_to_idx.items() if len(idxs) >= 2}

    def dist(i, j):
        return float(np.linalg.norm(mean_betas[i] - mean_betas[j]))

    within_dists = []
    per_genus = {}
    for g, idxs in multi_genus.items():
        ds = [dist(i, j) for a, i in enumerate(idxs) for j in idxs[a + 1:]]
        per_genus[g] = dict(n_specimens=len(idxs), mean_within_dist=float(np.mean(ds)), pair_dists=ds)
        within_dists.extend(ds)

    labeled_idx = [i for i, lb in enumerate(labels) if genus_by_label[lb] is not None]
    n_random_pairs = max(len(within_dists) * 20, 200)
    random_dists = []
    for _ in range(n_random_pairs):
        i, j = rng.choice(labeled_idx, size=2, replace=False)
        random_dists.append(dist(i, j))

    return dict(
        multi_specimen_genera=per_genus,
        mean_within_genus_dist=float(np.mean(within_dists)) if within_dists else None,
        mean_random_pair_dist=float(np.mean(random_dists)),
        n_multi_specimen_genera=len(multi_genus),
        n_within_pairs=len(within_dists),
    )


def raw_mesh_descriptors(labels):
    """Non-articulated baseline: bbox extents + aspect ratios from the RAW target mesh, no SMAL
    fitting involved at all."""
    feats = []
    used_labels = []
    for lb in labels:
        p = os.path.join(HOLDOUT_MESH_DIR, lb)
        if not os.path.isfile(p):
            continue
        v, _, _ = load_obj(p, load_textures=False)
        v = v.numpy()
        extent = v.max(0) - v.min(0)
        extent_sorted = np.sort(extent)[::-1]  # scale-invariant-ish ordering: major/mid/minor axis
        aspect1 = extent_sorted[0] / max(extent_sorted[1], 1e-9)
        aspect2 = extent_sorted[1] / max(extent_sorted[2], 1e-9)
        volume_proxy = float(np.prod(extent))
        feats.append([extent_sorted[0], extent_sorted[1], extent_sorted[2], aspect1, aspect2, volume_proxy])
        used_labels.append(lb)
    return used_labels, np.array(feats)


def main():
    rng = np.random.default_rng(0)
    labels, per_seed = load_betas_all_seeds()
    genus_by_label = {lb: parse_genus(lb) for lb in labels}
    n_labeled = sum(1 for g in genus_by_label.values() if g is not None)
    n_unlabeled = len(labels) - n_labeled
    print(f"{len(labels)} specimens total: {n_labeled} with genus label (from filename), "
          f"{n_unlabeled} numeric-ID with NO taxonomy label available on disk")

    # --- 1. cross-seed repeatability ---
    rep = cross_seed_repeatability(labels, per_seed)
    print(f"\n=== Cross-seed repeatability (raw beta space) ===")
    print(f"inter-specimen variance / intra-specimen (cross-seed) variance = {rep['variance_ratio_inter_over_intra']:.2f}")
    print("(>>1 means specimen identity dominates fitting-seed noise -- a necessary precondition "
          "before any taxonomy-clustering claim is meaningful)")

    # --- 2. PCA on articulated shape (betas), specimen-mean across 3 seeds ---
    betas_stack = np.stack([per_seed[s]["betas"] for s in SEEDS], axis=0)  # (3, 80, 25)
    mean_betas = betas_stack.mean(axis=0)  # (80, 25)
    proj_betas, var_ratio_betas, _ = pca(mean_betas, n_components=2)
    print(f"\n=== PCA on fitted betas (25-dim -> 2D) ===")
    print(f"PC1/PC2 explained variance ratio: {var_ratio_betas[0]:.3f} / {var_ratio_betas[1]:.3f}")

    genus_check_betas = genus_consistency_check(labels, mean_betas, genus_by_label, rng)
    print(f"multi-specimen genera (n={genus_check_betas['n_multi_specimen_genera']}): "
          f"mean within-genus dist={genus_check_betas['mean_within_genus_dist']:.3f} vs "
          f"mean random-pair dist={genus_check_betas['mean_random_pair_dist']:.3f} "
          f"(smaller within-genus = genus-consistent signal)")
    for g, d in genus_check_betas["multi_specimen_genera"].items():
        print(f"  {g}: n={d['n_specimens']}, mean_within_dist={d['mean_within_dist']:.3f}")

    # --- 3. non-articulated baseline: raw mesh bbox descriptors ---
    raw_labels, raw_feats = raw_mesh_descriptors(labels)
    # z-score each feature before PCA (mixed units: length vs aspect ratio vs volume)
    raw_feats_z = (raw_feats - raw_feats.mean(0)) / raw_feats.std(0).clip(min=1e-9)
    proj_raw, var_ratio_raw, _ = pca(raw_feats_z, n_components=2)
    print(f"\n=== Non-articulated baseline: raw bbox descriptors (6-dim -> 2D) ===")
    print(f"PC1/PC2 explained variance ratio: {var_ratio_raw[0]:.3f} / {var_ratio_raw[1]:.3f}")
    raw_label_to_idx = {lb: i for i, lb in enumerate(raw_labels)}
    raw_mean_by_orig_idx = np.stack([raw_feats_z[raw_label_to_idx[lb]] for lb in labels])
    genus_check_raw = genus_consistency_check(labels, raw_mean_by_orig_idx, genus_by_label, rng)
    print(f"mean within-genus dist={genus_check_raw['mean_within_genus_dist']:.3f} vs "
          f"mean random-pair dist={genus_check_raw['mean_random_pair_dist']:.3f}")

    ratio_betas = genus_check_betas["mean_within_genus_dist"] / genus_check_betas["mean_random_pair_dist"]
    ratio_raw = genus_check_raw["mean_within_genus_dist"] / genus_check_raw["mean_random_pair_dist"]
    print(f"\n=== Comparison ===")
    print(f"within/random distance ratio -- fitted betas: {ratio_betas:.3f}, raw bbox baseline: {ratio_raw:.3f}")
    print("(ratio << 1 = strong genus clustering; ratio ~= 1 = no genus signal beyond chance; "
          "articulated model adds signal over the baseline only if its ratio is meaningfully lower)")

    # --- plots ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def scatter_plot(proj, labels_list, genus_map, title, path):
        fig, ax = plt.subplots(figsize=(9, 7))
        genera = sorted(set(g for g in genus_map.values() if g is not None))
        cmap = plt.get_cmap("tab20", max(len(genera), 1))
        color_by_genus = {g: cmap(i) for i, g in enumerate(genera)}
        for i, lb in enumerate(labels_list):
            g = genus_map[lb]
            c = color_by_genus[g] if g is not None else (0.6, 0.6, 0.6, 0.5)
            marker = "o" if g is not None else "x"
            ax.scatter(proj[i, 0], proj[i, 1], color=c, marker=marker, s=40)
            if g is not None:
                ax.annotate(g, (proj[i, 0], proj[i, 1]), fontsize=6, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)

    scatter_plot(proj_betas, labels, genus_by_label,
                 "ALL_ANTS_CLEAN holdout: fitted SMAL betas PCA (genus-colored, x=unlabeled)",
                 os.path.join(OUT, "a1_pca_betas.png"))
    scatter_plot(proj_raw, labels, genus_by_label,
                 "ALL_ANTS_CLEAN holdout: raw bbox descriptors PCA, non-articulated baseline",
                 os.path.join(OUT, "a1_pca_raw_baseline.png"))
    print(f"\nwrote {os.path.join(OUT, 'a1_pca_betas.png')} and a1_pca_raw_baseline.png")

    report = dict(
        n_specimens=len(labels), n_labeled=n_labeled, n_unlabeled=n_unlabeled,
        cross_seed_repeatability=rep,
        pca_betas=dict(var_ratio=var_ratio_betas.tolist()),
        pca_raw_baseline=dict(var_ratio=var_ratio_raw.tolist()),
        genus_consistency_betas=genus_check_betas,
        genus_consistency_raw_baseline=genus_check_raw,
        within_over_random_ratio_betas=ratio_betas,
        within_over_random_ratio_raw=ratio_raw,
        unlabeled_specimens=[lb for lb in labels if genus_by_label[lb] is None],
    )
    with open(os.path.join(OUT, "a1_pca_taxonomy.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"wrote {os.path.join(OUT, 'a1_pca_taxonomy.json')}")


if __name__ == "__main__":
    main()
