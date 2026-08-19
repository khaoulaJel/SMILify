"""Task 1, evaluation phase: does recovered absolute physical scale add information to SMILify's
ant morphometrics beyond the existing scale-free (Mosimann log-shape-ratio) representation?

Runs on the 276-specimen calibrated subset ONLY (diagnostics/absolute_scale/morphometrics_absolute_mm.csv,
rows with source=='worker' and calibration_status=='ok') -- never on the full 757/616-specimen
baseline, so genus/lot composition is held fixed across all three arms and results are directly
comparable to each other. The calibration itself (which specimens get a scale, which don't, and
why) is NOT revisited here -- see calibrate_scale.py and recompute_absolute_measurements.py.

METHODOLOGY REUSE. `pcs`, `accession_lot`, `loo_1nn`, `perm_test` are copied verbatim from
diagnostics/morphometrics/analyse.py on feature/registration_moonshot (mirrored here as
analyse_source.py) -- the existing validated lot-blind genus-classification protocol. Not
reimplemented, not modified. The only new methodology added is (a) a lot-level bootstrap CI, which
that script does not compute, and (b) block-balanced standardisation for the combined arm, needed
because concatenating two feature blocks of different sizes without correction lets the larger
block dominate PCA purely by feature count -- a standard multi-block problem, resolved the same way
CORE_BLOCKS' sqrt(n) block-SNR aggregation in measure.py resolves it.
"""

import csv
import json
import os
import re
import sys

import numpy as np
from scipy import stats

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from taxonomy_source import SUBFAMILY, ECOLOGY  # noqa: E402

IN_CSV = os.path.join(HERE, "morphometrics_absolute_mm.csv")
OUT_DIR = os.path.join(HERE, "out_evaluation")
os.makedirs(OUT_DIR, exist_ok=True)

MIN_N_GENUS = 3  # matches analyse.py's --min_n default
NPC = 10  # matches analyse.py's --npc default
N_PERM = 400  # matches analyse.py's perm_test default
N_BOOT = 1000
SEED = 0


# ------------------------------------------------------------------ reused verbatim from analyse.py
def pcs(A, k):
    """Standardise, then project onto the leading k principal components."""
    A = np.nan_to_num(np.asarray(A, dtype=np.float64))
    A = (A - A.mean(0)) / np.maximum(A.std(0), 1e-9)
    U, S, _ = np.linalg.svd(A - A.mean(0), full_matrices=False)
    return (U * S)[:, : min(k, A.shape[1])]


_CASENT = re.compile(r"(CASENT|OKENT|ANTWEB|UCDC|MCZ)[0-9A-Za-z]*", re.I)


def specimen_code(label):
    """Extract just the accession code from a worker label, exactly as measure.py's
    parse_taxonomy does (label may be 'Genus_species_CASENT0878066_processed.obj' or similar) --
    accession_lot needs the CODE alone, not the full genus_species_code label, or every label
    fails to match at position 0 and silently collapses into one fallback '?' lot."""
    stem = label[:-4] if label.endswith(".obj") else label
    stem = stem.replace("_processed", "")
    m = _CASENT.search(stem)
    return m.group(0) if m else stem


def accession_lot(specimen):
    m = re.match(r"([A-Za-z]+)0*(\d+)", specimen or "")
    return f"{m.group(1)}{int(m.group(2)) // 100}" if m else "?"


def loo_1nn(F, y, groups=None):
    D = ((F[:, None, :] - F[None, :, :]) ** 2).sum(-1)
    if groups is None:
        np.fill_diagonal(D, np.inf)
    else:
        groups = np.asarray(groups)
        for g in np.unique(groups):
            k = np.where(groups == g)[0]
            D[np.ix_(k, k)] = np.inf
    ok = np.isfinite(D).any(1)
    if not ok.any():
        return float("nan")
    return float((y[D[ok].argmin(1)] == y[ok]).mean())


def perm_test(F, y, n=N_PERM, seed=SEED, groups=None):
    rng = np.random.default_rng(seed)
    a = loo_1nn(F, y, groups)
    null = np.array([loo_1nn(F, rng.permutation(y), groups) for _ in range(n)])
    return a, float(null.mean()), float(null.std()), float((null >= a).mean())


# ------------------------------------------------------------------ new: block-balanced PCA + bootstrap CI
def zscore(A):
    A = np.nan_to_num(np.asarray(A, dtype=np.float64))
    return (A - A.mean(0)) / np.maximum(A.std(0), 1e-9)


def pcs_from_standardized(A, k):
    U, S, _ = np.linalg.svd(A - A.mean(0), full_matrices=False)
    return (U * S)[:, : min(k, A.shape[1])]


def block_balanced_pcs(blocks, k):
    """Concatenate pre-standardised feature blocks, each downweighted by 1/sqrt(n_features), so
    every REPRESENTATION (not every raw feature) gets equal total variance in the PCA -- otherwise
    a block with more columns dominates purely by feature count, not by information content."""
    parts = [zscore(B) / np.sqrt(B.shape[1]) for B in blocks]
    return pcs_from_standardized(np.hstack(parts), k)


def bootstrap_ci_lot(F, y, lots, n_boot=N_BOOT, seed=1, alpha=0.05):
    """Lot-level (cluster) bootstrap: resample LOTS with replacement (not specimens), so any
    duplicated lot's specimens are still grouped together for the leave-one-lot-out exclusion --
    preserving the lot-blind guarantee inside the bootstrap, not just in the point estimate."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(lots)
    accs = []
    for _ in range(n_boot):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(lots == g)[0] for g in draw])
        acc = loo_1nn(F[idx], y[idx], groups=lots[idx])
        if np.isfinite(acc):
            accs.append(acc)
    accs = np.array(accs)
    lo, hi = np.percentile(accs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), accs


# ------------------------------------------------------------------ load data
def load_calibrated_rows():
    with open(IN_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    calibrated = [r for r in rows if r["source"] == "worker" and r["calibration_status"] == "ok"]
    return rows, calibrated


def main():
    all_rows, rows = load_calibrated_rows()
    print(f"Loaded {len(all_rows)} total rows; {len(rows)} calibrated worker specimens (source=worker, status=ok).")

    header = list(rows[0].keys())
    meta = {"label", "source", "genus", "species", "subfamily", "ecology", "asym_median", "log_size",
            "mm_per_model_unit", "calibration_status"}
    mm_suffix = "_mm"
    prop_cols = [c for c in header if c not in meta and not c.endswith(mm_suffix)]
    mm_cols = [c for c in header if c.endswith(mm_suffix)]
    shapedev_cols = [c for c in prop_cols if c.startswith("shapedev_")]
    length_ratio_cols = [c for c in prop_cols if c not in shapedev_cols]
    print(f"{len(length_ratio_cols)} length log-shape-ratio cols, {len(shapedev_cols)} shape-dev cols, "
          f"{len(mm_cols)} absolute-mm cols.")

    genus = np.array([r["genus"] or "?" for r in rows])
    species = np.array([r["species"] or "?" for r in rows])
    lot = np.array([accession_lot(specimen_code(r["label"])) for r in rows])
    ecology = np.array([ECOLOGY.get(g, "") for g in genus])

    Z = np.array([[float(r[c]) for c in length_ratio_cols] for r in rows])
    D = zscore(np.array([[float(r[c]) for c in shapedev_cols] for r in rows]))
    prop_matrix = np.hstack([Z, D])  # matches analyse.py's `features()` output exactly

    mm_raw = np.array([[float(r[c]) for c in mm_cols] for r in rows])
    log_mm = np.log(np.maximum(mm_raw, 1e-6))

    # ---------------- min_n genus filter, exactly as analyse.py's signal test does
    cnt = {g: int((genus == g).sum()) for g in set(genus)}
    keep = np.array([cnt.get(g, 0) >= MIN_N_GENUS and g != "?" for g in genus])
    print(f"\nAfter min_n>={MIN_N_GENUS} genus filter: {keep.sum()} specimens, {len(set(genus[keep]))} genera, "
          f"{len(set(lot[keep]))} lots.")

    gk, lk = genus[keep], lot[keep]

    results = {"n_calibrated_total": len(rows), "n_after_min_n_filter": int(keep.sum()),
               "n_genera": len(set(gk)), "n_lots": len(set(lk)), "min_n_genus": MIN_N_GENUS}

    # ==================================================================
    # 1. THREE-ARM MATCHED EVALUATION
    # ==================================================================
    print("\n" + "=" * 70)
    print("1. THREE-ARM MATCHED EVALUATION (lot-blind 1-NN genus classification)")
    print("=" * 70)

    arms = {}

    Fa = pcs(prop_matrix[keep], NPC)
    Fb = pcs(log_mm[keep], NPC)
    Fc = block_balanced_pcs([prop_matrix[keep], log_mm[keep]], NPC)

    for name, F in [("A_proportions_only", Fa), ("B_absolute_only", Fb), ("C_proportions_plus_absolute", Fc)]:
        a, m, s, p = perm_test(F, gk, groups=lk)
        lo, hi, _ = bootstrap_ci_lot(F, gk, lk)
        lift = a / max(m, 1e-9)
        arms[name] = dict(n=int(keep.sum()), n_genus=len(set(gk)), n_lots=len(set(lk)),
                           accuracy=a, null_mean=m, null_std=s, p_value=p, lift=lift,
                           ci95_lo=lo, ci95_hi=hi)
        print(f"\n  [{name}]")
        print(f"    n={keep.sum()} specimens, {len(set(gk))} genera, {len(set(lk))} lots")
        print(f"    accuracy (lot-blind 1-NN) = {100*a:.1f}%   [95% CI {100*lo:.1f}-{100*hi:.1f}%]")
        print(f"    permutation null = {100*m:.1f} +- {100*s:.1f}%   p={p:.4f}")
        print(f"    LIFT = {lift:.2f}x")

    results["three_arm"] = arms

    combined_beats_proportions = arms["C_proportions_plus_absolute"]["accuracy"] > arms["A_proportions_only"]["accuracy"]
    size_alone_has_signal = arms["B_absolute_only"]["p_value"] < 0.05
    print(f"\n  combined > proportions-only: {combined_beats_proportions} "
          f"({100*arms['C_proportions_plus_absolute']['accuracy']:.1f}% vs {100*arms['A_proportions_only']['accuracy']:.1f}%)")
    print(f"  size alone carries genus signal (p<0.05): {size_alone_has_signal}")

    # ==================================================================
    # 2. LOT CONFOUND CHECK FOR ABSOLUTE SIZE
    # ==================================================================
    print("\n" + "=" * 70)
    print("2. LOT CONFOUND CHECK — does absolute size alone predict COLLECTION LOT?")
    print("=" * 70)
    # This is NOT the lot-blind genus test. Here lot itself is the class being predicted, so the
    # natural protocol is plain leave-one-specimen-out (no groups) -- the question is "do same-lot
    # specimens cluster in absolute-size space", which is exactly what would happen under a
    # fixation/scan-batch artefact.
    Fb_full = pcs(log_mm[keep], NPC)
    a_lot, m_lot, s_lot, p_lot = perm_test(Fb_full, lk, groups=None)
    lift_lot = a_lot / max(m_lot, 1e-9)
    print(f"  absolute size -> lot: accuracy={100*a_lot:.1f}%  null={100*m_lot:.1f}+-{100*s_lot:.1f}%  "
          f"p={p_lot:.4f}  LIFT={lift_lot:.2f}x")
    results["size_predicts_lot"] = dict(accuracy=a_lot, null_mean=m_lot, null_std=s_lot, p_value=p_lot, lift=lift_lot)
    if p_lot < 0.05 and lift_lot > 1.5:
        print("  WARNING: absolute size shows a lot-associated signal -- treat arm B/C genus results "
              "with caution until this is investigated further (see report).")
        results["size_lot_confound_flag"] = True
    else:
        print("  No strong lot association detected for absolute size.")
        results["size_lot_confound_flag"] = False

    # ==================================================================
    # 3. DISTRIBUTION / PLAUSIBILITY CHECKS
    # ==================================================================
    print("\n" + "=" * 70)
    print("3. DISTRIBUTION AND PLAUSIBILITY OF RECOVERED ABSOLUTE MEASUREMENTS")
    print("=" * 70)
    log_size_abs = log_mm.mean(1)  # log geometric mean of the mm measurements, an isometric-size proxy
    size_abs_mm = np.exp(log_size_abs)
    order = np.argsort(size_abs_mm)
    print("  Smallest 5 (geometric-mean mm across measurements):")
    for i in order[:5]:
        print(f"    {rows[i]['label'][:45]:45s} {size_abs_mm[i]:.3f} mm   genus={genus[i]}")
    print("  Largest 5:")
    for i in order[-5:]:
        print(f"    {rows[i]['label'][:45]:45s} {size_abs_mm[i]:.3f} mm   genus={genus[i]}")

    genus_counts = {g: int((genus == g).sum()) for g in set(genus)}
    top_genus = max(genus_counts, key=genus_counts.get)
    lot_counts = {lotv: int((lot == lotv).sum()) for lotv in set(lot)}
    top_lot = max(lot_counts, key=lot_counts.get)
    print(f"\n  Most-represented genus in calibrated subset: {top_genus} ({genus_counts[top_genus]}/{len(rows)} "
          f"= {100*genus_counts[top_genus]/len(rows):.1f}%)")
    print(f"  Most-represented lot: {top_lot} ({lot_counts[top_lot]}/{len(rows)} = "
          f"{100*lot_counts[top_lot]/len(rows):.1f}%)")

    results["distribution"] = dict(
        size_mm_min=float(size_abs_mm.min()), size_mm_max=float(size_abs_mm.max()),
        size_mm_median=float(np.median(size_abs_mm)),
        top_genus=top_genus, top_genus_frac=genus_counts[top_genus] / len(rows),
        top_lot=top_lot, top_lot_frac=lot_counts[top_lot] / len(rows),
        n_genera_total=len(genus_counts), n_lots_total=len(lot_counts),
    )

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.hist(size_abs_mm, bins=30, color="#4a5568")
    ax.set_xlabel("geometric-mean absolute size (mm)")
    ax.set_ylabel("count")
    ax.set_title(f"Absolute-size distribution, calibrated subset (n={len(rows)})")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_size_distribution.png"), dpi=125)
    plt.close(fig)

    # ==================================================================
    # 4. DOES ABSOLUTE SIZE CHANGE THE SHAPE SPACE?
    # ==================================================================
    print("\n" + "=" * 70)
    print("4. ABSOLUTE SIZE vs THE EXISTING SCALE-FREE PCA")
    print("=" * 70)
    P_shape = pcs(prop_matrix[keep], 6)  # shape-only PCA, same as analyse.py's make_figures
    log_size_k = log_size_abs[keep]

    pc_corr = {}
    for j in range(3):
        r, p = stats.pearsonr(P_shape[:, j], log_size_k)
        pc_corr[f"PC{j+1}"] = dict(r=float(r), p=float(p))
        slope, intercept, r_reg, p_reg, se = stats.linregress(log_size_k, P_shape[:, j])
        pc_corr[f"PC{j+1}"]["regression_slope"] = float(slope)
        pc_corr[f"PC{j+1}"]["regression_r2"] = float(r_reg ** 2)
        print(f"  PC{j+1} vs log(absolute size): r={r:.3f} (p={p:.4f})   "
              f"regression R^2={r_reg**2:.3f} (p={p_reg:.4f})")
    results["pc_vs_size_correlation"] = pc_corr

    P_combined = block_balanced_pcs([prop_matrix[keep], log_mm[keep]], 6)
    pc1_shift_corr = float(np.corrcoef(P_shape[:, 0], P_combined[:, 0])[0, 1])
    print(f"\n  correlation between shape-only PC1 and combined-PCA PC1: r={pc1_shift_corr:.3f}")
    print("  (r close to 1 = combined PC1 is essentially the same axis as shape-only PC1;"
          " a lower r means absolute size measurably shifts what the dominant axis represents.)")
    results["pc1_shape_vs_combined_correlation"] = pc1_shift_corr

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].scatter(log_size_k, P_shape[:, 0], s=14, alpha=0.7, c="#2b6cb0")
    ax[0].set_xlabel("log(absolute size, mm)")
    ax[0].set_ylabel("shape-only PC1")
    ax[0].set_title(f"PC1 vs size (r={pc_corr['PC1']['r']:.2f}, p={pc_corr['PC1']['p']:.3f})", fontsize=10)
    ax[1].scatter(log_size_k, P_shape[:, 1], s=14, alpha=0.7, c="#2f855a")
    ax[1].set_xlabel("log(absolute size, mm)")
    ax[1].set_ylabel("shape-only PC2")
    ax[1].set_title(f"PC2 vs size (r={pc_corr['PC2']['r']:.2f}, p={pc_corr['PC2']['p']:.3f})", fontsize=10)
    ax[2].scatter(P_shape[:, 0], P_combined[:, 0], s=14, alpha=0.7, c="#805ad5")
    ax[2].set_xlabel("shape-only PC1")
    ax[2].set_ylabel("combined-PCA PC1")
    ax[2].set_title(f"shape PC1 vs combined PC1 (r={pc1_shift_corr:.2f})", fontsize=10)
    for a_ in ax:
        a_.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_pc_vs_size.png"), dpi=125)
    plt.close(fig)

    # ==================================================================
    # 5. ECOLOGICAL / FUNCTIONAL SIGNAL
    # ==================================================================
    print("\n" + "=" * 70)
    print("5. ECOLOGICAL / FUNCTIONAL GUILD SIGNAL")
    print("=" * 70)
    eco_k = ecology[keep]
    has_guild = eco_k != ""
    guild_counts = {g: int((eco_k[has_guild] == g).sum()) for g in set(eco_k[has_guild])}
    print(f"  {int(has_guild.sum())} of {keep.sum()} specimens have a guild label: {guild_counts}")

    eco_results = {"n_labeled": int(has_guild.sum()), "guild_counts": guild_counts}
    guilds_with_ge2 = {g for g, c in guild_counts.items() if c >= 2}
    eco_mask = has_guild & np.isin(eco_k, list(guilds_with_ge2))
    n_eco = int(eco_mask.sum())
    n_guilds = len(guilds_with_ge2)

    ROBUST_MIN = 15
    if n_eco >= ROBUST_MIN and n_guilds >= 3:
        eco_gk = eco_k[eco_mask]
        eco_lk = lk[eco_mask]  # lots restricted to the same rows
        for name, F_full in [("A_proportions_only", Fa), ("B_absolute_only", Fb), ("C_proportions_plus_absolute", Fc)]:
            F_eco = F_full[eco_mask]
            a, m, s, p = perm_test(F_eco, eco_gk, groups=eco_lk)
            lift = a / max(m, 1e-9)
            eco_results.setdefault("guild_classification", {})[name] = dict(
                n=n_eco, n_guild=n_guilds, accuracy=a, null_mean=m, p_value=p, lift=lift)
            print(f"  guild classification [{name}]: n={n_eco}, {n_guilds} guilds, "
                  f"acc={100*a:.1f}%  null={100*m:.1f}%  p={p:.4f}  LIFT={lift:.2f}x")

        # guild separation on PC1: between/within variance ratio (eta-squared), shape-only vs combined
        def eta_squared(pc1_vals, labels):
            grand_mean = pc1_vals.mean()
            ss_between = sum(((pc1_vals[labels == g]).mean() - grand_mean) ** 2 * (labels == g).sum()
                              for g in set(labels))
            ss_total = ((pc1_vals - grand_mean) ** 2).sum()
            return float(ss_between / ss_total) if ss_total > 0 else float("nan")

        eta_shape = eta_squared(P_shape[eco_mask, 0], eco_gk)
        eta_combined = eta_squared(P_combined[eco_mask, 0], eco_gk)
        print(f"\n  guild separation on PC1 (eta^2): shape-only={eta_shape:.3f}  combined={eta_combined:.3f}")
        print(f"  ({'strengthens' if eta_combined > eta_shape else 'weakens or unchanged'} with absolute size added)")
        eco_results["pc1_guild_eta2_shape"] = eta_shape
        eco_results["pc1_guild_eta2_combined"] = eta_combined

        # does absolute size alone predict guild?
        a_sz, m_sz, s_sz, p_sz = perm_test(Fb[eco_mask], eco_gk, groups=eco_lk)
        print(f"  absolute size alone -> guild: acc={100*a_sz:.1f}%  null={100*m_sz:.1f}%  p={p_sz:.4f}")
        eco_results["size_alone_predicts_guild"] = dict(accuracy=a_sz, null_mean=m_sz, p_value=p_sz)
    else:
        print(f"  Only {n_eco} specimens across {n_guilds} guilds with >=2 members -- below the "
              f"{ROBUST_MIN}-specimen/3-guild threshold for a formal permutation test. Treating as "
              f"exploratory / descriptive only, per instructions not to force a test past its sample size.")
        eco_results["robust_test_skipped"] = True
        eco_results["reason"] = f"n_eco={n_eco}, n_guilds={n_guilds} below robustness threshold"

    results["ecology"] = eco_results

    if has_guild.sum() >= 5:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        cmap = plt.get_cmap("tab10")
        guild_list = sorted(guild_counts, key=lambda g: -guild_counts[g])
        for i, g in enumerate(guild_list):
            m = has_guild & (eco_k == g)
            ax[0].scatter(P_shape[m, 0], P_shape[m, 1], s=30, color=cmap(i % 10), label=f"{g} ({guild_counts[g]})")
            ax[1].scatter(P_combined[m, 0], P_combined[m, 1], s=30, color=cmap(i % 10), label=f"{g}")
        ax[0].set_title("Shape-only PCA, by guild", fontsize=10)
        ax[1].set_title("Combined (shape+absolute) PCA, by guild", fontsize=10)
        for a_ in ax:
            a_.set_xlabel("PC1")
            a_.set_ylabel("PC2")
            a_.grid(alpha=0.3)
        ax[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "fig_guild_pca.png"), dpi=125)
        plt.close(fig)

    # ==================================================================
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nWrote {os.path.join(OUT_DIR, 'results.json')}")
    print("Wrote figures to", OUT_DIR)


if __name__ == "__main__":
    main()
