"""Task 1, validation pass: four follow-up checks on the evaluation phase results.

A. Does the absolute-size -> genus signal survive lot-residualisation, computed leak-free?
B. Do the guild results (the +5.7pp combined-over-shape lift) survive the same correction?
C. Is the 276-specimen calibratable subset a biased sample of the full 757-worker corpus?
D. What actually loads onto the combined-PCA's leading axis -- descriptively, not by assumption.

LEAKAGE DISCIPLINE FOR A/B (read before touching this). "Residualise size against lot" here means:
for each specimen, subtract the mean of its OWN lot's OTHER members (leave-one-out within lot) from
its raw log-mm features. This uses only (a) the specimen's own already-known lot label and (b) its
lot-mates' measurements -- never a genus label, and never any OTHER lot's data. It is computed ONCE,
globally, before any train/test split, and is safe under leave-one-LOT-out CV for the reason the
existing `loo_1nn` group-exclusion already guarantees structurally: a specimen's own lot can never
be used as its nearest-neighbour candidate (whole lot is excluded from the distance search), so
whether that specimen's feature vector was itself computed using its lot-mates' data or not, no
information about ANOTHER lot or about the genus label ever enters this transform. Singleton lots
(no other member to compute a mean from) fall back to the global grand mean instead -- documented,
not silently dropped.
"""

import csv
import json
import os
import sys

import numpy as np
from scipy import stats

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from taxonomy_source import SUBFAMILY, ECOLOGY  # noqa: E402
from evaluate_absolute_scale import (  # noqa: E402
    pcs, specimen_code, accession_lot, loo_1nn, perm_test, zscore,
    pcs_from_standardized, block_balanced_pcs, bootstrap_ci_lot,
    MIN_N_GENUS, NPC,
)

IN_CSV = os.path.join(HERE, "morphometrics_absolute_mm.csv")
SOURCE_CSV = os.path.join(HERE, "morphometrics_source.csv")  # ALL 757 workers + 81 clean, proportions only
OUT_DIR = os.path.join(HERE, "out_validation")
os.makedirs(OUT_DIR, exist_ok=True)


def loo_lot_residualize(X, lot):
    """Leave-one-out within-lot mean subtraction. Singleton lots fall back to the grand mean."""
    X = np.asarray(X, dtype=np.float64)
    lot = np.asarray(lot)
    out = np.zeros_like(X)
    grand_mean = X.mean(0)
    n_singleton = 0
    for g in np.unique(lot):
        idx = np.where(lot == g)[0]
        if len(idx) == 1:
            out[idx[0]] = X[idx[0]] - grand_mean
            n_singleton += 1
            continue
        s = X[idx].sum(0)
        for i in idx:
            loo_mean = (s - X[i]) / (len(idx) - 1)
            out[i] = X[i] - loo_mean
    return out, n_singleton


def load_calibrated():
    with open(IN_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    cal = [r for r in rows if r["source"] == "worker" and r["calibration_status"] == "ok"]
    header = list(cal[0].keys())
    meta = {"label", "source", "genus", "species", "subfamily", "ecology", "asym_median", "log_size",
            "mm_per_model_unit", "calibration_status"}
    mm_cols = [c for c in header if c.endswith("_mm")]
    prop_cols = [c for c in header if c not in meta and not c.endswith("_mm")]
    shapedev_cols = [c for c in prop_cols if c.startswith("shapedev_")]
    length_cols = [c for c in prop_cols if c not in shapedev_cols]
    return cal, length_cols, shapedev_cols, mm_cols


def main():
    results = {}
    cal, length_cols, shapedev_cols, mm_cols = load_calibrated()
    genus = np.array([r["genus"] or "?" for r in cal])
    lot = np.array([accession_lot(specimen_code(r["label"])) for r in cal])
    ecology = np.array([ECOLOGY.get(g, "") for g in genus])

    Z = np.array([[float(r[c]) for c in length_cols] for r in cal])
    D = zscore(np.array([[float(r[c]) for c in shapedev_cols] for r in cal]))
    prop_matrix = np.hstack([Z, D])
    mm_raw = np.array([[float(r[c]) for c in mm_cols] for r in cal])
    log_mm = np.log(np.maximum(mm_raw, 1e-6))

    cnt = {g: int((genus == g).sum()) for g in set(genus)}
    keep = np.array([cnt.get(g, 0) >= MIN_N_GENUS and g != "?" for g in genus])
    gk, lk = genus[keep], lot[keep]
    print(f"n={keep.sum()} specimens, {len(set(gk))} genera, {len(set(lk))} lots (same set as evaluation phase).")

    # ==================================================================
    # A. Size -> genus, before/after leak-free lot residualisation
    # ==================================================================
    print("\n" + "=" * 70)
    print("A. SIZE -> GENUS, BEFORE vs AFTER LOT RESIDUALISATION")
    print("=" * 70)

    log_mm_resid, n_singleton_lots = loo_lot_residualize(log_mm, lot)
    print(f"  ({n_singleton_lots} singleton lots fell back to grand-mean centring)")

    Fb_before = pcs(log_mm[keep], NPC)
    Fb_after = pcs(log_mm_resid[keep], NPC)

    a0, m0, s0, p0 = perm_test(Fb_before, gk, groups=lk)
    a1, m1, s1, p1 = perm_test(Fb_after, gk, groups=lk)
    lo0, hi0, _ = bootstrap_ci_lot(Fb_before, gk, lk)
    lo1, hi1, _ = bootstrap_ci_lot(Fb_after, gk, lk)

    print(f"  BEFORE residualisation: acc={100*a0:.1f}%  null={100*m0:.1f}+-{100*s0:.1f}%  p={p0:.4f}  "
          f"lift={a0/max(m0,1e-9):.2f}x  CI[{100*lo0:.1f}-{100*hi0:.1f}%]")
    print(f"  AFTER  residualisation: acc={100*a1:.1f}%  null={100*m1:.1f}+-{100*s1:.1f}%  p={p1:.4f}  "
          f"lift={a1/max(m1,1e-9):.2f}x  CI[{100*lo1:.1f}-{100*hi1:.1f}%]")
    retained_frac = (a1 / max(m1, 1e-9)) / (a0 / max(m0, 1e-9)) if a0 > 0 else float("nan")
    print(f"  lift retained after residualisation: {100*retained_frac:.1f}% of original lift")
    verdict_a = "SURVIVES (substantially above chance)" if (p1 < 0.05 and a1 / max(m1, 1e-9) > 1.3) else \
                "COLLAPSES (mostly collection structure)"
    print(f"  VERDICT: {verdict_a}")

    results["A_size_lot_residual"] = dict(
        before=dict(acc=a0, null=m0, p=p0, lift=a0/max(m0,1e-9), ci95=[lo0, hi0]),
        after=dict(acc=a1, null=m1, p=p1, lift=a1/max(m1,1e-9), ci95=[lo1, hi1]),
        lift_retained_fraction=retained_frac, n_singleton_lots=n_singleton_lots, verdict=verdict_a,
    )

    # ==================================================================
    # B. Guild classification, before/after, shape / size / shape+size
    # ==================================================================
    print("\n" + "=" * 70)
    print("B. GUILD CLASSIFICATION UNDER THE SAME CORRECTION")
    print("=" * 70)
    eco_k = ecology[keep]
    has_guild = eco_k != ""
    guild_counts = {g: int((eco_k[has_guild] == g).sum()) for g in set(eco_k[has_guild])}
    guilds_ge2 = {g for g, c in guild_counts.items() if c >= 2}
    eco_mask = has_guild & np.isin(eco_k, list(guilds_ge2))
    n_eco, n_guilds = int(eco_mask.sum()), len(guilds_ge2)
    print(f"  {n_eco} specimens, {n_guilds} guilds: {guild_counts}")

    Fa_full = pcs(prop_matrix[keep], NPC)
    Fb_full_before = Fb_before
    Fb_full_after = Fb_after
    Fc_full_before = block_balanced_pcs([prop_matrix[keep], log_mm[keep]], NPC)
    Fc_full_after = block_balanced_pcs([prop_matrix[keep], log_mm_resid[keep]], NPC)

    guild_res = {}
    if n_eco >= 15 and n_guilds >= 3:
        eco_gk = eco_k[eco_mask]
        eco_lk = lk[eco_mask]
        for tag, F_full in [
            ("shape", Fa_full),
            ("size_before", Fb_full_before), ("size_after", Fb_full_after),
            ("shape+size_before", Fc_full_before), ("shape+size_after", Fc_full_after),
        ]:
            F_eco = F_full[eco_mask]
            a, m, s, p = perm_test(F_eco, eco_gk, groups=eco_lk)
            guild_res[tag] = dict(acc=a, null=m, p=p, lift=a/max(m,1e-9))
            print(f"  [{tag:<20}] acc={100*a:.1f}%  null={100*m:.1f}%  p={p:.4f}  lift={a/max(m,1e-9):.2f}x")

        gap_before = guild_res["shape+size_before"]["acc"] - guild_res["shape"]["acc"]
        gap_after = guild_res["shape+size_after"]["acc"] - guild_res["shape"]["acc"]
        print(f"\n  combined-over-shape gap BEFORE residualisation: {100*gap_before:+.1f}pp")
        print(f"  combined-over-shape gap AFTER  residualisation: {100*gap_after:+.1f}pp")
        gap_verdict = "SURVIVES" if gap_after > 0.3 * gap_before else "MOSTLY DUE TO LOT-ASSOCIATED SIZE VARIATION"
        print(f"  VERDICT: the +{100*gap_before:.1f}pp gap {gap_verdict}")
        guild_res["gap_before_pp"] = 100 * gap_before
        guild_res["gap_after_pp"] = 100 * gap_after
        guild_res["gap_verdict"] = gap_verdict
    else:
        print(f"  n={n_eco}, guilds={n_guilds} below robustness threshold -- skipped, matches evaluation-phase gate.")
        guild_res["skipped"] = True
    results["B_guild_lot_residual"] = guild_res

    # ==================================================================
    # C. Retained (276) vs excluded (478) worker specimens
    # ==================================================================
    print("\n" + "=" * 70)
    print("C. RETAINED (calibratable) vs EXCLUDED (no raw-scan match) WORKER SPECIMENS")
    print("=" * 70)
    with open(SOURCE_CSV, newline="") as f:
        src_rows = list(csv.DictReader(f))
    worker_rows = [r for r in src_rows if r["source"] == "worker"]
    src_header = list(worker_rows[0].keys())
    src_meta = {"label", "source", "genus", "species", "subfamily", "ecology", "asym_median", "log_size"}
    src_length_cols = [c for c in src_header if c not in src_meta and not c.startswith("shapedev_")]
    src_shapedev_cols = [c for c in src_header if c.startswith("shapedev_")]

    calibrated_labels = {r["label"] for r in cal}
    is_retained = np.array([r["label"] in calibrated_labels for r in worker_rows])
    print(f"  {is_retained.sum()} retained, {(~is_retained).sum()} excluded, of {len(worker_rows)} total workers.")

    w_genus = np.array([r["genus"] or "?" for r in worker_rows])
    w_sub = np.array([SUBFAMILY.get(g, "?") for g in w_genus])
    w_logsize = np.array([float(r["log_size"]) for r in worker_rows])
    w_asym = np.array([float(r["asym_median"]) for r in worker_rows])
    w_Z = np.array([[float(r[c]) for c in src_length_cols] for r in worker_rows])
    w_D = zscore(np.array([[float(r[c]) for c in src_shapedev_cols] for r in worker_rows]))
    w_prop = np.hstack([w_Z, w_D])
    w_P = pcs(w_prop, 6)  # single global shape PCA over ALL 757 workers, for a fair retained-vs-excluded comparison

    retained_genera = set(w_genus[is_retained])
    excluded_only_genera = set(w_genus[~is_retained]) - retained_genera
    print(f"  genera: {len(retained_genera)} in retained, {len(excluded_only_genera)} present ONLY in excluded "
          f"(never calibratable): {sorted(excluded_only_genera)[:15]}{'...' if len(excluded_only_genera) > 15 else ''}")

    def two_sample(name, x):
        a, b = x[is_retained], x[~is_retained]
        t, p_t = stats.ttest_ind(a, b, equal_var=False)
        ks, p_ks = stats.ks_2samp(a, b)
        print(f"  {name:<18} retained mean={a.mean():.3f} (sd={a.std():.3f})  excluded mean={b.mean():.3f} "
              f"(sd={b.std():.3f})   Welch t p={p_t:.4f}   KS p={p_ks:.4f}")
        return dict(retained_mean=float(a.mean()), excluded_mean=float(b.mean()), t_p=float(p_t), ks_p=float(p_ks))

    c_res = {"n_retained": int(is_retained.sum()), "n_excluded": int((~is_retained).sum()),
             "n_genera_retained": len(retained_genera), "n_genera_excluded_only": len(excluded_only_genera),
             "excluded_only_genera": sorted(excluded_only_genera)}
    c_res["log_size_model_units"] = two_sample("log_size (model units)", w_logsize)
    c_res["asym_median"] = two_sample("asym_median", w_asym)
    c_res["PC1_shape"] = two_sample("shape PC1", w_P[:, 0])
    c_res["PC2_shape"] = two_sample("shape PC2", w_P[:, 1])

    sub_counts_r = {s_: int(((w_sub == s_) & is_retained).sum()) for s_ in set(w_sub)}
    sub_counts_e = {s_: int(((w_sub == s_) & ~is_retained).sum()) for s_ in set(w_sub)}
    print("\n  subfamily composition (retained | excluded):")
    for s_ in sorted(set(w_sub), key=lambda s_: -(sub_counts_r.get(s_, 0) + sub_counts_e.get(s_, 0))):
        if s_ == "?":
            continue
        print(f"    {s_:<16} {sub_counts_r.get(s_,0):>4} | {sub_counts_e.get(s_,0):>4}")
    c_res["subfamily_retained"] = sub_counts_r
    c_res["subfamily_excluded"] = sub_counts_e
    results["C_retained_vs_excluded"] = c_res

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].hist(w_logsize[is_retained], bins=25, alpha=0.6, label="retained", density=True, color="#2b6cb0")
    ax[0].hist(w_logsize[~is_retained], bins=25, alpha=0.6, label="excluded", density=True, color="#c05621")
    ax[0].set_xlabel("log_size (model units)")
    ax[0].set_title("Isometric size, retained vs excluded", fontsize=10)
    ax[0].legend()
    ax[1].scatter(w_P[~is_retained, 0], w_P[~is_retained, 1], s=10, alpha=0.4, color="#c05621", label="excluded")
    ax[1].scatter(w_P[is_retained, 0], w_P[is_retained, 1], s=10, alpha=0.6, color="#2b6cb0", label="retained")
    ax[1].set_xlabel("PC1")
    ax[1].set_ylabel("PC2")
    ax[1].set_title("Shape space, retained vs excluded", fontsize=10)
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_retained_vs_excluded.png"), dpi=125)
    plt.close(fig)

    # ==================================================================
    # D. Inspect the "size axis" -- descriptively, no presupposition
    # ==================================================================
    print("\n" + "=" * 70)
    print("D. INSPECTING THE COMBINED-PCA LEADING AXIS")
    print("=" * 70)
    prop_names = length_cols + shapedev_cols
    abs_names = [c.replace("_mm", "") + "_abs" for c in mm_cols]
    block_prop = zscore(prop_matrix[keep]) / np.sqrt(prop_matrix.shape[1])
    block_abs = zscore(log_mm[keep]) / np.sqrt(log_mm.shape[1])
    combined_std = np.hstack([block_prop, block_abs])
    combined_names = prop_names + abs_names
    U, S, Vt = np.linalg.svd(combined_std - combined_std.mean(0), full_matrices=False)
    PC_combined = (U * S)
    loadings_pc1 = Vt[0]  # loading of each (block-scaled, standardised) feature onto PC1

    order = np.argsort(-np.abs(loadings_pc1))
    print("  Top 15 |loading| on combined PC1:")
    top_loadings = []
    for i in order[:15]:
        print(f"    {combined_names[i]:<22} {loadings_pc1[i]:+.3f}")
        top_loadings.append((combined_names[i], float(loadings_pc1[i])))

    n_prop = block_prop.shape[1]
    prop_loading_mass = float((loadings_pc1[:n_prop] ** 2).sum())
    abs_loading_mass = float((loadings_pc1[n_prop:] ** 2).sum())
    print(f"\n  squared-loading mass on PC1: proportions block={prop_loading_mass:.3f}   "
          f"absolute block={abs_loading_mass:.3f}  (sums to 1.0)")

    log_size_k = log_mm[keep].mean(1)
    r_size, p_size = stats.pearsonr(PC_combined[:, 0], log_size_k)
    print(f"\n  combined PC1 vs log(geometric-mean absolute size): r={r_size:.3f} (p={p_size:.4f})")

    if "mesosoma_len_mm" in mm_cols:
        wl_proxy = np.log(np.maximum(mm_raw[keep, mm_cols.index("mesosoma_len_mm")], 1e-6))
        r_wl, p_wl = stats.pearsonr(PC_combined[:, 0], wl_proxy)
        print(f"  combined PC1 vs log(mesosoma_len_mm) [Weber's-length PROXY -- not the standardised "
              f"AntWeb landmark, no exact WL measurement exists in this pipeline]: r={r_wl:.3f} (p={p_wl:.4f})")
    else:
        r_wl, p_wl = float("nan"), float("nan")

    def shape_index(col_a, col_b):
        # Z columns are log-shape-ratios: Z_a - Z_b = log(raw_a) - log(raw_b) = log(raw_a/raw_b),
        # independent of log_size -- a genuine scale-free shape index.
        if col_a not in length_cols or col_b not in length_cols:
            return None
        za = Z[keep][:, length_cols.index(col_a)]
        zb = Z[keep][:, length_cols.index(col_b)]
        return za - zb

    indices = {
        "cephalic_index (head_wid/head_len)": ("head_wid", "head_len"),
        "mandible_index (mandible_len/head_len)": ("mandible_len", "head_len"),
        "scape_index (seg_an_1/head_len)": ("seg_an_1", "head_len"),
    }
    index_corr = {}
    print()
    for name, (a_, b_) in indices.items():
        idx = shape_index(a_, b_)
        if idx is None:
            print(f"  {name}: columns not found, skipped")
            continue
        r, p = stats.pearsonr(PC_combined[:, 0], idx)
        print(f"  combined PC1 vs {name}: r={r:.3f} (p={p:.4f})")
        index_corr[name] = dict(r=float(r), p=float(p))

    eco_k2 = eco_k
    if has_guild.sum() >= 5:
        print("\n  PC1 by guild (mean +- sd):")
        guild_stats = {}
        for g in sorted(guild_counts, key=lambda g: -guild_counts[g]):
            m = has_guild & (eco_k2 == g)
            print(f"    {g:<26} {PC_combined[m, 0].mean():+.3f} +- {PC_combined[m, 0].std():.3f}  (n={m.sum()})")
            guild_stats[g] = dict(mean=float(PC_combined[m, 0].mean()), sd=float(PC_combined[m, 0].std()), n=int(m.sum()))
    else:
        guild_stats = {}

    genus_means = {g: float(PC_combined[genus[keep] == g, 0].mean()) for g in set(gk)}
    top_genus_pc1 = sorted(genus_means.items(), key=lambda kv: -kv[1])[:8]
    bot_genus_pc1 = sorted(genus_means.items(), key=lambda kv: kv[1])[:8]
    print("\n  Genera with HIGHEST mean combined-PC1:")
    for g, v in top_genus_pc1:
        print(f"    {g:<20} {v:+.3f}")
    print("  Genera with LOWEST mean combined-PC1:")
    for g, v in bot_genus_pc1:
        print(f"    {g:<20} {v:+.3f}")

    results["D_axis_inspection"] = dict(
        top_loadings=top_loadings,
        proportions_block_loading_mass=prop_loading_mass,
        absolute_block_loading_mass=abs_loading_mass,
        pc1_vs_log_size=dict(r=float(r_size), p=float(p_size)),
        pc1_vs_weber_proxy_mesosoma_len=dict(r=float(r_wl), p=float(p_wl)),
        pc1_vs_shape_indices=index_corr,
        pc1_by_guild=guild_stats,
        pc1_top_genera=top_genus_pc1,
        pc1_bottom_genera=bot_genus_pc1,
    )

    is_size_axis = (
        abs_loading_mass > 0.3 and abs(r_size) > 0.3 and p_size < 0.05
    )
    print(f"\n  Descriptive verdict: combined PC1 {'DOES' if is_size_axis else 'does NOT cleanly'} "
          f"behave like a simple 'size axis' (needs BOTH substantial absolute-block loading mass "
          f"AND a significant, non-trivial correlation with log-size; naming it 'size' is not "
          f"assumed just because absolute features were added to the PCA).")
    results["D_axis_inspection"]["is_simple_size_axis"] = bool(is_size_axis)

    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nWrote {os.path.join(OUT_DIR, 'results.json')}")


if __name__ == "__main__":
    main()
