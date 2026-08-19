"""Rank 1 (master prompt Section 9.5.1): GT-free candidate selection-score validation.

Gates family D (multi-start, Section 12/15) before any multi-start compute is spent. Question:
can a fit be chosen well WITHOUT ground truth, using only quantities the fitter/eval pipeline
already computes?

Zero new fitting jobs. Reuses:
  - diagnostics/moonshot/runs/<run>/metrics.csv        (per-specimen BLIND candidate scores --
    chamfer, deform magnitude, part-distance-to-nearest-GT-part proxies, edge distortion,
    midline deviation -- all computable without ground truth)
  - diagnostics/absolute_scale/measure.py               (measure_run / measure_verts: the SAME
    code path used by every prior reliability report in this project)
  - diagnostics/moonshot/synth_*/ground_truth.npz + *.obj (ground truth, for the LABEL only)

Two analyses:
  A. Pooled correlation. Every (run, specimen) pair with both a metrics.csv row and a ground
     truth match becomes one observation. Spearman-correlate each blind score against the
     specimen's true leg_distal relative error (mean |fit-gt|/|gt| over leg_distal bones).
  B. Oracle-vs-blind-selector gap, WITHIN specimen, across the "hypothesis sets" that already
     exist in this repo: for pose25, the SAME 12 specimens were independently fit 6 different
     ways (baseline w5, splitdistal, stratq05/10/20, gtinit); for pose0, 4 ways (baseline,
     splitdistal, stratq05/10/20). Within each specimen x hypothesis-set, does the fit with the
     BEST blind score also have the LOWEST true leg_distal error? Report:
       oracle_err    = min true error across hypotheses for that specimen
       selected_err  = true error of the hypothesis the best blind score would have picked
       baseline_err  = true error of the (always-available) baseline hypothesis
       recovered_frac = (baseline_err - selected_err) / (baseline_err - oracle_err)
     averaged over specimens, per blind score. This is the number the decision rule in Section
     9.5.1 is stated in terms of (>=60% -> family D viable; ~0% -> family D collapses).

Decision rule (verbatim from the master prompt, applied here, not altered):
  recovered_frac >= 0.60  -> multi-start viable as a production candidate
  recovered_frac ~= 0     -> family D collapses in its current form, priority shifts to B/C/E
"""

import csv
import json
import os
import re
import sys

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
ABS_SCALE = os.path.join(REPO, "diagnostics", "absolute_scale")
MOON = os.path.join(REPO, "diagnostics", "moonshot")
sys.path.insert(0, REPO)
sys.path.insert(0, ABS_SCALE)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import measure as ms  # noqa: E402

OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

# blind candidate scores available in metrics.csv, and the direction that means "worse" (so a
# rank correlation sign is interpretable the same way for every score: positive rho = higher
# score -> higher true error, i.e. the score is a usable badness indicator)
BLIND_SCORES = [
    "chamfer_l2",
    "deform_mag_mean",
    "deform_mag_p95",
    "edge_logratio_absmean",
    "midline_dev_excess",
    "part_leg_distal_dist_mean",
    "part_leg_distal_dist_p95",
]
# part_leg_distal_within_tau is a GOODNESS score (higher = better); flip sign for a uniform
# "higher = worse" convention before correlating.
GOODNESS_SCORES = ["part_leg_distal_within_tau"]

# run_name (dir under diagnostics/moonshot/runs) -> corpus dir under diagnostics/moonshot/
RUN_TO_CORPUS = {}


def infer_corpus(run_name):
    """SYN_<cond>_w<weight> -> synth_<cond'>, per the naming convention documented in this
    project's D5/Intervention-A/C sbatch scripts. pose25 has no explicit corpus suffix because
    `synth_clean` (no pose tag) IS the pose_scale=0.25 default corpus."""
    base = re.sub(r"_w\d+(_\d+)?$", "", run_name)
    base = re.sub(r"^SYN_", "", base)
    base = re.sub(r"_gtinit$", "", base)
    base = re.sub(r"_splitdistal$", "", base)
    base = re.sub(r"_stratq\d+$", "", base)
    if base == "clean_pose25" or base == "clean":
        return "synth_clean"
    if base == "noisy":
        return "synth_noisy"
    return f"synth_{base}"


def load_gt_verts(corpus_dir):
    from pytorch3d.io import load_obj

    gt = np.load(os.path.join(MOON, corpus_dir, "ground_truth.npz"))
    gtv_all, names = gt["verts"], [str(x) for x in gt["names"]]
    out = {}
    for i, nm in enumerate(names):
        objp = os.path.join(MOON, corpus_dir, f"{nm}.obj")
        if not os.path.isfile(objp):
            continue
        ov, _, _ = load_obj(objp, load_textures=False)
        ov = ov.numpy()
        c = ov.mean(0)
        out[nm] = (gtv_all[i] - c) / np.abs(ov - c).max()
    return out


def leg_distal_relerr_per_specimen(fit_rows, gt_rows_by_label, bones):
    """mean |fit-gt|/|gt| over leg_distal bone-length columns, per specimen row."""
    ld_bones = [b["name"] for b in bones if ms.block_of(b["name"]) == "leg_distal"] if hasattr(ms, "block_of") else None
    if ld_bones is None:
        from calib_features import block_of

        ld_bones = [b["name"] for b in bones if block_of(b["name"]) == "leg_distal"]
    out = {}
    for r in fit_rows:
        # fit labels carry the .obj suffix (from measure_run/npz "labels"); GT rows are keyed by
        # the bare specimen stem (ground_truth.npz "names") -- normalise before joining.
        stem = r["label"][:-4] if r["label"].endswith(".obj") else r["label"]
        g = gt_rows_by_label.get(stem)
        if g is None:
            continue
        errs = []
        for b in ld_bones:
            if b not in r or b not in g:
                continue
            gt_v = g[b]
            if abs(gt_v) < 1e-9:
                continue
            errs.append(abs(r[b] - gt_v) / abs(gt_v))
        if errs:
            out[r["label"]] = float(np.mean(errs))
    return out


def main():
    M = ms.load_model()
    bones = ms.bone_table(M)
    tpa = ms.template_part_axes(M)

    runs = sorted(
        d
        for d in os.listdir(os.path.join(MOON, "runs"))
        if os.path.isfile(os.path.join(MOON, "runs", d, "metrics.csv"))
    )
    print(f"found {len(runs)} runs with metrics.csv")

    gt_cache = {}
    pooled_rows = []  # list of dicts: run, corpus, label, blind scores..., true_relerr
    skipped = []

    for run in runs:
        corpus = infer_corpus(run)
        corpus_path = os.path.join(MOON, corpus)
        if not os.path.isdir(corpus_path) or not os.path.isfile(os.path.join(corpus_path, "ground_truth.npz")):
            skipped.append((run, corpus, "no matching corpus/ground_truth.npz"))
            continue

        # blind scores, per specimen, from metrics.csv
        with open(os.path.join(MOON, "runs", run, "metrics.csv")) as fh:
            metric_rows = {row["mesh"]: row for row in csv.DictReader(fh)}
        if not metric_rows:
            skipped.append((run, corpus, "empty metrics.csv"))
            continue

        # true fitted measurements (leg_distal bone lengths) via the SAME measure_run() code
        # path every prior reliability report in this project used
        fit_rows, _, _ = ms.measure_run(run, "synth", M, bones, tpa)
        if not fit_rows:
            skipped.append((run, corpus, "no npz found by measure_run"))
            continue

        if corpus not in gt_cache:
            gt_verts = load_gt_verts(corpus)
            labels = list(gt_verts.keys())
            gt_rows, _, _ = ms.measure_verts(
                np.stack([gt_verts[l] for l in labels]).astype(np.float64), labels, "synth", M, bones, tpa
            )
            gt_cache[corpus] = {r["label"]: r for r in gt_rows}
        gt_by_label = gt_cache[corpus]

        relerr = leg_distal_relerr_per_specimen(fit_rows, gt_by_label, bones)

        n_joined = 0
        for r in fit_rows:
            lab = r["label"]
            mrow = metric_rows.get(lab)
            if mrow is None or lab not in relerr:
                continue
            entry = dict(run=run, corpus=corpus, label=lab, true_leg_distal_relerr=relerr[lab])
            ok = True
            for k in BLIND_SCORES + GOODNESS_SCORES:
                v = mrow.get(k, "")
                try:
                    entry[k] = float(v)
                except ValueError:
                    ok = False
                    break
            if ok:
                pooled_rows.append(entry)
                n_joined += 1
        print(f"  {run:45s} corpus={corpus:24s} joined {n_joined}/{len(fit_rows)} specimens")

    print(f"\npooled N = {len(pooled_rows)} (run,specimen) observations across {len(set(r['run'] for r in pooled_rows))} runs")
    for run, corpus, why in skipped:
        print(f"  SKIPPED {run}: {why}")

    # ---------------- analysis A: pooled Spearman correlation ----------------
    y = np.array([r["true_leg_distal_relerr"] for r in pooled_rows])
    corrA = {}
    for k in BLIND_SCORES:
        x = np.array([r[k] for r in pooled_rows])
        rho, p = spearmanr(x, y)
        corrA[k] = dict(rho=float(rho), p=float(p), direction="higher=worse")
    for k in GOODNESS_SCORES:
        x = -np.array([r[k] for r in pooled_rows])  # flip: higher (of -score) = worse
        rho, p = spearmanr(x, y)
        corrA[k] = dict(rho=float(rho), p=float(p), direction="flipped (raw score higher=better)")

    print("\n=== Analysis A: pooled Spearman rho (blind score vs true leg_distal relerr), N=%d ===" % len(y))
    for k, v in sorted(corrA.items(), key=lambda kv: -abs(kv[1]["rho"])):
        print(f"  {k:32s} rho={v['rho']:+.3f}  p={v['p']:.4f}")

    # ---------------- analysis B: oracle-vs-blind-selector gap within hypothesis sets --------
    HYP_SETS = {
        "pose25": ["SYN_clean_w5", "SYN_clean_pose25_splitdistal_w5", "SYN_clean_pose25_stratq05_w5",
                   "SYN_clean_pose25_stratq10_w5", "SYN_clean_pose25_stratq20_w5", "SYN_clean_pose25_gtinit_w5"],
        "pose0": ["SYN_clean_pose0_w5", "SYN_clean_pose0_splitdistal_w5", "SYN_clean_pose0_stratq05_w5",
                  "SYN_clean_pose0_stratq10_w5", "SYN_clean_pose0_stratq20_w5"],
    }
    by_run_label = {}
    for r in pooled_rows:
        by_run_label.setdefault(r["run"], {})[r["label"]] = r

    resultsB = {}
    for setname, hyp_runs in HYP_SETS.items():
        present = [rn for rn in hyp_runs if rn in by_run_label]
        missing = [rn for rn in hyp_runs if rn not in by_run_label]
        if len(present) < 2:
            resultsB[setname] = dict(status="SKIPPED", reason=f"fewer than 2 hypothesis runs present, missing={missing}")
            continue
        baseline_run = hyp_runs[0]
        labels = set.intersection(*[set(by_run_label[rn].keys()) for rn in present])
        if not labels:
            resultsB[setname] = dict(status="SKIPPED", reason="no specimen label present in all hypothesis runs")
            continue
        per_score_recovered = {k: [] for k in BLIND_SCORES + GOODNESS_SCORES}
        n_specimens = 0
        for lab in sorted(labels):
            rows_h = [by_run_label[rn][lab] for rn in present]
            true_errs = [rh["true_leg_distal_relerr"] for rh in rows_h]
            oracle_err = min(true_errs)
            baseline_err = by_run_label[baseline_run][lab]["true_leg_distal_relerr"] if baseline_run in by_run_label and lab in by_run_label[baseline_run] else true_errs[0]
            if abs(baseline_err - oracle_err) < 1e-9:
                continue  # oracle can't beat baseline for this specimen; undefined recovered_frac
            n_specimens += 1
            for k in BLIND_SCORES:
                scores = [rh[k] for rh in rows_h]
                sel_idx = int(np.argmin(scores))  # lower blind score assumed better (all BLIND_SCORES are "higher=worse")
                sel_err = true_errs[sel_idx]
                per_score_recovered[k].append((baseline_err - sel_err) / (baseline_err - oracle_err))
            for k in GOODNESS_SCORES:
                scores = [rh[k] for rh in rows_h]
                sel_idx = int(np.argmax(scores))  # higher goodness score assumed better
                sel_err = true_errs[sel_idx]
                per_score_recovered[k].append((baseline_err - sel_err) / (baseline_err - oracle_err))
        resultsB[setname] = dict(
            status="OK",
            n_hypotheses=len(present),
            hypothesis_runs=present,
            missing_runs=missing,
            n_specimens_scored=n_specimens,
            recovered_frac_mean={k: (float(np.mean(v)) if v else None) for k, v in per_score_recovered.items()},
            recovered_frac_median={k: (float(np.median(v)) if v else None) for k, v in per_score_recovered.items()},
        )

    print("\n=== Analysis B: oracle-vs-blind-selector recovered_frac, per hypothesis set ===")
    for setname, res in resultsB.items():
        print(f"\n  hypothesis set: {setname}  status={res.get('status')}")
        if res.get("status") != "OK":
            print(f"    {res.get('reason')}")
            continue
        print(f"    hypotheses present: {res['hypothesis_runs']}  (missing: {res['missing_runs']})")
        print(f"    specimens scored (oracle beats baseline): {res['n_specimens_scored']}")
        for k, v in sorted(res["recovered_frac_mean"].items(), key=lambda kv: -(kv[1] if kv[1] is not None else -9)):
            vm = res["recovered_frac_median"][k]
            print(f"    {k:32s} recovered_frac mean={v}  median={vm}")

    # Combine across hypothesis sets per blind score, specimen-count-weighted, so one favourable
    # set cannot carry a PASS verdict that a second set contradicts (the master prompt's decision
    # rule is about whether a selector is USABLE, not whether it works somewhere).
    combined = {}
    ok_sets = [(setname, res) for setname, res in resultsB.items() if res.get("status") == "OK"]
    for k in BLIND_SCORES + GOODNESS_SCORES:
        num, den = 0.0, 0
        per_set = {}
        for setname, res in ok_sets:
            v = res["recovered_frac_mean"].get(k)
            n = res["n_specimens_scored"]
            per_set[setname] = v
            if v is not None:
                num += v * n
                den += n
        combined[k] = dict(weighted_mean=(num / den if den else None), per_set=per_set)

    best_k, best_v = None, -1e9
    for k, v in combined.items():
        if v["weighted_mean"] is not None and v["weighted_mean"] > best_v:
            best_k, best_v = k, v["weighted_mean"]

    # consistency check: does the best-by-combined score also recover decently in EVERY set on
    # its own, or does it only pass because one set's large positive value outweighs another
    # set's negative value? A selector that is negative in any set is not blind-deployable,
    # because at deployment time there is no way to know which regime (e.g. pose severity) a
    # given specimen is in.
    any_negative_for_best = any(
        (v is not None and v < 0) for v in combined.get(best_k, {}).get("per_set", {}).values()
    ) if best_k else True

    if best_v >= 0.60 and not any_negative_for_best:
        verdict = "PASS: family D (multi-start) is viable as a production candidate per the >=60% rule, consistently across hypothesis sets."
    elif best_v <= 0.05:
        verdict = "FAIL: family D collapses in its current form (~0% recovered); priority shifts to B/C/E."
    else:
        sign_note = (
            " Additionally, this figure is an artifact of averaging positive recovered_frac at one pose "
            "severity against strongly NEGATIVE recovered_frac (worse than picking baseline blind) at another -- "
            "no score tested is sign-consistent across regimes."
            if any_negative_for_best
            else ""
        )
        verdict = (
            f"MIXED / DO NOT PROMOTE: best combined recovered_frac={best_v:.3f} (score={best_k}) does not clear "
            f"the 60% threshold.{sign_note} Per the master prompt's decision rule this does not license "
            "promoting family D; treat multi-start evidence (Rank/family D) as a diagnostic result about the "
            "optimization landscape only (Section 6/H1), not a deployable selector, until a selector clears 60% "
            "with a consistent sign across regimes."
        )

    out = dict(
        n_runs_found=len(runs),
        n_pooled_observations=len(pooled_rows),
        skipped_runs=skipped,
        analysis_A_pooled_spearman=corrA,
        analysis_B_oracle_gap=resultsB,
        analysis_B_combined_across_sets=combined,
        best_blind_score=best_k,
        best_recovered_frac=best_v if best_v > -1e8 else None,
        gating_verdict=verdict,
    )
    with open(os.path.join(OUT, "rank1_selection_score.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n{verdict}")
    print(f"wrote {os.path.join(OUT, 'rank1_selection_score.json')}")


if __name__ == "__main__":
    main()
