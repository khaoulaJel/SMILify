"""Per-specimen characterization of Intervention C (Task 7), pose25 primary + pose0 for contrast.

Answers, per specimen, not just cohort-level:
  - does leg_distal joint error improve/regress under each quota?
  - does antenna joint error move in the SAME specimens?
  - does q20 rescue the specimens that were catastrophic at baseline, or just shift the mean?
  - paired bootstrap CIs on the mean leg_distal error reduction (formalizing the dose-response).
Joins with the already-computed correspondence audit (C_d2b_correspondence_audit.json) for
per-specimen cross-leg/within-leg numbers.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MOON = os.path.join(REPO, "diagnostics", "moonshot")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "diagnostics", "absolute_scale"))
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import measure as ms  # noqa: E402
from pytorch3d.io import load_obj  # noqa: E402

OUT = os.path.join(HERE, "out")
N_BOOT = 5000
RNG = np.random.default_rng(0)


def group_of(name):
    if name.startswith("l_"):
        seg = name.split("_")[2]
        return "leg_distal" if seg in ("ti", "ta", "pt") else "leg_prox"
    if name.startswith("an_"):
        return "antenna"
    return "other"


def per_specimen_errors(run, corpus, M):
    """Returns dict: specimen -> {group: mean joint-position error over that group's joints}."""
    Jr = M["Jr"]
    jn = M["jnames"]
    d = np.load(os.path.join(MOON, "runs", run, "Stage_3_deform_fine.npz"))
    labels = [str(x) for x in d["labels"]]
    fitted_verts = d["verts"].astype(np.float64)

    gt = np.load(os.path.join(MOON, corpus, "ground_truth.npz"))
    gtv_all, names = gt["verts"], [str(x) for x in gt["names"]]

    gt_verts = []
    for lab in labels:
        stem = lab[:-4] if lab.endswith(".obj") else lab
        ov, _, _ = load_obj(os.path.join(MOON, corpus, f"{stem}.obj"), load_textures=False)
        ov = ov.numpy()
        c = ov.mean(0)
        gt_verts.append((gtv_all[names.index(stem)] - c) / np.abs(ov - c).max())
    gt_verts = np.stack(gt_verts)

    J_fit = ms.joints(fitted_verts, Jr)
    J_gt = ms.joints(gt_verts, Jr)
    err = np.linalg.norm(J_fit - J_gt, axis=-1)  # (n, J)

    groups = {j: group_of(jn[j]) for j in range(len(jn))}
    out = {}
    for i, lab in enumerate(labels):
        stem = lab[:-4] if lab.endswith(".obj") else lab
        row = {}
        for g in ("leg_distal", "leg_prox", "antenna"):
            idx = [j for j in groups if groups[j] == g]
            row[g] = float(err[i, idx].mean())
        out[stem] = row
    return out


def paired_bootstrap_ci(baseline, arm, n_boot=N_BOOT):
    """Mean(baseline - arm) with a percentile bootstrap CI over specimens. Negative delta_key
    convention: positive value here means arm has LOWER error than baseline (an improvement)."""
    specimens = sorted(set(baseline) & set(arm))
    diffs = np.array([baseline[s] - arm[s] for s in specimens])
    boot = np.array([RNG.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(mean_reduction=float(diffs.mean()), ci95=(float(lo), float(hi)), n=len(specimens))


def analyze_condition(label, corpus, base_run, arm_runs, M):
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    base = per_specimen_errors(base_run, corpus, M)
    results = {"baseline": base}
    for qlabel, run in arm_runs.items():
        results[qlabel] = per_specimen_errors(run, corpus, M)

    specimens = sorted(base.keys())
    print(f"\n--- per-specimen leg_distal error, baseline vs quotas ---")
    header = f"{'specimen':<12}{'baseline':>10}" + "".join(f"{q:>10}" for q in arm_runs)
    print(header)
    for s in specimens:
        row = f"{s:<12}{base[s]['leg_distal']:>10.4f}"
        for qlabel in arm_runs:
            row += f"{results[qlabel][s]['leg_distal']:>10.4f}"
        print(row)

    # catastrophic-rescue check: top-4 (of 12) worst baseline specimens vs the rest
    sorted_by_base = sorted(specimens, key=lambda s: -base[s]["leg_distal"])
    catastrophic = set(sorted_by_base[:4])
    noncatastrophic = set(specimens) - catastrophic
    print(f"\ncatastrophic (top-4 baseline leg_distal error): {sorted(catastrophic)}")
    for qlabel in arm_runs:
        arm = results[qlabel]
        cat_delta = np.mean([base[s]["leg_distal"] - arm[s]["leg_distal"] for s in catastrophic])
        non_delta = np.mean([base[s]["leg_distal"] - arm[s]["leg_distal"] for s in noncatastrophic])
        print(f"  {qlabel}: mean error REDUCTION, catastrophic={cat_delta:.4f}  non-catastrophic={non_delta:.4f}")

    # antenna vs leg_distal per-specimen co-movement
    print(f"\n--- antenna vs leg_distal error co-movement (per specimen delta = baseline - arm) ---")
    for qlabel in arm_runs:
        arm = results[qlabel]
        d_distal = np.array([base[s]["leg_distal"] - arm[s]["leg_distal"] for s in specimens])
        d_antenna = np.array([base[s]["antenna"] - arm[s]["antenna"] for s in specimens])
        r = float(np.corrcoef(d_distal, d_antenna)[0, 1]) if d_distal.std() > 1e-12 and d_antenna.std() > 1e-12 else float("nan")
        print(f"  {qlabel}: corr(distal_improvement, antenna_improvement) across specimens = {r:.3f}")
        n_both_improve = int(((d_distal > 0) & (d_antenna > 0)).sum())
        n_distal_improve_antenna_worse = int(((d_distal > 0) & (d_antenna < 0)).sum())
        print(f"    specimens where BOTH improve: {n_both_improve}/12; "
              f"distal improves but antenna WORSENS: {n_distal_improve_antenna_worse}/12")

    # paired bootstrap CIs
    print(f"\n--- paired bootstrap 95% CI, mean leg_distal error reduction (baseline - arm) ---")
    boot_results = {}
    for qlabel in arm_runs:
        r = paired_bootstrap_ci(
            {s: base[s]["leg_distal"] for s in specimens},
            {s: results[qlabel][s]["leg_distal"] for s in specimens},
        )
        boot_results[qlabel] = r
        sig = "CI excludes 0" if (r["ci95"][0] > 0 or r["ci95"][1] < 0) else "CI INCLUDES 0 (not significant)"
        print(f"  {qlabel}: mean reduction={r['mean_reduction']:.4f}  95% CI=({r['ci95'][0]:.4f}, {r['ci95'][1]:.4f})  [{sig}]")

    return dict(base=base, arms=results, catastrophic=sorted(catastrophic), bootstrap=boot_results)


def main():
    os.makedirs(OUT, exist_ok=True)
    M = ms.load_model()

    pose25 = analyze_condition(
        "pose25 (primary -- where the dose-response effect is real)",
        "synth_clean",
        "SYN_clean_w5",
        {
            "q05": "SYN_clean_pose25_stratq05_w5",
            "q10": "SYN_clean_pose25_stratq10_w5",
            "q20": "SYN_clean_pose25_stratq20_w5",
        },
        M,
    )
    pose0 = analyze_condition(
        "pose0 (contrast -- effect runs backward here)",
        "synth_clean_pose0",
        "SYN_clean_pose0_w5",
        {
            "q05": "SYN_clean_pose0_stratq05_w5",
            "q10": "SYN_clean_pose0_stratq10_w5",
            "q20": "SYN_clean_pose0_stratq20_w5",
        },
        M,
    )

    def strip(res):
        return dict(base=res["base"], arms=res["arms"], catastrophic=res["catastrophic"], bootstrap=res["bootstrap"])

    json.dump(
        {"pose25": strip(pose25), "pose0": strip(pose0)},
        open(os.path.join(OUT, "C_per_specimen.json"), "w"),
        indent=1,
    )
    print(f"\nwrote {OUT}/C_per_specimen.json")


if __name__ == "__main__":
    main()
