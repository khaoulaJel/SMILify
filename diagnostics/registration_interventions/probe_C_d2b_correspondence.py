"""Metric B/C for Intervention C. Same design as `probe_A_d2b_correspondence.py`: reuses Task 6's
`probe_d2b_correspondence_audit.py` functions UNCHANGED, auditing cross-leg confusion and
within-leg segment mismatch against a FIXED leg-granularity yardstick, regardless of which
mechanism (baseline, split_distal, or now stratified sampling) produced the fitted vertices.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
FAILDIR = os.path.join(REPO, "diagnostics", "registration_failure")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "diagnostics", "absolute_scale"))
sys.path.insert(0, FAILDIR)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import measure as ms  # noqa: E402
from fitter_3d.trainer_hierarchical import vertex_groups  # noqa: E402
from probe_d2b_correspondence_audit import audit_run, true_face_labels  # noqa: E402

OUT = os.path.join(HERE, "out")


def main():
    M = ms.load_model()
    weights = M["weights"]
    jnames = M["jnames"]
    faces = np.asarray(M["dd"]["f"])
    vg, group_names = vertex_groups(weights, jnames, split_distal=False)
    leg_of_face, seg_of_face = true_face_labels(M)
    rng = np.random.default_rng(1)

    conditions = [
        ("SYN_clean_pose0_w5", "synth_clean_pose0"),
        ("SYN_clean_pose0_stratq05_w5", "synth_clean_pose0"),
        ("SYN_clean_pose0_stratq10_w5", "synth_clean_pose0"),
        ("SYN_clean_pose0_stratq20_w5", "synth_clean_pose0"),
        ("SYN_clean_w5", "synth_clean"),
        ("SYN_clean_pose25_stratq05_w5", "synth_clean"),
        ("SYN_clean_pose25_stratq10_w5", "synth_clean"),
        ("SYN_clean_pose25_stratq20_w5", "synth_clean"),
    ]

    MOON = os.path.join(REPO, "diagnostics", "moonshot")
    all_rows = []
    for run, corpus in conditions:
        p = os.path.join(MOON, "runs", run, "Stage_3_deform_fine.npz")
        if not os.path.isfile(p):
            print(f"[skip] {run} not found (not fit yet?)")
            continue
        rows = audit_run(run, corpus, M, vg, group_names, leg_of_face, seg_of_face, faces, rng)
        all_rows.extend(rows)

    print("\n=== summary: mean cross-leg confusion / within-leg mismatch, baseline vs each quota ===")
    by_run = {}
    for r in all_rows:
        by_run.setdefault(r["run"], []).append(r)
    print(f"{'run':<34}{'xleg_ti':>9}{'xleg_ta':>9}{'xleg_pt':>9}{'wl_ti':>9}{'wl_ta':>9}{'wl_pt':>9}")
    for run, _ in conditions:
        if run not in by_run:
            continue
        rs = by_run[run]
        xleg = [np.nanmean([r[f"cross_leg_{seg}"] for r in rs]) for seg in ("ti", "ta", "pt")]
        wl = [np.nanmean([r[f"within_leg_mismatch_{seg}"] for r in rs]) for seg in ("ti", "ta", "pt")]
        print(f"{run:<34}{xleg[0]:>9.3f}{xleg[1]:>9.3f}{xleg[2]:>9.3f}{wl[0]:>9.3f}{wl[1]:>9.3f}{wl[2]:>9.3f}")

    os.makedirs(OUT, exist_ok=True)
    json.dump(all_rows, open(os.path.join(OUT, "C_d2b_correspondence_audit.json"), "w"), indent=1)
    print(f"\nwrote {OUT}/C_d2b_correspondence_audit.json")


if __name__ == "__main__":
    main()
