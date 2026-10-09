"""D3 -- is the pretarsus's missing joint-rotation limit (confirmed +-pi on all 3 axes, vs.
~20-40deg on other leg joints -- see PLAN/REPORT) actually being exploited by the optimizer, or
just theoretically available? Reads fitted `joint_rot` from each run's Stage_3_deform_fine.npz
and reports, per segment, what fraction of the joint's OWN rotation limit range each fitted
specimen's rotation actually uses (|rotation| / limit_half_range per axis, capped at 1.0 since
values can exceed a SOFT hinge penalty's nominal range).

D4 -- offset dilution: `w_offset` penalizes `deform_verts.pow(2).sum(-1).mean()`, a MEAN over all
~10235 vertices (trainer_hierarchical.py:559-562 / trainer_moonshot.py's Stage_2/3 analogue).
Reads `deform_verts` from the same runs and compares mean |offset| on distal-leg dominant
vertices against the corpus-wide mean the loss actually penalizes, per specimen -- directly
testing whether large individual displacements on the ~2-5% of vertices belonging to distal legs
are invisible to the mean-based penalty.
"""

import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MOON = os.path.join(REPO, "diagnostics", "moonshot")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "diagnostics", "absolute_scale"))
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import measure as ms  # noqa: E402


def joint_limits_by_name(M):
    with open(os.path.join(REPO, "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl"), "rb") as fh:
        u = pickle._Unpickler(fh)
        u.encoding = "latin1"
        dd = u.load()
    jn = [str(x) for x in dd["J_names"]]
    jl = np.asarray(dd["joint_limits"])  # (55, 3, 2)
    return {n: jl[i] for i, n in enumerate(jn)}


def d3_limit_engagement(M, jl_by_name, conditions):
    jn = M["jnames"]  # 55 names, joint_rot has 54 rows = joints[1:] (root excluded)
    out = {}
    for run, corpus in conditions:
        p = os.path.join(MOON, "runs", run, "Stage_3_deform_fine.npz")
        if not os.path.isfile(p):
            continue
        d = np.load(p)
        jr = d["joint_rot"]  # (n, 54, 3)
        rows = []
        for seg in ("co", "tr", "fe", "ti", "ta", "pt"):
            usage = []
            for k in (1, 2, 3):
                for side in ("r", "l"):
                    name = f"l_{k}_{seg}_{side}"
                    if name not in jn:
                        continue
                    j = jn.index(name)
                    row_idx = j - 1  # joint_rot excludes root (joint 0)
                    if row_idx < 0 or row_idx >= jr.shape[1]:
                        continue
                    lim = jl_by_name[name]  # (3,2)
                    half_range = (lim[:, 1] - lim[:, 0]) / 2.0
                    half_range = np.where(half_range > 1e-6, half_range, np.nan)
                    frac = np.abs(jr[:, row_idx, :]) / half_range  # (n, 3)
                    usage.append(frac)
            if usage:
                usage = np.concatenate(usage, axis=0)  # (n*legs*sides, 3)
                rows.append((seg, float(np.nanmedian(usage)), float(np.nanmean(usage > 0.9))))
        out[run] = rows
        print(f"\n{run} ({corpus})")
        print(f"  {'seg':<6}{'median frac-of-limit-range':>28}{'P(>90% of limit)':>20}")
        for seg, med, p90 in rows:
            print(f"  {seg:<6}{med:>28.3f}{p90:>20.3f}")
    return out


def d4_offset_dilution(M, conditions):
    dom = M["dominant"]
    jn = M["jnames"]

    def group_of_vertex(j):
        n = jn[j]
        if n.startswith("l_"):
            seg = n.split("_")[2]
            return "leg_distal" if seg in ("ti", "ta", "pt") else "leg_prox"
        return "other"

    vgroup = np.array([group_of_vertex(j) for j in dom])
    distal_mask = vgroup == "leg_distal"
    print(f"\ndistal-leg vertices: {distal_mask.sum()} / {len(dom)} ({100*distal_mask.mean():.1f}% of mesh)")

    out = {}
    for run, corpus in conditions:
        p = os.path.join(MOON, "runs", run, "Stage_3_deform_fine.npz")
        if not os.path.isfile(p):
            continue
        d = np.load(p)
        off = d["deform_verts"]  # (n, V, 3)
        mag = np.linalg.norm(off, axis=-1)  # (n, V)
        corpus_mean = mag.mean(axis=1)  # (n,) -- what the mean-penalty actually sees
        distal_mean = mag[:, distal_mask].mean(axis=1)  # (n,)
        ratio = distal_mean / np.maximum(corpus_mean, 1e-12)
        out[run] = dict(
            corpus_mean_median=float(np.median(corpus_mean)),
            distal_mean_median=float(np.median(distal_mean)),
            ratio_median=float(np.median(ratio)),
        )
        print(f"{run:<24} corpus-wide mean|offset|={np.median(corpus_mean):.5f}  "
              f"distal-leg mean|offset|={np.median(distal_mean):.5f}  "
              f"distal/corpus ratio={np.median(ratio):.2f}x")
    return out


def main():
    M = ms.load_model()
    jl_by_name = joint_limits_by_name(M)
    conditions = [
        ("SYN_clean_pose0_w5", "synth_clean_pose0"),
        ("SYN_clean_w5", "synth_clean"),
        ("SYN_clean_drop30_w5", "synth_clean_drop30"),
        ("SYN_clean_drop60_w5", "synth_clean_drop60"),
    ]
    print("=== D3: joint-limit engagement ===")
    d3 = d3_limit_engagement(M, jl_by_name, conditions)
    print("\n=== D4: offset dilution ===")
    d4 = d4_offset_dilution(M, conditions)

    OUT = os.path.join(HERE, "out")
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(d3=d3, d4=d4), open(os.path.join(OUT, "d3_d4.json"), "w"), indent=1)
    print(f"\nwrote {OUT}/d3_d4.json")


if __name__ == "__main__":
    main()
