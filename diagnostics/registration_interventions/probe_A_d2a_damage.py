"""Extends Task 6's D2a sample-starvation probe (`diagnostics/registration_failure/
probe_d2a_sample_starvation.py`) to the two damage corpora (drop30/drop60), which the original
probe only ran over the pose sweep. Reuses every function unchanged (face_group, face_areas,
sample_counts) -- this is pure geometric sampling of the target .obj files, independent of any
fit, so it needs no GPU and answers Task 7 Metric D directly: under `--split_distal`'s grouping
(ti+ta+pt pooled into one 'distal' group per leg, matching `distal_groups` below exactly), how
many of the fitter's own n_sample=8000 area-weighted target points actually land on that group,
under damage.

Task 6's pose-sweep numbers (clean_pose0..pose35) are already in
diagnostics/registration_failure/out/d2a_sample_starvation.json and are NOT recomputed here --
this script only adds the two conditions that file is missing.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MOON = os.path.join(REPO, "diagnostics", "moonshot")
FAILDIR = os.path.join(REPO, "diagnostics", "registration_failure")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "diagnostics", "absolute_scale"))
sys.path.insert(0, FAILDIR)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import measure as ms  # noqa: E402
from pytorch3d.io import load_obj  # noqa: E402
from probe_d2a_sample_starvation import face_group, face_areas, sample_counts, N_SAMPLE, N_DRAWS  # noqa: E402


def main():
    M = ms.load_model()
    fgroup = face_group(M)
    faces = np.asarray(M["dd"]["f"])
    rng = np.random.default_rng(0)

    legs = [(k, s) for k in (1, 2, 3) for s in ("r", "l")]
    distal_groups = [f"l{k}d_{s}" for k, s in legs]
    proximal_groups = [f"l{k}p_{s}" for k, s in legs]

    corpora = [
        ("clean_drop30", "synth_clean_drop30"),
        ("clean_drop60", "synth_clean_drop60"),
    ]

    print(f"n_sample={N_SAMPLE}, {N_DRAWS} independent draws per specimen (damage corpora)\n")
    print(f"{'condition':<14}{'specimen':<12}{'distal(median)':>16}{'distal(p10)':>13}"
          f"{'P(distal==0)':>14}{'proximal(median)':>18}")
    results = {}
    for label, corpus in corpora:
        cdir = os.path.join(MOON, corpus)
        if not os.path.isdir(cdir):
            print(f"[skip] {corpus} not found")
            continue
        obj_files = sorted(f for f in os.listdir(cdir) if f.endswith(".obj"))
        cond_rows = []
        for fn in obj_files:
            ov, of, _ = load_obj(os.path.join(cdir, fn), load_textures=False)
            verts = ov.numpy()
            f_idx = of.verts_idx.numpy()
            if f_idx.shape != faces.shape:
                print(f"[warn] topology mismatch in {corpus}/{fn}: {f_idx.shape} vs {faces.shape} -- "
                      f"damage corpora may remesh; skipping template-indexed fgroup for this file")
                continue
            per_leg = sample_counts(verts, f_idx, fgroup, distal_groups + proximal_groups, N_SAMPLE, N_DRAWS, rng)
            all_distal = np.stack([per_leg[g] for g in distal_groups])
            all_prox = np.stack([per_leg[g] for g in proximal_groups])
            row = dict(
                specimen=fn,
                distal_median=float(np.median(all_distal)),
                distal_p10=float(np.percentile(all_distal, 10)),
                p_distal_zero=float((all_distal == 0).mean()),
                proximal_median=float(np.median(all_prox)),
                per_leg_distal_median={g: float(np.median(per_leg[g])) for g in distal_groups},
            )
            cond_rows.append(row)
            print(f"{label:<14}{fn:<12}{row['distal_median']:>16.1f}{row['distal_p10']:>13.1f}"
                  f"{row['p_distal_zero']:>14.3f}{row['proximal_median']:>18.1f}")
        results[label] = cond_rows

    import json
    OUT = os.path.join(HERE, "out")
    os.makedirs(OUT, exist_ok=True)
    json.dump(results, open(os.path.join(OUT, "d2a_sample_starvation_damage.json"), "w"), indent=1)
    print(f"\nwrote {OUT}/d2a_sample_starvation_damage.json")


if __name__ == "__main__":
    main()
