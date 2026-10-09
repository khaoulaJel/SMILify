"""Shared, frame-CORRECTED loader for the R6/R7/R8 re-scoring (2026-09-16).

The void originals (`diagnostics/groundtruth/r{6,7,8}_*.py`) read `landmarks[k]["original"]` RAW.
That field is stored in the annotation scene's Blender Z-up frame; the scan `.obj` frame is
`obj = (x, z, -y)` of the stored value -- the canonical correction at
`diagnostics/joint_alignment_benchmark/tools/gt_registration.py:143`.

Measured here (probe, all 12 annotated specimens, min distance to the scan surface in % of the
mesh bounding-box diagonal):

    corrected   median 0.057 - 0.203 % per specimen   (worst single landmark 0.802 %)
    raw         median 0.308 - 3.099 % per specimen

so `assert median < 0.5 %` genuinely discriminates the two frames.

Specimen set: the 12 expert specimens (`annotation/gt_expert/`), minus Dolichoderus, which was
annotated in a scene whose mesh IS the Discothyrea scan (JAB registration audit: tier EXCLUDED,
residual 8e-9).  n = 11.  `gt_batch1` is superseded and is not touched.

On the four MIRRORED scenes (Aenictus, Aphaenogaster, Leptogenys, Odontomachus in the JAB audit):
the mirror + `_r`/`_l` swap is a property of the JOINT annotation scenes.  The surface landmarks
land on the scan surface after `fix` alone on all four (0.057-0.132 % median), so no reflection is
applied to them here.  All three scored outcomes below are in any case invariant to an `_r`/`_l`
relabelling (they ask whether two structures are DISTINCT / straddle, never which one is right).

Expert JOINTS, where needed (R7), are taken from the JAB registration
(`joint_alignment_benchmark/data/gt_joints_fitframe.json`, obj frame, mirror and label swap already
applied, residual <= 7e-8) instead of the fit-derived `annotation/joint_to_obj_transform.json`.
"""
import glob
import json
import os

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MESHDIR = "/hpcwork/nao48500/worker_alt_data"
JAB_GT = os.path.join(REPO, "diagnostics/joint_alignment_benchmark/data/gt_joints_fitframe.json")

EXCLUDE = {"Dolichoderus_cf.bidens_CASENT0744033":
           "annotated on the Discothyrea mesh (JAB audit tier EXCLUDED, resid 8e-9)"}
CAVEAT = {"Cephalotes_atratus_CASENT0744612":
          "joints annotated on the older bench_10 mesh (JAB tier CHAIN); its trait landmarks "
          "nevertheless sit 0.184% diag off the fit-target scan, so they are used"}

ON_SURFACE_BAR = 0.5   # % of mesh diagonal

fix = lambda v: np.array([v[0], v[2], -v[1]], float)   # noqa: E731  Blender Z-up -> .obj


def _norm_sid(s):
    s = s.replace("_EDITED_MARKERS", "").replace("_edited", "").replace("_joints", "")
    h = len(s) // 2
    if len(s) % 2 == 0 and s[:h] == s[h:]:
        s = s[:h]
    return s


def expert_specimens():
    """The gt_expert specimen ids, minus the excluded ones."""
    sids = sorted({_norm_sid(os.path.basename(p)[:-5])
                   for p in glob.glob(os.path.join(REPO, "annotation/gt_expert/*_joints.json"))})
    return [s for s in sids if s not in EXCLUDE]


def load_landmarks(verbose=True):
    """{sid: {landmark: xyz in .obj frame}} for the expert specimens, frame-corrected and asserted
    to lie on the scan surface."""
    out, report = {}, []
    for sid in expert_specimens():
        p = os.path.join(REPO, "annotation/landmarks", f"{sid}_traits.json")
        f = os.path.join(MESHDIR, f"{sid}_processed.obj")
        if not (os.path.exists(p) and os.path.exists(f)):
            print(f"[skip] {sid}: missing landmarks or mesh")
            continue
        d = json.load(open(p))["landmarks"]
        o = np.asarray(trimesh.load(f, process=False).vertices, float)
        diag = float(np.linalg.norm(o.max(0) - o.min(0)))

        cor, raw, L = {}, {}, {}
        for k, v in d.items():
            pc, pr = fix(v["original"]), np.asarray(v["original"], float)
            cor[k] = float(np.linalg.norm(o - pc, axis=1).min()) / diag * 100.0
            raw[k] = float(np.linalg.norm(o - pr, axis=1).min()) / diag * 100.0
            L[k] = pc
        med_c, med_r = float(np.median(list(cor.values()))), float(np.median(list(raw.values())))
        # THE DISCRIMINATING ASSERT: after the frame fix the annotation must lie ON the scan.
        assert med_c < ON_SURFACE_BAR, \
            f"{sid}: corrected landmarks {med_c:.3f}% diag off-surface (bar {ON_SURFACE_BAR}%)"
        out[sid] = L
        report.append((sid, len(L), med_c, med_r))
        if verbose:
            print(f"[lm] {sid[:40]:42s} n={len(L):2d}  on-surface corrected {med_c:6.3f}%  "
                  f"raw {med_r:6.3f}%")
    if verbose:
        print(f"[lm] {len(out)} specimens | median on-surface: corrected "
              f"{np.median([r[2] for r in report]):.3f}%  raw {np.median([r[3] for r in report]):.3f}%")
        assert np.median([r[2] for r in report]) < ON_SURFACE_BAR
    return out


def load_expert_joints_obj():
    """{sid: {joint: xyz in .obj frame}} from the JAB registration (mirror/label swap handled)."""
    d = json.load(open(JAB_GT))
    return {sid: {j: np.asarray(v["obj"], float) for j, v in rec["joints"].items()}
            for sid, rec in d.items() if sid not in EXCLUDE}
