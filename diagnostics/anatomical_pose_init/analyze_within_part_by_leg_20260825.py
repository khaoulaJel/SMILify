#!/usr/bin/env python3
"""Per-LEG-POSITION within-part (segment-level) accuracy, zero-init vs Dense_GT_oracle
(2026-08-25 falsifiable check, pre-registered before the dense-oracle result was seen):

Does l2 (mesothoracic) improve LESS than l1/l3 under perfect dense correspondence, despite
starting lower -- the geometric-resolution-limit signature (mesothoracic starvation, already
documented in geom_leg_init.py) -- or does l2 improve proportionally as much or more, which would
undercut that standing explanation and point at the correspondence MECHANISM instead?

`within_leg_segment_confusion` (confusion.py) already restricts to points whose leg-level match
is correct and internally splits its per-point segment lookup by leg identity -- it just pools
all legs into ONE accumulator before returning. This script calls it once per (specimen, leg) with
pre-filtered inputs, reusing it UNCHANGED, to get the per-leg breakdown it doesn't expose itself.

Uses a FRESH per-specimen rng(seed=1) for each run being compared, so the SAME target point
sample is used for both zero-init's and Dense_GT_oracle's fitted meshes at every specimen --
apples-to-apples, not an artifact of a shared/advancing rng across many unrelated runs the way
run_audit.py's own main() loop does.
"""
import os
import sys

import numpy as np
import torch
from pytorch3d.io import load_obj

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MOON = os.path.join(REPO, "diagnostics", "moonshot")
CORR = os.path.join(REPO, "diagnostics", "correspondence_accuracy")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "diagnostics", "morphometrics"))
sys.path.insert(0, CORR)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import measure as ms  # noqa: E402
import labels as lb  # noqa: E402
import confusion as cf  # noqa: E402

N_SAMPLE = 8000
DEVICE = "cpu"
CORPUS = "synth_clean"
RUNS = {"zero": "SYN_clean_zero_wsl", "dense": "Dense_GT_oracle"}


def per_leg_seg_acc_for_run(npz_path, spec_labels, template_faces, face_lab, vlabels):
    fitted = np.load(npz_path)["verts"].astype(np.float64)
    out = {}  # stem -> {leg_key: seg_acc}
    for i, spec_lab in enumerate(spec_labels):
        stem = spec_lab[:-4] if spec_lab.endswith(".obj") else spec_lab
        rng = np.random.default_rng(1)  # FRESH per specimen, so both runs see the same points
        obj_path = os.path.join(MOON, CORPUS, f"{stem}.obj")
        ov, of, _ = load_obj(obj_path, load_textures=False)
        tgt_verts, tgt_faces = ov.numpy(), of.verts_idx.numpy()
        pts, point_lab = cf.sample_true_labeled_points(tgt_verts, tgt_faces, face_lab, N_SAMPLE, rng)
        pts_t = torch.tensor(pts, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        true_is_leg = point_lab["leg_id"] != None  # noqa: E711
        fitted_i = torch.tensor(fitted[i], dtype=torch.float32, device=DEVICE).unsqueeze(0)

        leg_acc_i, leg_pred, pts_leg, true_leg_sub = cf.leg_confusion(
            point_lab["leg_id"], true_is_leg, pts_t, fitted_i, vlabels
        )
        true_seg_sub_all = point_lab["leg_seg"][true_is_leg]

        per_leg = {}
        for leg_key in sorted(set(true_leg_sub.tolist())):
            m = true_leg_sub == leg_key
            if m.sum() < 10:
                continue
            seg_acc_leg = cf.within_leg_segment_confusion(
                true_seg_sub_all[m], np.ones(int(m.sum()), dtype=bool),
                pts_leg[:, m, :], true_leg_sub[m], leg_pred[m], fitted_i, vlabels
            )
            d = seg_acc_leg.as_dict()
            if d["counts"] and sum(map(sum, d["counts"])) > 0:
                per_leg[leg_key] = d["accuracy"]
        out[stem] = per_leg
    return out


def main():
    M = ms.load_model()
    template_faces = np.asarray(M["dd"]["f"])
    vlabels = lb.vertex_labels(M["jnames"], M["dominant"])
    face_lab = lb.face_labels(template_faces, vlabels)

    d0 = np.load(os.path.join(MOON, "runs", RUNS["zero"], "Stage_3_deform_fine.npz"))
    spec_labels = [str(x) for x in d0["labels"]]

    results = {}
    for key, run in RUNS.items():
        npz_path = os.path.join(MOON, "runs", run, "Stage_3_deform_fine.npz")
        print(f"[analyze] {key} ({run}) ...")
        results[key] = per_leg_seg_acc_for_run(npz_path, spec_labels, template_faces, face_lab, vlabels)

    specimens = sorted(results["zero"].keys())
    all_legs = sorted({leg for s in specimens for leg in results["zero"][s]})

    print()
    print("=" * 90)
    print("PER-(specimen, leg) within-part (segment) accuracy: zero-init vs Dense_GT_oracle")
    print("=" * 90)
    by_leg_deltas = {leg: [] for leg in all_legs}
    for s in specimens:
        for leg in all_legs:
            z = results["zero"][s].get(leg)
            dns = results["dense"][s].get(leg)
            if z is None or dns is None:
                continue
            by_leg_deltas[leg].append((s, z, dns, dns - z))

    print(f"\n{'leg':8s} {'n':>4s} {'mean_zero':>10s} {'mean_dense':>11s} {'mean_delta':>11s}")
    leg_position = lambda leg: leg[1]  # 'l1_r' -> '1'
    by_position = {}
    for leg in all_legs:
        rows = by_leg_deltas[leg]
        if not rows:
            continue
        z = np.array([r[1] for r in rows])
        dns = np.array([r[2] for r in rows])
        delta = dns - z
        print(f"{leg:8s} {len(rows):4d} {z.mean():10.3f} {dns.mean():11.3f} {delta.mean():+11.3f}")
        pos = leg_position(leg)
        by_position.setdefault(pos, []).extend(delta.tolist())
        by_position.setdefault(pos + "_zero", []).extend(z.tolist())

    print()
    print("Collapsed by leg POSITION (l1=front, l2=mid/mesothoracic, l3=hind), left+right pooled:")
    for pos in ("1", "2", "3"):
        d = np.array(by_position.get(pos, []))
        z = np.array(by_position.get(pos + "_zero", []))
        if len(d) == 0:
            continue
        print(f"  l{pos}: n={len(d):3d}  mean_zero={z.mean():.3f}  mean_delta={d.mean():+.3f}  "
              f"(room to improve: {1 - z.mean():.3f})")

    d2 = np.array(by_position.get("2", []))
    d1 = np.array(by_position.get("1", []))
    d3 = np.array(by_position.get("3", []))
    d13 = np.concatenate([d1, d3])
    print()
    print(f"FALSIFIABLE CHECK: l2 mean_delta={d2.mean():+.3f} vs l1+l3 mean_delta={d13.mean():+.3f}")
    if d2.mean() < 0.5 * d13.mean() and d13.mean() > 0:
        print("  -> l2 improved LESS THAN HALF as much as l1/l3: consistent with a geometric/data-"
              "sparsity ceiling (mesothoracic starvation), not a correspondence-mechanism limit.")
    elif d2.mean() >= d13.mean():
        print("  -> l2 improved AS MUCH OR MORE than l1/l3: undercuts the standing mesothoracic-"
              "starvation explanation -- correspondence mechanism, not point density, was limiting.")
    else:
        print("  -> in between: neither pattern is clean-cut.")


if __name__ == "__main__":
    main()
