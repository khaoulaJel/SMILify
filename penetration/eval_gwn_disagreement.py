"""TASK 3 deliverable: ship the validated GWN disagreement-rate diagnostic
(Step 6, scripts/penetration_joint_study/FINDINGS.md, r=0.947) into the moonshot
metric suite as a read-only, CPU-only eval script.

Runs on any completed run's saved stage .npz (same convention as eval_run.py):
for every specimen, every anatomically non-adjacent part pair, both directions,
reports the GWN-vs-proximity-test disagreement rate. No training, no loss
weights touched -- instrumentation only.

Usage:
  python eval_gwn_disagreement.py --run_dir fit3d_results_all_gentle \
      [--stage Stage_3_deform_fine] [--out <path>]

Output: one row per (specimen, part_a, part_b, direction), same key columns as
Stage.compute_eval_metrics()'s own *_per_pair_penetration.csv so the two can be
joined directly (e.g. to reproduce the Step-6 asymmetry-vs-count-delta
correlation without re-deriving anything).
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(__file__))

from joint_placement_common import winding_number  # noqa: E402
from gwn_disagreement import compute_specimen_gwn_disagreement, K_MARGIN_DEFAULT  # noqa: E402
from fitter_3d.trainer import SMAL3DFitter  # noqa: E402
from fitter_3d.part_groups import get_part_vertex_indices, get_non_adjacent_pairs, PART_GROUPS_COARSE  # noqa: E402
from fitter_3d.penetration_loss import _build_part_faces  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--stage", default=None, help="stage name; default = last stage npz in run_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--k_margin", type=float, default=K_MARGIN_DEFAULT)
    args = ap.parse_args()

    stage_files = sorted(p for p in glob.glob(os.path.join(args.run_dir, "*.npz")) if "_batch_" not in p)
    if not stage_files:
        raise SystemExit(f"no stage npz in {args.run_dir}")
    if args.stage:
        stage_files = [s for s in stage_files if os.path.basename(s).startswith(args.stage)]
    stage_file = stage_files[-1]
    stage_name = os.path.basename(stage_file).replace(".npz", "")

    fitter = SMAL3DFitter(batch_size=1, device="cpu", shape_family=-1)
    faces = fitter.faces[0].cpu().numpy()
    part_vertex_indices = get_part_vertex_indices(PART_GROUPS_COARSE)
    part_faces = _build_part_faces(faces, part_vertex_indices)
    all_pairs = get_non_adjacent_pairs(PART_GROUPS_COARSE)
    # Some PART_GROUPS_COARSE entries (e.g. "waist": no vertex's argmax skinning
    # weight ever lands there) own zero vertices/faces -- degenerate, not
    # checkable in either direction. penetration_loss.py's own training/eval path
    # silently no-ops on these (n_query forced to 0 inside _directional_penalty's
    # empty-input branch); dropping them here keeps this CSV's pair set identical
    # to Stage.compute_eval_metrics()'s *_per_pair_penetration.csv (10 pairs, no
    # "waist" rows) rather than emitting meaningless None/0.0 rows.
    degenerate = {p for p, idx in part_vertex_indices.items() if len(idx) == 0}
    non_adjacent_pairs = [(a, b) for a, b in all_pairs if a not in degenerate and b not in degenerate]
    if degenerate:
        print(f"skipping degenerate parts (0 vertices): {sorted(degenerate)} "
              f"-- {len(all_pairs) - len(non_adjacent_pairs)}/{len(all_pairs)} pairs dropped")

    d = np.load(stage_file, allow_pickle=True)
    labels = list(d["labels"])
    all_verts = d["verts"]

    out_path = args.out or os.path.join(args.run_dir, f"{stage_name}_gwn_disagreement.csv")
    rows_out = []
    for i, spec in enumerate(labels):
        rows = compute_specimen_gwn_disagreement(
            all_verts[i], faces, part_vertex_indices, part_faces, non_adjacent_pairs,
            winding_number_fn=winding_number, k_margin=args.k_margin,
        )
        for r in rows:
            r2 = {"stage": stage_name, "specimen": spec, **r}
            rows_out.append(r2)
        print(f"[{i + 1}/{len(labels)}] {spec}: "
              f"mean disagreement_rate over trustworthy pairs = "
              f"{np.mean([r['disagreement_rate'] for r in rows if r['disagreement_rate'] is not None]):.4f}",
              flush=True)

    keys = ["stage", "specimen", "part_a", "part_b", "direction",
             "disagreement_rate", "n_trustworthy", "n_query", "trustworthy_frac"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nwrote {out_path}")

    valid = [r for r in rows_out if r["disagreement_rate"] is not None]
    overall_trust_frac = np.mean([r["trustworthy_frac"] for r in rows_out])
    print(f"\n=== SUMMARY ({len(labels)} specimens, {len(non_adjacent_pairs)} pairs x 2 directions) ===")
    print(f"overall mean trustworthy_frac: {overall_trust_frac:.4f}")
    print(f"rows with >=1 trustworthy query vertex: {len(valid)}/{len(rows_out)}")
    print("\nmean disagreement_rate by pair/direction (population level, not per-specimen):")
    by_pd = {}
    for r in valid:
        key = (r["part_a"], r["part_b"], r["direction"])
        by_pd.setdefault(key, []).append(r["disagreement_rate"])
    for key, vals in sorted(by_pd.items(), key=lambda kv: -np.mean(kv[1])):
        print(f"  {key[0]:<10} {key[2]:<10} {key[1]:<10} mean={np.mean(vals):.4f}  n_specimens={len(vals)}")


if __name__ == "__main__":
    main()
