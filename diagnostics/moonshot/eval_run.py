"""Evaluate any completed run (baseline or moonshot) from its saved stage .npz files.

Works off the saved `verts` so it can score runs produced by either optimise.py or
optimise_moonshot.py, and can score EVERY stage of a run -- which is what exposes the
'later stages improve chamfer while destroying correspondence' pattern.

Usage:
  python eval_run.py --run_dir runs/baseline --mesh_dir bench50 [--stage Stage_3_deform_fine]
"""

import argparse
import csv
import glob
import os
import pickle
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(__file__))

import config  # noqa: E402
import metrics as M  # noqa: E402
from fitter_3d.utils import load_meshes  # noqa: E402

DEV = "cuda:0"


def _gwn_setup():
    """Lazy import + one-time precompute for --gwn_disagreement (TASK 3): the
    validated GWN-vs-proximity disagreement instrument (Step 6, FINDINGS.md,
    r=0.947), wired in here so it sits alongside deform_mag/edge_logratio in
    the same per-run report instead of living only as a standalone script.
    CPU-only; opt-in because it costs ~30s/specimen vs. this script's default
    ~1s/specimen, not because it needs anything eval_run.py doesn't already have."""
    from joint_placement_common import winding_number
    from gwn_disagreement import compute_specimen_gwn_disagreement
    from fitter_3d.trainer import SMAL3DFitter
    from fitter_3d.part_groups import get_part_vertex_indices, get_non_adjacent_pairs, PART_GROUPS_COARSE
    from fitter_3d.penetration_loss import _build_part_faces

    fitter = SMAL3DFitter(batch_size=1, device="cpu", shape_family=-1)
    faces_np = fitter.faces[0].cpu().numpy()
    part_vertex_indices = get_part_vertex_indices(PART_GROUPS_COARSE)
    part_faces = _build_part_faces(faces_np, part_vertex_indices)
    all_pairs = get_non_adjacent_pairs(PART_GROUPS_COARSE)
    degenerate = {p for p, idx in part_vertex_indices.items() if len(idx) == 0}
    pairs = [(a, b) for a, b in all_pairs if a not in degenerate and b not in degenerate]
    return dict(
        winding_number_fn=winding_number,
        compute_fn=compute_specimen_gwn_disagreement,
        part_vertex_indices=part_vertex_indices,
        part_faces=part_faces,
        pairs=pairs,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--mesh_dir", required=True)
    ap.add_argument("--stage", default=None, help="stage name; default = last stage")
    ap.add_argument("--all_stages", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n_points", type=int, default=30000)
    ap.add_argument("--gwn_disagreement", action="store_true",
                     help="TASK 3: also compute mean GWN-vs-proximity disagreement rate per "
                          "specimen (adds gwn_disagreement_mean column). CPU-only, ~30s/specimen "
                          "extra -- opt-in, off by default.")
    args = ap.parse_args()

    gwn_ctx = _gwn_setup() if args.gwn_disagreement else None

    with open(os.path.join(REPO, config.SMAL_FILE), "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        dd = u.load()
    sym_verts = torch.tensor(np.asarray(dd["sym_verts"]).astype(np.int64))
    part_ids, part_names = M.template_part_segmentation(np.asarray(dd["weights"]), list(dd["J_names"]))

    mesh_files = sorted(glob.glob(os.path.join(args.mesh_dir, "*.obj")))
    names = [os.path.basename(p) for p in mesh_files]
    print(f"loading {len(mesh_files)} target meshes...", flush=True)
    _, targets = load_meshes(mesh_files=mesh_files, device=DEV)

    # baseline runs name stages 'Stage_*'; moonshot configs use 'A_init', 'B_pose_coarse',
    # ... so glob everything and filter, keeping alphabetical order (which is also
    # execution order for both naming schemes).
    stage_files = sorted(glob.glob(os.path.join(args.run_dir, "*.npz")))
    stage_files = [s for s in stage_files if "_batch_" not in s]
    if not stage_files:
        raise SystemExit(f"no stage npz in {args.run_dir}")
    if args.stage:
        stage_files = [s for s in stage_files if os.path.basename(s).startswith(args.stage)]
    elif not args.all_stages:
        stage_files = [stage_files[-1]]

    all_rows = []
    for sf in stage_files:
        sname = os.path.basename(sf).replace(".npz", "")
        d = np.load(sf, allow_pickle=True)
        verts = torch.tensor(d["verts"], dtype=torch.float32, device=DEV)
        faces = torch.tensor(d["faces"][0].astype(np.int64), dtype=torch.int64, device=DEV)
        dv = torch.tensor(d["deform_verts"], dtype=torch.float32, device=DEV)
        # rest geometry = fit WITHOUT the free-form offsets, recovered by subtraction
        rest = verts - dv

        print(f"\n=== {sname} ===", flush=True)
        rows = []
        for i in range(len(names)):
            r = M.evaluate(
                verts[i],
                faces,
                targets[i],
                rest[i],
                sym_verts,
                part_ids,
                part_names,
                deform_verts=dv[i],
                device=DEV,
                n_points=args.n_points,
            )
            if gwn_ctx is not None:
                verts_np = verts[i].detach().cpu().numpy()
                faces_np = faces.detach().cpu().numpy()
                pair_rows = gwn_ctx["compute_fn"](
                    verts_np, faces_np, gwn_ctx["part_vertex_indices"], gwn_ctx["part_faces"],
                    gwn_ctx["pairs"], winding_number_fn=gwn_ctx["winding_number_fn"],
                )
                valid_rates = [pr["disagreement_rate"] for pr in pair_rows if pr["disagreement_rate"] is not None]
                r["gwn_disagreement_mean"] = float(np.mean(valid_rates)) if valid_rates else None
                r["gwn_disagreement_n_pairs_valid"] = len(valid_rates)

            r["mesh"] = names[i]
            r["stage"] = sname
            rows.append(r)
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(names)}", flush=True)
        all_rows += rows

        def m(k):
            return float(np.mean([x[k] for x in rows]))

        print(f"  chamfer_l2               {m('chamfer_l2'):.6f}")
        print(f"  fscore@0.01              {m('fscore@0.01'):.4f}")
        print(f"  fscore@0.02              {m('fscore@0.02'):.4f}")
        print(f"  hausdorff_95             {m('hausdorff_95'):.4f}")
        print(f"  normal_consistency       {m('normal_consistency'):.4f}")
        print(f"  edge_logratio_absmean    {m('edge_logratio_absmean'):.4f}")
        print(f"  tri_quality_mean         {m('tri_quality_mean'):.4f}")
        print(f"  deform_mag_mean          {m('deform_mag_mean'):.5f}")
        print(f"  midline_dev_mean         {m('midline_dev_mean'):.5f}")
        if gwn_ctx is not None:
            valid = [x["gwn_disagreement_mean"] for x in rows if x["gwn_disagreement_mean"] is not None]
            print(f"  gwn_disagreement_mean    {np.mean(valid):.4f}  ({len(valid)}/{len(rows)} specimens valid)")
        for p in part_names:
            k = f"part_{p}_dist_mean"
            if k in rows[0]:
                print(f"  part {p:<12} dist    {m(k):.5f}")

    out = args.out or os.path.join(args.run_dir, "metrics.csv")
    keys = ["mesh", "stage"] + [k for k in all_rows[0] if k not in ("mesh", "stage")]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
