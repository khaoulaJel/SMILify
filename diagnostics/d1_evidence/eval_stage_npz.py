"""Evaluate a stock fitter_3d/optimise.py stage .npz with the moonshot metric suite.

optimise_moonshot.py has a built-in --eval pass (see run_eval() there); the stock
optimise.py entrypoint has no equivalent. This mirrors that same eval logic so Arm S
(stock) and Arm D (D1 hierarchical->moonshot) are scored with the identical metric
definitions from diagnostics/moonshot/metrics.py, ported from feature/registration_moonshot
for exactly this comparison (see diagnostics/d1_evidence/scorecard.md).

Usage:
  python -u diagnostics/d1_evidence/eval_stage_npz.py \
      --npz <results_dir>/Stage_3_deform_fine.npz \
      --mesh_dir <MESH_DIR> \
      --out_csv diagnostics/d1_evidence/metrics_stock.csv
"""

import argparse
import csv
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "moonshot")))

import config  # noqa: E402
from fitter_3d.utils import load_meshes  # noqa: E402
from fitter_3d.trainer import SMAL3DFitter  # noqa: E402
import metrics as M  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True, help="path to the stock stage .npz (e.g. Stage_3_deform_fine.npz)")
    p.add_argument("--mesh_dir", required=True)
    p.add_argument("--out_csv", required=True)
    return p


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    d = np.load(args.npz, allow_pickle=True)
    labels = list(d["labels"]) if "labels" in d else None
    n = d["betas"].shape[0]

    mesh_files = sorted(
        os.path.join(args.mesh_dir, f) for f in os.listdir(args.mesh_dir) if f.endswith(".obj")
    )
    mesh_names = [os.path.basename(f) for f in mesh_files]
    if labels is not None:
        assert list(labels) == mesh_names, (
            f"npz labels {labels} do not match sorted mesh_dir listing {mesh_names}; "
            "eval requires the same ordering the optimiser used"
        )
    _, target_meshes = load_meshes(mesh_files=mesh_files, device=device)

    smal = SMAL3DFitter(batch_size=n, device=device, shape_family=-1)
    with torch.no_grad():
        for k in ["global_rot", "joint_rot", "betas", "log_beta_scales", "trans", "betas_trans", "deform_verts"]:
            if k in d:
                getattr(smal, k).data = torch.tensor(d[k], dtype=torch.float32, device=device)

    with open(config.SMAL_FILE, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        dd = u.load()
    sym_verts = torch.tensor(np.asarray(dd["sym_verts"]).astype(np.int64))
    weights = np.asarray(dd["weights"])
    jnames = list(dd["J_names"])
    part_ids, part_names = M.template_part_segmentation(weights, jnames)

    with torch.no_grad():
        pred = smal().detach()
        rest = smal(deform_verts=torch.zeros_like(smal.deform_verts)).detach()
        dv = smal.deform_verts.detach()
    faces = smal.faces[0].detach()

    rows = []
    for i, name in enumerate(mesh_names):
        tgt = target_meshes[i]
        r = M.evaluate(
            pred[i],
            faces,
            tgt,
            rest[i],
            sym_verts,
            part_ids,
            part_names,
            deform_verts=dv[i],
            device=device,
        )
        r["mesh"] = name
        rows.append(r)
        print(f"  eval {i + 1}/{len(mesh_names)}: {name}", flush=True)

    keys = ["mesh"] + [k for k in rows[0] if k != "mesh"]
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out_csv}", flush=True)


if __name__ == "__main__":
    main(build_parser().parse_args())
