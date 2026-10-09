"""Intervention B, PRE-REGISTERED validation (must run and be reported BEFORE any fitting run
uses the geometric initialiser -- see fitter_3d/geom_leg_init.py's LEAKAGE AUDIT).

Runs ONLY the H0_body stage (identical recipe/weights to the real baseline pipeline:
midline=2.0, beta_prior=0.0, limit=0.273, split_distal=False, body_its=900, GLOBAL
non-partitioned chamfer, exactly `optimise_hierarchical.py`'s H0) to get each specimen's
H0-fitted `global_rot`/`trans` -- the ONE piece of information both arms below are allowed to
share (see the module docstring). Everything downstream is then computed TWICE from that shared
H0 state:

  zero-init  : leg joint_rot = 0 (today's default, unmodified)
  geom-init  : leg joint_rot = fitter_3d.geom_leg_init.init_joint_rot_for_specimen(...)

and BOTH are scored against ground-truth joint positions (evaluation only -- ground truth is
never read by the initialiser itself, only by this script, after the initialiser has already
produced its estimate).

Hyperparameters (MIN_PTS_PER_BAND, BAND_PAD_FRAC) are the fixed constants in geom_leg_init.py,
chosen by geometric argument before this script was ever run, and are NOT adjusted based on
what this script reports.
"""

import json
import os
import sys

import numpy as np
import torch
from pytorch3d.io import load_obj
from pytorch3d.ops import sample_points_from_meshes

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MOON = os.path.join(REPO, "diagnostics", "moonshot")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "diagnostics", "absolute_scale"))
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import config  # noqa: E402
import measure as ms  # noqa: E402
from fitter_3d.utils import load_meshes  # noqa: E402
from fitter_3d.trainer import SMAL3DFitter  # noqa: E402
from fitter_3d.trainer_hierarchical import HierarchicalStage, vertex_groups  # noqa: E402
from fitter_3d.geom_leg_init import (  # noqa: E402
    leg_chains,
    rest_joint_positions,
    init_joint_rot_for_specimen,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_SAMPLE = 8000
BODY_ITS = 900
OUT = os.path.join(HERE, "out")

CONDITIONS = [
    ("pose0", "synth_clean_pose0"),
    ("pose25", "synth_clean"),
    ("drop30", "synth_clean_drop30"),
    ("drop60", "synth_clean_drop60"),
]


def run_h0(mesh_dir):
    import glob

    files = sorted(glob.glob(os.path.join(mesh_dir, "*.obj")))
    names = [os.path.basename(f) for f in files]
    _, targets = load_meshes(mesh_files=files, device=DEVICE)

    import pickle

    with open(config.SMAL_FILE, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        dd = u.load()
    jnames = list(dd["J_names"])
    vg, gnames = vertex_groups(np.asarray(dd["weights"]), jnames, split_distal=False, split_anterior=False)

    smal = SMAL3DFitter(batch_size=len(targets), device=DEVICE, shape_family=-1)
    ANTERIOR = {"head", "mandible", "antenna"}
    body_groups = ["body"] + [g for g in gnames if g in ANTERIOR]
    st = HierarchicalStage(
        "H0_body",
        BODY_ITS,
        smal=smal,
        target_meshes=targets,
        vertex_group=vg,
        group_names=gnames,
        joint_names=jnames,
        active_groups=body_groups,
        partitioned=False,
        lr=0.02,
        joint_lr=0.01,
        robust_kernel="gm",
        robust_scale=0.25,
        loss_weights={"w_edge": 0.05, "w_beta_prior": 0.0, "w_sym": 0.5, "w_midline": 2.0, "w_limit": 0.273},
        device=DEVICE,
        out_dir="/tmp",
    )
    st.run()
    return smal, targets, names, jnames, dd


def normalized_gt_verts(corpus, names, gt):
    gtv_all, gnames_gt, ext = gt["verts"], [str(x) for x in gt["names"]], gt["extent"]
    out = []
    for lab in names:
        stem = lab[:-4] if lab.endswith(".obj") else lab
        ov, _, _ = load_obj(os.path.join(MOON, corpus, f"{stem}.obj"), load_textures=False)
        ov = ov.numpy()
        c = ov.mean(0)
        out.append((gtv_all[gnames_gt.index(stem)] - c) / np.abs(ov - c).max())
    return np.stack(out)


def group_of(name):
    if name.startswith("l_"):
        seg = name.split("_")[2]
        return "leg_distal" if seg in ("ti", "ta", "pt") else "leg_prox"
    return "other"


def main():
    os.makedirs(OUT, exist_ok=True)
    all_results = {}
    for label, corpus in CONDITIONS:
        print(f"\n===== {label} ({corpus}) =====", flush=True)
        mesh_dir = os.path.join(MOON, corpus)
        smal, targets, names, jnames, dd = run_h0(mesh_dir)
        M = ms.load_model()
        Jr = M["Jr"]
        rest_J = rest_joint_positions(dd["J_regressor"], dd["v_template"]).to(DEVICE)
        chains = leg_chains(jnames)

        gt = np.load(os.path.join(mesh_dir, "ground_truth.npz"))
        gt_verts = normalized_gt_verts(corpus, names, gt)
        J_gt = ms.joints(gt_verts, Jr)  # (n, J, 3)

        tgt_pts = sample_points_from_meshes(targets, N_SAMPLE).detach()  # (n, N_SAMPLE, 3)

        B = len(names)
        n_pose = len(jnames) - 1
        zero_jr = torch.zeros(B, n_pose, 3, device=DEVICE)
        geom_jr = torch.zeros(B, n_pose, 3, device=DEVICE)
        for b in range(B):
            geom_jr[b] = init_joint_rot_for_specimen(
                smal.global_rot[b].detach(), smal.trans[b].detach(), rest_J, jnames, tgt_pts[b]
            )

        with torch.no_grad():
            verts_zero = smal(joint_rot=zero_jr, deform_verts=torch.zeros_like(smal.deform_verts)).cpu().numpy()
            verts_geom = smal(joint_rot=geom_jr, deform_verts=torch.zeros_like(smal.deform_verts)).cpu().numpy()
        J_zero = ms.joints(verts_zero.astype(np.float64), Jr)
        J_geom = ms.joints(verts_geom.astype(np.float64), Jr)

        err_zero = np.linalg.norm(J_zero - J_gt, axis=-1)  # (n, J)
        err_geom = np.linalg.norm(J_geom - J_gt, axis=-1)

        rows = []
        for j, name in enumerate(jnames):
            g = group_of(name)
            if g == "other":
                continue
            rows.append(
                dict(
                    joint=name,
                    group=g,
                    median_err_zero=float(np.median(err_zero[:, j])),
                    median_err_geom=float(np.median(err_geom[:, j])),
                )
            )
        print(f"{'joint':<12}{'group':<12}{'zero-init':>12}{'geom-init':>12}{'delta':>10}")
        for r in rows:
            delta = r["median_err_geom"] - r["median_err_zero"]
            print(f"{r['joint']:<12}{r['group']:<12}{r['median_err_zero']:>12.5f}{r['median_err_geom']:>12.5f}{delta:>10.5f}")

        summary = {}
        for g in ("leg_prox", "leg_distal"):
            m = [r for r in rows if r["group"] == g]
            summary[g] = dict(
                zero=float(np.median([r["median_err_zero"] for r in m])),
                geom=float(np.median([r["median_err_geom"] for r in m])),
            )
        print(f"\n{label} summary: leg_prox zero={summary['leg_prox']['zero']:.5f} "
              f"geom={summary['leg_prox']['geom']:.5f} | "
              f"leg_distal zero={summary['leg_distal']['zero']:.5f} geom={summary['leg_distal']['geom']:.5f}")

        # per-specimen leg_distal error (for the initial->final and paired-specimen plots later)
        distal_joint_idx = [j for j, n in enumerate(jnames) if group_of(n) == "leg_distal"]
        per_spec = dict(
            specimen=names,
            leg_distal_err_zero=np.median(err_zero[:, distal_joint_idx], axis=1).tolist(),
            leg_distal_err_geom=np.median(err_geom[:, distal_joint_idx], axis=1).tolist(),
        )

        all_results[label] = dict(rows=rows, summary=summary, per_specimen=per_spec)
        del smal, targets
        torch.cuda.empty_cache()

    json.dump(all_results, open(os.path.join(OUT, "B0_geom_init_validation.json"), "w"), indent=1)
    print(f"\nwrote {OUT}/B0_geom_init_validation.json")


if __name__ == "__main__":
    main()
