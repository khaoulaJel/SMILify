"""TASK 1 Hard Rule item 4: visual renders for the outlier specimens flagged by
task1_hardrule_reanalysis_PROBE.py (largest |delta gaster-legs count| and
|delta fscore@0.01| between fit3d_results_all_baseline and fit3d_results_all_gentle).

Reuses diagnostics/moonshot/render_fits.py's lateral/anterior scatter-by-deform-
magnitude convention, but selects specific flagged specimens by name instead of
n evenly-spaced indices.
"""

import argparse
import glob
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from pytorch3d.io import load_obj  # noqa: E402
from pytorch3d.structures import Meshes  # noqa: E402
from pytorch3d.ops import sample_points_from_meshes  # noqa: E402

DEV = "cuda:0"

FLAGGED = [
    # largest |delta gaster-legs count|
    "Strumigenys_crassicornis_CASENT0744273_processed.obj",
    "Polyrhachis_dives_CASENT0878093_processed.obj",
    "Ectatomma_brunneum_CASENT0744575_processed.obj",
    # largest |delta fscore@0.01|
    "Heteroponera_panamensis_CASENT0744055_processed.obj",
    "Anochetus_risii_CASENT0877608_processed.obj",
    "Acanthostichus_aff.brevicornis_CASENT0744328_processed.obj",
]

RUNS = ["fit3d_results_all_baseline", "fit3d_results_all_gentle"]
MESH_DIR = "diagnostics/moonshot/bench50_clean"
OUT = "diagnostics/khaoula_review/out"


def norm(v):
    v = v - v.mean(0)
    return v / v.abs().max()


def main():
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(REPO, MESH_DIR, "*.obj")))
    name_to_idx = {os.path.basename(f): i for i, f in enumerate(files)}
    sel = [name_to_idx[n] for n in FLAGGED]

    tgt = {}
    for i in sel:
        v, f, _ = load_obj(files[i], load_textures=False)
        v = norm(v.to(DEV))
        m = Meshes(verts=[v], faces=[f.verts_idx.to(DEV)])
        tgt[i] = sample_points_from_meshes(m, 9000)[0].cpu().numpy()

    fits = {}
    for rd in RUNS:
        full = os.path.join(REPO, rd)
        npzs = sorted([p for p in glob.glob(os.path.join(full, "*.npz")) if "_batch_" not in p])
        d = np.load(npzs[-1], allow_pickle=True)
        fits[os.path.basename(rd)] = (d["verts"], d["faces"][0], d["deform_verts"])

    names = list(fits.keys())
    for view, (a, b), vname in [((0, 2), (0, 2), "lateral_XZ"), ((1, 2), (1, 2), "anterior_YZ")]:
        ncol = len(names) + 1
        fig, axes = plt.subplots(len(sel), ncol, figsize=(3.0 * ncol, 2.8 * len(sel)))
        axes = np.atleast_2d(axes)
        for r, i in enumerate(sel):
            axes[r, 0].scatter(tgt[i][:, a], tgt[i][:, b], s=0.4, alpha=0.4, c="#666", linewidths=0)
            axes[r, 0].set_aspect("equal")
            axes[r, 0].set_title("TARGET" if r == 0 else "", fontsize=9)
            axes[r, 0].set_ylabel(FLAGGED[r][:22], fontsize=6)
            axes[r, 0].tick_params(labelsize=5)
            for c, nm in enumerate(names):
                verts, faces, dv = fits[nm]
                V = verts[i]
                mag = np.linalg.norm(dv[i], axis=-1)
                ax = axes[r, c + 1]
                ax.scatter(tgt[i][:, a], tgt[i][:, b], s=0.4, alpha=0.15, c="#bbb", linewidths=0)
                ax.scatter(V[:, a], V[:, b], s=0.5, alpha=0.7, c=mag, cmap="inferno", vmin=0, vmax=0.06, linewidths=0)
                ax.set_aspect("equal")
                ax.tick_params(labelsize=5)
                if r == 0:
                    ax.set_title(nm, fontsize=8)
        fig.suptitle(
            f"{vname}: TASK1 outlier specimens (top |Δgaster-legs count| and |Δfscore@0.01|), "
            f"baseline vs gentle, coloured by |deform_verts|. Grey = target.",
            fontsize=10,
        )
        plt.tight_layout()
        p = os.path.join(REPO, OUT, f"task1_outliers_{vname}.png")
        fig.savefig(p, dpi=105)
        plt.close(fig)
        print("wrote", p)


if __name__ == "__main__":
    main()
