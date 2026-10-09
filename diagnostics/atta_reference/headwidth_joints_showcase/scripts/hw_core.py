"""Shared core for the head-width reporter-bone showcase.

Reproduces EXACTLY what the SMIL Model Importer did for b_h_l / b_h_r:
  1. bone placed by hand on the Basis (template) mesh      -> rest position (trilaterated, exact)
  2. exporter attaches it to its 10 nearest Basis vertices -> inverse-distance weights (1/d, sum 1)
  3. on every registered specimen the joint is REGRESSED:  joint = sum_i w_i * v_i(specimen)
Validated against the Blender export CSV for the same npz (see validate_against_blender()).
"""
import os, sys, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SHOW = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(SHOW, "..", "..", ".."))
os.chdir(REPO)                       # config.py resolves the model path relative to the repo root
sys.path[:0] = [REPO, f"{REPO}/diagnostics/groundtruth", f"{REPO}/diagnostics/morphometrics",
                f"{REPO}/diagnostics/morphometric_validation"]
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")
os.environ.setdefault("SMILIFY_COUPLE_JOINT_BLENDSHAPES", "0")

from measure import load_model, kabsch, body_frame                    # noqa: E402
from mv_framework import forward_pair                                 # noqa: E402
from fitter_3d.joint_limits import (build_head_width_reporter_matrix,  # noqa: E402
                                    _ALLO_B_H_L_REST, _ALLO_B_H_R_REST)

ATTA = f"{REPO}/diagnostics/atta_reference"
NPZ = f"{ATTA}/blender_bundle_rematch_20260907/ATTA20_ARM_A_rematch.npz"
EXPORT = f"{ATTA}/blender_export_rematch_20260907/OmniAnt_25PCs_joint_limited_joint_distances.csv"
EXPORT_SUBMITTED = f"{ATTA}/blender_export/OmniAnt_25PCs_joint_limited_joint_distances.csv"
REF = f"{ATTA}/measurements/ant_body_lengths_and_estimated_head_widths.csv"
SCANS = "/hpcwork/nao48500/atta20"


def load_all():
    M = load_model()
    R = build_head_width_reporter_matrix(M["v_template"]).numpy()
    obs, can, labels = forward_pair(NPZ)
    labels = [l.replace(".obj", "") for l in labels]
    jn = M["jnames"]
    J = np.einsum("ij,njk->nik", M["Jr"], obs)
    bl_model = np.linalg.norm(J[:, jn.index("b_t")] - J[:, jn.index("b_a_5")], axis=-1)
    ref = {}
    with open(REF) as fh:
        for row in csv.DictReader(fh):
            ref[f"{int(row['Shape']):02d}"] = dict(BL=float(row["b_t to b_a_5 [mm]"]),
                                                   HW=float(row["head width [mm]"]),
                                                   mass=float(row["mass [mg]"]))
    BL = np.array([ref[l]["BL"] for l in labels])
    scale = BL / bl_model                                  # mm per model unit, per specimen
    PL = np.einsum("v,nvc->nc", R[0], obs)
    PR = np.einsum("v,nvc->nc", R[1], obs)
    HW_mm = np.linalg.norm(PL - PR, axis=1) * scale
    faces = np.asarray(M["dd"]["f"], dtype=np.int64)
    return dict(M=M, R=R, obs=obs, can=can, labels=labels, J=J, scale=scale, BL=BL,
                ref=ref, PL=PL, PR=PR, HW_mm=HW_mm, faces=faces,
                mass=np.array([ref[l]["mass"] for l in labels]),
                HW_ref=np.array([ref[l]["HW"] for l in labels]))


def blender_hw(path):
    out = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            if {row["Joint1"], row["Joint2"]} == {"b_h_l", "b_h_r"} and row["Shape"] != "Base":
                out[row["Shape"].replace(".obj", "")] = float(row["Scaled Distance in mm"])
    return out


def head_frame(verts, M):
    """Per-specimen head frame: rigid-align head part to the template head, then use the template
    body_frame axes (0 lateral, 1 antero-posterior, 2 dorso-ventral). Returns R, centre, idx."""
    V0 = M["v_template"]
    idx = np.where(M["dominant"] == M["jnames"].index("b_h"))[0]
    T = V0[idx] - V0[idx].mean(0)
    P = verts[idx]; c = P.mean(0)
    Rk = kabsch(P - c, T)
    return Rk, c, idx, body_frame(M)


def to_head_coords(points, Rk, c, bf):
    Q = (np.atleast_2d(points) - c) @ Rk
    return np.stack([Q @ bf[0], Q @ bf[1], Q @ bf[2]], axis=1)   # lat, ap, dv


def load_scan_normalised(label):
    """Raw scan in the fitter's frame: fitter_3d/utils.py::load_meshes centres on the vertex mean
    and divides by max|coord|. Fitted vertices live in that frame, so scans must be normalised
    identically before any overlay (see project memory: frame-mismatch gotcha)."""
    V = []
    for ln in open(f"{SCANS}/{label}.obj"):
        if ln.startswith("v "):
            V.append([float(x) for x in ln.split()[1:4]])
    V = np.asarray(V)
    V = V - V.mean(0)
    return V / np.abs(V).max()


def load_scan_mesh_normalised(label):
    """Raw scan WITH faces, normalised exactly like load_meshes (see load_scan_normalised)."""
    V, Fc = [], []
    for ln in open(f"{SCANS}/{label}.obj"):
        if ln.startswith("v "):
            V.append([float(x) for x in ln.split()[1:4]])
        elif ln.startswith("f "):
            Fc.append([int(t.split("/")[0]) - 1 for t in ln.split()[1:4]])
    V = np.asarray(V); V = V - V.mean(0); V = V / np.abs(V).max()
    return V, np.asarray(Fc, dtype=np.int64)
