"""X2 Step 0 — differentiable torch replica of the Blender addon's b_h_l/b_h_r head-width
reporter bones, validated against the Blender CSV ground truth.

Algorithm replicated EXACTLY from the SMIL Model Importer addon (smil_importer.zip, extracted
from diagnostics/atta_reference/blender_bundle/smil_importer.zip), specifically:

- `core_mesh.py::find_nearest_neighbors` (the "10 nearest, inverse distance" formula):
      distances = np.linalg.norm(vertices - joint_loc, axis=1)
      nearest_indices[i] = np.argpartition(distances, n)[:n]     # n = 10
      weights = 1.0 / nearest_distances                          # 1/d, NOT 1/d^2
      weights /= weights.sum()                                   # normalized to sum to 1

- `core_mesh.py::export_J_regressor_to_npy` / `mesh_to_numpy`: the vertex array searched is
  `mesh_obj.data.vertices` (`vert.co` for each vertex) -- confirmed in
  diagnostics/atta_reference/REPORT_ATTA_HEADWIDTH.md ("the exporter's nearest-vertex search
  reads mesh.vertices, which always holds Basis coordinates regardless of which shape key is
  displayed") to be the model's Basis/rest-pose vertex array, i.e. the pkl's `v_template`.

- The reference point each reporter bone's "10 nearest" is measured from is NOT b_h (the parent
  joint) -- it is the reporter bone's OWN `head_local` position in the rest armature:
  `core_mesh.py::export_J_regressor_to_npy` builds `joint_locations` from
  `[bone.head_local for bone in armature_obj.data.bones]`, one row per bone in the CURRENT
  armature (which includes b_h_l/b_h_r themselves, each already placed by the artist in Blender
  at "the widest point of the head capsule", per REPORT_ATTA_HEADWIDTH.md ~line 76-82). This
  refutes the assumption in the preregistration text that nearest vertices are relative to
  b_h's position -- confirmed from the addon source, not assumed.

- `measurements.py::build_J_regressor_with_reporters`: rows for bones already in the model's
  trained J_regressor (55 of them, matched by NAME) are copied VERBATIM; only b_h_l/b_h_r get a
  freshly computed inverse-distance row. Joint/reporter positions every frame are then
  `J_regressor @ vertex_positions` (`measurements.py::recalculate_joint_positions`), the same
  pattern as `joints = J_regressor @ verts` in smal_model/smal_torch.py:394-399.

Since this task has no Blender access, b_h_l/b_h_r's rest-pose `head_local` positions (needed to
determine their "10 nearest" Basis vertices) are recovered by trilateration from the Blender CSV's
own `Base` row: `Base` gives exact pairwise distances between every bone (b_h_l, b_h_r, and all 55
trained joints) at Basis pose, and the 55 trained joints' Basis positions are exactly
`J_regressor_trained @ v_template` (confirmed byte-identical to the CSV's own Base distances for
trained-joint pairs, e.g. Base b_t-b_a_5 = 0.9048573021251339 both ways). With 55 anchors this
linear least-squares trilateration is enormously overdetermined; the residual is ~1e-9 model
units, i.e. essentially exact, not an approximation.
"""

import csv
import pickle
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PKL_PATH = REPO / "diagnostics/atta_reference/blender_bundle/OmniAnt_25PCs_joint_limited.pkl"
CSV_PATH = REPO / "diagnostics/atta_reference/blender_export/OmniAnt_25PCs_joint_limited_joint_distances.csv"
NPZ_PATH = REPO / "diagnostics/atta_reference/blender_bundle/ATTA20_ARM_A.npz"

N_NEAREST = 10
REPORTER_NAMES = ["b_h_l", "b_h_r"]


def _load_pkl():
    with open(PKL_PATH, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    return d


def _load_csv_rows():
    with open(CSV_PATH, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        rows = list(r)
    assert header[:4] == ["Shape", "Joint1", "Joint2", "Distance"], header
    return rows


def _base_distance(rows, name_a, name_b):
    """Exact Basis-pose distance between two bones from the CSV's 'Base' rows."""
    pair = {name_a, name_b}
    for row in rows:
        if row[0] == "Base" and {row[1], row[2]} == pair:
            return float(row[3])
    raise KeyError(f"No Base row for {name_a}-{name_b}")


def _trilaterate(target_name, rows, anchor_names, anchor_pos):
    """Recover a bone's rest-pose 3D position from its Basis-pose distances to a set of
    already-known anchor positions (linear least squares, exact if noiseless)."""
    dists = np.array([_base_distance(rows, target_name, n) for n in anchor_names])
    a0, r0 = anchor_pos[0], dists[0]
    A = 2 * (anchor_pos[1:] - a0)
    b = (r0**2 - dists[1:] ** 2) + np.sum(anchor_pos[1:] ** 2, axis=1) - np.sum(a0**2)
    p, *_ = np.linalg.lstsq(A, b, rcond=None)
    resid = np.linalg.norm(anchor_pos - p, axis=1) - dists
    return p, resid


def _nearest_weights(point, vertices, n=N_NEAREST):
    """Exact replica of core_mesh.py::find_nearest_neighbors for a single joint location."""
    dist = np.linalg.norm(vertices - point, axis=1)
    idx = np.argpartition(dist, n)[:n]
    d = dist[idx]
    w = 1.0 / d
    w = w / w.sum()
    return idx, w


def build_reporter_matrix(dtype=np.float32):
    """Build the frozen (2, V) reporter regressor R for b_h_l (row 0) / b_h_r (row 1),
    matching the addon's dtype (float32) for the nearest-neighbor search itself.

    Returns:
        R: (2, V) numpy array, rows sum to 1.
        info: dict with trilateration residuals and the recovered rest-pose positions,
              for the mechanism checks in Step D.
    """
    d = _load_pkl()
    J_names = list(d["J_names"])
    v_template = d["v_template"].astype(np.float64)  # trilateration anchors: keep full precision
    J_regressor = d["J_regressor"].astype(np.float64)
    anchor_pos = J_regressor @ v_template  # (55, 3), exact Basis positions of trained joints

    rows = _load_csv_rows()

    v_search = v_template.astype(dtype)  # what find_nearest_neighbors actually searches, addon dtype

    R = np.zeros((2, v_template.shape[0]), dtype=np.float64)
    info = {"positions": {}, "trilat_resid_rms": {}}
    for row_i, name in enumerate(REPORTER_NAMES):
        pos, resid = _trilaterate(name, rows, J_names, anchor_pos)
        idx, w = _nearest_weights(pos.astype(dtype), v_search, N_NEAREST)
        R[row_i, idx] = w
        info["positions"][name] = pos
        info["trilat_resid_rms"][name] = float(np.sqrt(np.mean(resid**2)))
    return R, info


# ---------------------------------------------------------------------------
# The differentiable reporter itself.
# ---------------------------------------------------------------------------

class HeadWidthReporter:
    """Frozen (2, V) sparse-as-dense regressor + differentiable head_width(verts)."""

    def __init__(self, R: np.ndarray, device=None, dtype=torch.float64):
        self.R = torch.as_tensor(R, dtype=dtype, device=device)  # (2, V)

    def head_bone_pos(self, verts: torch.Tensor) -> torch.Tensor:
        """verts: (V, 3) or (B, V, 3) -> (2, 3) or (B, 2, 3)."""
        return torch.einsum("jv,...vc->...jc", self.R, verts)

    def head_width(self, verts: torch.Tensor) -> torch.Tensor:
        pos = self.head_bone_pos(verts)  # (..., 2, 3)
        return torch.linalg.norm(pos[..., 0, :] - pos[..., 1, :], dim=-1)


def _gradient_sanity_check(reporter: HeadWidthReporter, n_verts: int):
    torch.manual_seed(0)
    verts = torch.randn(n_verts, 3, dtype=torch.float64, requires_grad=True)
    hw = reporter.head_width(verts)
    hw.backward()
    assert verts.grad is not None, "no gradient reached verts"
    assert torch.any(verts.grad != 0), "gradient is all zero"
    n_nonzero = int((verts.grad.abs().sum(dim=-1) > 0).sum())
    print(f"[gradcheck-lite] head_width={hw.item():.6f}, grad populated on {n_nonzero} vertices "
          f"(expect <= {2 * N_NEAREST}), grad norm={verts.grad.norm().item():.6f}")
    assert n_nonzero <= 2 * N_NEAREST

    # torch.autograd.gradcheck on a tiny random subproblem restricted to the involved vertices,
    # for a numerically-checked derivative, not just "grad is populated".
    involved = torch.nonzero(reporter.R.sum(dim=0) != 0).squeeze(-1)
    small_verts = torch.randn(len(involved), 3, dtype=torch.float64, requires_grad=True)

    def f(x):
        full = torch.zeros(reporter.R.shape[1], 3, dtype=torch.float64)
        full[involved] = x
        return reporter.head_width(full).unsqueeze(0)

    ok = torch.autograd.gradcheck(f, (small_verts,), eps=1e-6, atol=1e-4)
    print(f"[gradcheck] torch.autograd.gradcheck passed: {ok}")


# ---------------------------------------------------------------------------
# Step C validation against the 20 Atta specimens.
# ---------------------------------------------------------------------------

def validate_against_blender_csv(R: np.ndarray):
    rows = _load_csv_rows()
    npz = np.load(NPZ_PATH, allow_pickle=True)
    verts_all = npz["verts"]  # (20, V, 3), fitter's final posed+deformed output per specimen
    labels = [str(l) for l in npz["labels"]]

    reporter = HeadWidthReporter(R, dtype=torch.float64)

    results = []
    for i, label in enumerate(labels):
        v = torch.as_tensor(verts_all[i], dtype=torch.float64)
        pred = reporter.head_width(v).item()
        gt = _base_distance(rows, "b_h_l", "b_h_r") if label == "Base" else None
        if gt is None:
            for row in rows:
                if row[0] == label and {row[1], row[2]} == {"b_h_l", "b_h_r"}:
                    gt = float(row[3])
                    break
        assert gt is not None, f"no CSV row for {label}"
        abs_err = pred - gt
        rel_err_pct = abs_err / gt * 100.0
        results.append({
            "specimen": label,
            "pred": pred,
            "gt": gt,
            "abs_err": abs_err,
            "rel_err_pct": rel_err_pct,
        })
    return results


def _also_check_base_pose(R: np.ndarray):
    """Sanity check independent of any specimen fit: does R reproduce the CSV's own Base
    (rest-pose) b_h_l-b_h_r distance? This isolates algorithm correctness from any question
    about whether the posed vertex source (npz) matches what Blender actually measured."""
    d = _load_pkl()
    v_template = torch.as_tensor(d["v_template"], dtype=torch.float64)
    reporter = HeadWidthReporter(R, dtype=torch.float64)
    pred = reporter.head_width(v_template).item()
    gt = _base_distance(_load_csv_rows(), "b_h_l", "b_h_r")
    return pred, gt


if __name__ == "__main__":
    print("=" * 78)
    print("Building reporter matrix R (2, V) from trilaterated bone positions ...")
    R, info = build_reporter_matrix()
    for name in REPORTER_NAMES:
        print(f"  {name}: trilateration residual RMS = {info['trilat_resid_rms'][name]:.3e} "
              f"model units (recovered position {info['positions'][name]})")

    print()
    print("Gradient sanity check (toy random verts) ...")
    reporter = HeadWidthReporter(R, dtype=torch.float64)
    _gradient_sanity_check(reporter, n_verts=R.shape[1])

    print()
    print("Base/rest-pose check (algorithm-only, no specimen-fit vertex source involved) ...")
    pred_base, gt_base = _also_check_base_pose(R)
    print(f"  predicted={pred_base:.8f}  csv_base={gt_base:.8f}  "
          f"rel_err={abs(pred_base - gt_base) / gt_base * 100:.6f}%")

    print()
    print("Validating against the 20 Atta specimens (posed+deformed fitted verts) ...")
    results = validate_against_blender_csv(R)
    print(f"{'specimen':10s} {'pred':>10s} {'gt':>10s} {'abs_err':>10s} {'rel_err%':>10s}")
    for r in results:
        print(f"{r['specimen']:10s} {r['pred']:10.5f} {r['gt']:10.5f} "
              f"{r['abs_err']:10.5f} {r['rel_err_pct']:10.3f}")

    rel = np.array([abs(r["rel_err_pct"]) for r in results])
    print()
    print(f"mean |rel_err|  = {rel.mean():.3f}%")
    print(f"max  |rel_err|  = {rel.max():.3f}%")
    print(f"n within 1%     = {int((rel <= 1.0).sum())} / {len(rel)}")
    tol = 1.0
    verdict = "PASS" if rel.max() <= tol else "FAIL"
    print(f"VERDICT (<= {tol}% per specimen): {verdict}")
