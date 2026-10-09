"""X2 Task 4/5 -- PROXY + MECHANISM measurements, and the gating Chamfer cost condition.

PROXY: |log10(head_width) - allo_target_log(body_length)| per specimen, arms A/B/C, computed
       POST-HOC from each arm's Stage_3_deform_fine.npz verts, using the SAME reporter matrix /
       b_t,b_a_5 indices / reference coefficients as the training-time term
       (fitter_3d/joint_limits.py). joints are recovered from verts via J_regressor exactly as
       smal_model/smal_torch.py:397-400 does it (joints = J_regressor.T @ verts on the POSED
       mesh) -- npz does not store joints directly, so this is the same computation, just
       applied outside the training loop.

MECHANISM: head/thorax deform ratio, reusing x1_anterior_mechanism.py's load_arm/vertex_groups/
       anterior_ratio VERBATIM (imported, not reimplemented) so the statistic cannot silently
       drift from X1's.

COST CONDITION (gating, PREREGISTRATION_X2 Sec5): mean Chamfer L2 (metrics.surface_metrics,
       30000 area-sampled points/side) of each arm's FINAL fitted mesh vs. its own target scan.
       arm C must be <= 1.4x arm B.
"""
import glob
import json
import os
import pickle
import sys

import numpy as np
import torch
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "fitter_3d"))
sys.path.insert(0, os.path.join(REPO, "diagnostics", "moonshot"))
sys.path.insert(0, HERE)  # for x1_anterior_mechanism -- explicit, don't rely on script-dir auto-add
sys.path.insert(0, REPO)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import config  # noqa: E402
from joint_limits import (  # noqa: E402
    build_head_width_reporter_matrix, ALLO_EXPONENT_REF, ALLO_INTERCEPT_REF,
)
from x1_anterior_mechanism import (  # noqa: E402
    PART_GROUPS_COARSE, PROBE22_COUNTS, ANTERIOR, vertex_groups, load_arm, three_tests,
)
import metrics as M  # noqa: E402
from fitter_3d.utils import load_meshes  # noqa: E402

ARMS = {
    "A": "diagnostics/moonshot/runs/X1_A_nocap",
    "B": "diagnostics/moonshot/runs/X1_B_scalecap",
    "C": "diagnostics/moonshot/runs/X2_C_allometric",
}
MESH_DIR = "diagnostics/moonshot/bench50_clean"
OUT_DIR = "diagnostics/anterior_mechanism/out_X2"


def allo_proxy_residual(verts_np, dd, R, bt_idx, ba5_idx):
    """abs(log10(head_width) - target_log(body_length)) per specimen, from posed verts (N,V,3)."""
    verts = torch.as_tensor(verts_np, dtype=torch.float64)
    Jreg = dd["J_regressor"]
    Jreg = Jreg.T.todense() if hasattr(Jreg, "todense") else np.asarray(Jreg)
    # smal_torch.py stores J_regressor as (V, J) after the .T; joints = verts^T @ J_regressor
    if Jreg.shape[0] != verts.shape[1]:
        Jreg = Jreg.T
    Jreg_t = torch.as_tensor(np.asarray(Jreg), dtype=torch.float64)  # (V, J)
    joints = torch.einsum("vj,nvc->njc", Jreg_t, verts)
    bl = torch.linalg.norm(joints[:, bt_idx] - joints[:, ba5_idx], dim=-1)

    hw_pts = torch.einsum("jv,nvc->njc", R.double(), verts)
    hw = torch.linalg.norm(hw_pts[:, 0] - hw_pts[:, 1], dim=-1)

    target_log = ALLO_INTERCEPT_REF + ALLO_EXPONENT_REF * torch.log10(bl.clamp_min(1e-8))
    pred_log = torch.log10(hw.clamp_min(1e-8))
    resid = (pred_log - target_log).abs()
    return resid.numpy(), bl.numpy(), hw.numpy()


def chamfer_cost(run_dir, mesh_files, device="cpu"):
    """Mean Chamfer L2 of the arm's final fitted mesh vs its own target, metrics.py's own
    surface_metrics (30000 area-sampled pts/side), matching the live eval suite."""
    p = os.path.join(REPO, run_dir, "Stage_3_deform_fine.npz")
    d = np.load(p)
    verts = torch.as_tensor(d["verts"], dtype=torch.float32, device=device)
    faces_batch = torch.as_tensor(d["faces"], dtype=torch.int64, device=device)  # (N, F, 3)
    assert faces_batch.dim() == 3 and faces_batch.shape[0] == verts.shape[0], (
        f"expected faces saved per-specimen (N,F,3), got {tuple(faces_batch.shape)} "
        f"for {verts.shape[0]} specimens"
    )
    _, target_meshes = load_meshes(mesh_files=mesh_files, device=device)
    vals = []
    for i in range(verts.shape[0]):
        pred_mesh = M._as_meshes(verts[i], faces_batch[i], device)
        sm = M.surface_metrics(pred_mesh, target_meshes[i])
        vals.append(sm["chamfer_l2"])
        if (i + 1) % 10 == 0:
            print(f"    chamfer {i + 1}/{verts.shape[0]}", flush=True)
    return np.array(vals)


def main():
    out_dir = os.path.join(REPO, OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(REPO, config.SMAL_FILE), "rb") as fh:
        u = pickle._Unpickler(fh)
        u.encoding = "latin1"
        dd = u.load()
    jnames = list(dd["J_names"])
    bt_idx = jnames.index("b_t")
    ba5_idx = jnames.index("b_a_5")
    R = build_head_width_reporter_matrix(dd["v_template"], dtype=torch.float64)

    groups = vertex_groups(dd)

    print("=" * 90)
    print("PROXY -- |log10(head_width) - allo_target_log(body_length)| per specimen (lower=better)")
    print("=" * 90)
    proxy_results = {}
    resid_by_arm = {}
    for arm, run_dir in ARMS.items():
        p = os.path.join(REPO, run_dir, "Stage_3_deform_fine.npz")
        d = np.load(p)
        resid, bl, hw = allo_proxy_residual(d["verts"], dd, R, bt_idx, ba5_idx)
        resid_by_arm[arm] = resid
        proxy_results[arm] = {
            "mean_abs_residual": float(resid.mean()),
            "median_abs_residual": float(np.median(resid)),
            "body_length_mean": float(bl.mean()),
            "head_width_mean": float(hw.mean()),
            "n": int(len(resid)),
        }
        print(f"  arm {arm}: mean |resid| = {resid.mean():.5f}  median = {np.median(resid):.5f}  "
              f"(n={len(resid)}, run={run_dir})")

    t_ca = three_tests(resid_by_arm["A"], resid_by_arm["C"])  # A - C > 0 => C smaller (better)
    t_cb = three_tests(resid_by_arm["B"], resid_by_arm["C"])
    proxy_c_lower_than_a = float(resid_by_arm["C"].mean()) < float(resid_by_arm["A"].mean())
    proxy_c_lower_than_b = float(resid_by_arm["C"].mean()) < float(resid_by_arm["B"].mean())
    print(f"\n  C vs A: sign p = {t_ca['sign_p']:.3e}   C lower than A: {proxy_c_lower_than_a}")
    print(f"  C vs B: sign p = {t_cb['sign_p']:.3e}   C lower than B: {proxy_c_lower_than_b}")
    proxy_holds = proxy_c_lower_than_a and proxy_c_lower_than_b
    print(f"\n  PROXY holds (C reduces allometric residual vs BOTH A and B): "
          f"{'PASS' if proxy_holds else 'FAIL -- VOID'}")

    print("\n" + "=" * 90)
    print("MECHANISM -- head/thorax deform ratio, x1_anterior_mechanism.py logic reused verbatim")
    print("=" * 90)
    arms_loaded = {arm: load_arm(run_dir, groups) for arm, run_dir in ARMS.items()}
    for arm in ARMS:
        print(f"  arm {arm}: n={arms_loaded[arm]['n']}")

    mech = {}
    for g in ["head", "mandible", "antenna", "gaster", "waist", "legs"]:
        if g not in arms_loaded["A"]["ratios"][0]:
            continue
        a = np.array([r[g] for r in arms_loaded["A"]["ratios"]])
        b = np.array([r[g] for r in arms_loaded["B"]["ratios"]])
        c = np.array([r[g] for r in arms_loaded["C"]["ratios"]])
        da, db, dc = np.abs(a - 1.0), np.abs(b - 1.0), np.abs(c - 1.0)
        t_ac = three_tests(da, dc)  # da > dc => C closer to 1.0 than A
        closer = float(dc.mean()) < float(da.mean())
        improvement = float(da.mean() - dc.mean())
        mech[g] = {
            "A": float(a.mean()), "B": float(b.mean()), "C": float(c.mean()),
            "absA": float(da.mean()), "absB": float(db.mean()), "absC": float(dc.mean()),
            "closer_C_vs_A": bool(closer), "sign_p_C_vs_A": t_ac["sign_p"],
            "improvement_C_vs_A": improvement,
        }
        print(f"  {g:<10} A={a.mean():.3f} B={b.mean():.3f} C={c.mean():.3f}   "
              f"|A-1|={da.mean():.3f} |C-1|={dc.mean():.3f}  closer={closer}  "
              f"improvement={improvement:+.3f}  sign_p={t_ac['sign_p']:.3e}")

    h = mech["head"]
    mech_verdict = ("PASS" if (h["improvement_C_vs_A"] >= 0.05 and h["sign_p_C_vs_A"] < 0.05
                                and h["closer_C_vs_A"])
                     else "PARTIAL" if (h["closer_C_vs_A"] and h["sign_p_C_vs_A"] < 0.05)
                     else "FAIL")
    if not proxy_holds:
        mech_verdict = "VOID (proxy did not reproduce)"

    print(f"\n  ENDPOINT: head ratio A {h['A']:.3f} -> C {h['C']:.3f}; "
          f"distance to 1.0 improved by {h['improvement_C_vs_A']:+.3f} "
          f"(bar >=0.05, sign p<0.05)")
    print(f"  MECHANISM VERDICT (pre-registered, before cost condition): {mech_verdict}")

    print("\n" + "=" * 90)
    print("COST CONDITION (gating) -- mean Chamfer L2, arm C vs arm B")
    print("=" * 90)
    mesh_files = sorted(glob.glob(os.path.join(REPO, MESH_DIR, "*.obj")))
    cost_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  (chamfer device={cost_device})")
    cost = {}
    for arm in ["A", "B", "C"]:
        vals = chamfer_cost(ARMS[arm], mesh_files, device=cost_device)
        cost[arm] = {"mean_chamfer_l2": float(vals.mean()), "n": int(len(vals))}
        print(f"  arm {arm}: mean chamfer_l2 = {vals.mean():.6f}  (n={len(vals)})")
    cost_ratio_c_over_b = cost["C"]["mean_chamfer_l2"] / cost["B"]["mean_chamfer_l2"]
    cost_ok = cost_ratio_c_over_b <= 1.4
    print(f"\n  C/B chamfer ratio = {cost_ratio_c_over_b:.4f}   "
          f"(bar: <=1.4x)   {'PASS' if cost_ok else 'FAIL'}")

    # ---------------------------------------------------------------- final combined verdict
    if not proxy_holds:
        final_verdict = "VOID"
    elif mech_verdict == "PASS" and cost_ok:
        final_verdict = "PASS"
    elif mech_verdict in ("PASS", "PARTIAL"):
        final_verdict = "PARTIAL"
    else:
        final_verdict = "FAIL"

    print("\n" + "=" * 90)
    print(f"FINAL VERDICT (Sec5, all 3 gating conditions): {final_verdict}")
    print("=" * 90)

    payload = {
        "proxy": proxy_results,
        "proxy_holds": bool(proxy_holds),
        "proxy_C_vs_A": t_ca,
        "proxy_C_vs_B": t_cb,
        "mechanism": mech,
        "mechanism_verdict_precost": mech_verdict,
        "cost": cost,
        "cost_ratio_C_over_B": cost_ratio_c_over_b,
        "cost_condition_ok": bool(cost_ok),
        "final_verdict": final_verdict,
        "arms": ARMS,
    }
    with open(os.path.join(out_dir, "x2_results.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {out_dir}/x2_results.json")


if __name__ == "__main__":
    main()
