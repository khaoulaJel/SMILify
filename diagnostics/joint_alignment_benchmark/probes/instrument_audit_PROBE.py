"""INSTRUMENT AUDIT (runs BEFORE any strategy arm): how much did the old ruler differ from the new
one, and does the existing production fit independently corroborate the L/R-swap decision?

Uses only the pre-existing production fits (Z8_W*, D1 recipe) -- no new fitting.

  new-abs     GT via .blend registration (fit-independent), no post-hoc alignment
  new-noswap  same, but mirrored specimens WITHOUT the _r/_l swap        (corroboration test)
  new-PA      per-specimen similarity Procrustes on the evaluated joints (PA-MPJPE analogue)
  old-V9      raw GT similarity-aligned to the production joints, no swap, as V9 did
Errors: % of Weber's length (human landmarks, frame-corrected), median over joints per specimen.
"""
import json
import os
import sys
import glob

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(BENCH, "..", ".."))
sys.path.insert(0, os.path.join(BENCH, "tools"))
import model_joints as MJ  # noqa: E402
from gt_registration import swap_side, norm_sid  # noqa: E402

EXCL = {"w_1_r", "w_2_r", "w_1_l", "w_2_l", "b_h"}


def sim_align(src, dst):
    cs, cd = src.mean(0), dst.mean(0)
    U, S, Vt = np.linalg.svd((dst - cd).T @ (src - cs))
    E = np.eye(3); E[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ E @ Vt
    s = np.trace(np.diag(S) @ E) / ((src - cs) ** 2).sum()
    return (src - cs) @ R.T * s + cd


def main():
    G = json.load(open(os.path.join(BENCH, "data/gt_joints_fitframe.json")))
    names = MJ.joint_names()
    fits = {}
    for p in sorted(glob.glob(os.path.join(REPO, "diagnostics/moonshot/runs/Z8_W*/Stage_3_deform_fine.npz"))):
        fits.update(MJ.load_fit(p, wanted=set(G)))
    rows = []
    for sid, g in sorted(G.items()):
        if sid not in fits:
            print("no production fit for", sid); continue
        use = [n for n in names if n in g["joints"] and n not in EXCL]
        gt = np.array([g["joints"][n]["fit"] for n in use])
        wl = g["WL_fit"]
        r = dict(sid=sid, tier=g["tier"], n=len(use))
        for jdef in ("FK", "REG", "SKIN"):
            J = fits[sid][jdef]
            pr = np.array([J[names.index(n)] for n in use])
            r[f"{jdef}_new_abs"] = float(np.median(np.linalg.norm(pr - gt, axis=1)) / wl * 100)
            # undo the swap: the GT name n was produced from blend name swap_side(n)
            if g["labels_swapped"]:
                pr_ns = np.array([J[names.index(swap_side(n))] for n in use])
                r[f"{jdef}_new_noswap"] = float(np.median(np.linalg.norm(pr_ns - gt, axis=1)) / wl * 100)
            r[f"{jdef}_new_PA"] = float(np.median(np.linalg.norm(sim_align(gt, pr) - pr, axis=1)) / wl * 100)
        # old V9 ruler: raw annotation (blend frame, original labels) aligned onto production REG joints
        gj = [q for q in glob.glob(os.path.join(REPO, "annotation/gt_expert/*_joints.json"))
              if norm_sid(json.load(open(q))["specimen_id"]) == sid][0]
        raw = {j["joint_name"]: np.array(j["position"], float) for j in json.load(open(gj))["joints"]
               if j.get("position") is not None}
        use_raw = [n for n in names if n in raw and n not in EXCL]
        A = np.array([raw[n] for n in use_raw]); Bj = np.array([fits[sid]["REG"][names.index(n)] for n in use_raw])
        r["REG_old_V9"] = float(np.median(np.linalg.norm(sim_align(A, Bj) - Bj, axis=1)) / wl * 100)
        r["FK_vs_REG"] = float(np.median(np.linalg.norm(fits[sid]["FK"] - fits[sid]["REG"], axis=1)) / wl * 100)
        r["repro_err"] = fits[sid]["repro_err"]
        rows.append(r)

    keys = ["FK_new_abs", "FK_new_noswap", "FK_new_PA", "REG_new_abs", "REG_new_noswap", "SKIN_new_abs", "REG_old_V9", "FK_vs_REG"]
    print(f"{'specimen':34s}{'tier':>9}" + "".join(f"{k:>15}" for k in keys))
    for r in rows:
        print(f"{r['sid'][:34]:34s}{r['tier']:>9}" + "".join(
            f"{r[k]:>15.1f}" if k in r else f"{'-':>15}" for k in keys))
    for k in keys:
        v = [r[k] for r in rows if k in r]
        print(f"  median {k:16s} {np.median(v):6.1f}%  (n={len(v)})")
    mir = [r for r in rows if "FK_new_noswap" in r]
    print(f"\nSWAP CORROBORATION (mirrored specimens): swap better on "
          f"{sum(r['FK_new_abs'] < r['FK_new_noswap'] for r in mir)}/{len(mir)} (FK), "
          f"{sum(r['REG_new_abs'] < r['REG_new_noswap'] for r in mir)}/{len(mir)} (REG)")
    json.dump(rows, open(os.path.join(HERE, "instrument_audit_PROBE_out.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
