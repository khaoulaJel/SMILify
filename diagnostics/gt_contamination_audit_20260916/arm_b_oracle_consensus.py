"""Arm B -- repeat R1's consensus-vertex search on the ORACLE-SUPERVISED fit.

R1 measured landmark agreement through the PRODUCTION fit (A_prod_s0), whose own median
joint error on these 11 specimens is 26.9 % WL (JAB scores_specimen.csv, seed 0, stage
Stage_3_deform_fine). The oracle arm O1_all_s0, supervised on ALL expert joints, reaches
17.4 % WL on the same specimens -- a 35.2 % reduction in fit error.

PREDICTION, stated before looking:
  if R1's 16.6-57.5 % WL residual is dominated by FIT ERROR, the consensus residual should
  fall by roughly the same 35 % under O1.
  if it is dominated by ABSENCE OF CORRESPONDENCE, it should barely move.
A paired per-landmark, per-specimen comparison is reported with a bootstrap CI on the ratio.

Everything else (frame correction, normalisation, global 10,235-vertex search, LOO
stability) is identical to recalibrate_landmarks.py, deliberately.
"""
import json
import os

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
LM_DIR = os.path.join(REPO, "annotation", "landmarks")
MESH_DIR = "/hpcwork/nao48500/worker_alt_data"
FITS = {"A_prod": "/hpcwork/nao48500/jab_runs/A_prod_s0/Stage_3_deform_fine.npz",
        "O1_all": "/hpcwork/nao48500/jab_runs/O1_all_s0/Stage_3_deform_fine.npz"}

fix = lambda v: np.array([v[0], v[2], -v[1]], float)  # noqa: E731
rng = np.random.default_rng(0)

# ----------------------------------------------------------------- landmarks (frame-corrected)
def load_landmarks(sids):
    out = {}
    for sid in sids:
        m = trimesh.load(os.path.join(MESH_DIR, f"{sid}_processed.obj"), process=False)
        o = np.asarray(m.vertices, float)
        c, s = o.mean(0), np.abs(o - o.mean(0)).max()
        diag = float(np.linalg.norm(o.max(0) - o.min(0)))
        d = json.load(open(os.path.join(LM_DIR, f"{sid}_traits.json")))["landmarks"]
        L, off = {}, []
        for k, v in d.items():
            pc = fix(v["original"])
            off.append(float(np.linalg.norm(o - pc, axis=1).min()) / diag * 100.0)
            L[k] = (pc - c) / s
        med = float(np.median(off))
        assert med < 0.5, f"{sid}: corrected landmarks {med:.3f}% diag off-surface"
        out[sid] = dict(L=L, WL=float(np.linalg.norm(L["wl_anterior_r"] - L["wl_posterior_r"])),
                        on_surface=med)
    return out


res, percell = {}, {}
for arm, path in FITS.items():
    z = np.load(path, allow_pickle=True)
    V = np.asarray(z["verts"], float)
    sids = [str(x).replace("_processed.obj", "") for x in z["labels"]]
    assert np.isfinite(V).all() and V.std() > 1e-3
    lm = load_landmarks(sids)
    names = sorted(set().union(*[set(lm[s]["L"]) for s in sids]))
    out = {}
    print(f"\n=== {arm}   {path}")
    print(f"{'landmark':<26} {'n':>3} {'vertex':>7} {'mean %WL':>9} {'max %WL':>8} {'LOO':>8}")
    for k in names:
        sub = [s for s in sids if k in lm[s]["L"]]
        if len(sub) < 3:
            continue
        D = np.stack([np.linalg.norm(V[sids.index(s)] - lm[s]["L"][k], axis=1) / lm[s]["WL"] * 100
                      for s in sub])
        best = int(np.argmin(D.mean(0)))
        loo = sum(int(np.argmin(np.delete(D, i, 0).mean(0)) == best) for i in range(len(sub)))
        out[k] = dict(vertex=best, n=len(sub), mean_pct_wl=float(D.mean(0)[best]),
                      max_pct_wl=float(D[:, best].max()), loo_stable=loo,
                      per_specimen={s: float(D[i, best]) for i, s in enumerate(sub)})
        percell[(arm, k)] = {s: float(D[i, best]) for i, s in enumerate(sub)}
        print(f"{k:<26} {len(sub):>3} {best:>7} {D.mean(0)[best]:>9.2f} "
              f"{D[:, best].max():>8.2f} {loo:>5}/{len(sub)}")
    res[arm] = out

# ----------------------------------------------------------------- paired comparison
print(f"\n{'landmark':<26} {'A_prod':>8} {'O1_all':>8} {'ratio':>7}  vertex A->O")
keys = sorted(res["A_prod"])
pa, po = [], []
for k in keys:
    a, o = res["A_prod"][k]["mean_pct_wl"], res["O1_all"][k]["mean_pct_wl"]
    pa.append(a)
    po.append(o)
    print(f"{k:<26} {a:>8.2f} {o:>8.2f} {o/a:>7.3f}  "
          f"{res['A_prod'][k]['vertex']:>5} -> {res['O1_all'][k]['vertex']:<5}"
          f"{'  SAME' if res['A_prod'][k]['vertex']==res['O1_all'][k]['vertex'] else ''}")
pa, po = np.array(pa), np.array(po)

# paired per (landmark, specimen) cell, bootstrapped over LANDMARKS (the unit R1 reports)
cells = [(k, s) for k in keys for s in percell[("A_prod", k)]]
ca = np.array([percell[("A_prod", k)][s] for k, s in cells])
co = np.array([percell[("O1_all", k)][s] for k, s in cells])
bs = [np.median(co[i]) / np.median(ca[i])
      for i in (rng.integers(0, len(ca), (4000, len(ca))))]
lo, hi = np.percentile(bs, [2.5, 97.5])
print(f"\nmean over landmarks:  A_prod {pa.mean():.2f} %WL   O1_all {po.mean():.2f} %WL   "
      f"ratio {po.mean()/pa.mean():.3f}")
print(f"median over (landmark, specimen) cells: A_prod {np.median(ca):.2f}  O1_all "
      f"{np.median(co):.2f}  ratio {np.median(co)/np.median(ca):.3f}  "
      f"95% CI [{lo:.3f}, {hi:.3f}]  (n={len(ca)} cells, bootstrap over cells)")
print(f"fit-error ratio to beat (JAB med_fk): 17.43 / 26.95 = {17.431/26.945:.3f}")
print(f"landmarks whose consensus vertex is UNCHANGED under oracle supervision: "
      f"{sum(res['A_prod'][k]['vertex']==res['O1_all'][k]['vertex'] for k in keys)}/{len(keys)}")

json.dump(dict(built="2026-09-16", fits=FITS,
               jab_med_fk_pct_wl=dict(A_prod=26.945, O1_all=17.431),
               arms=res,
               cell_ratio=dict(median=float(np.median(co) / np.median(ca)),
                               ci95=[float(lo), float(hi)], n_cells=len(ca))),
          open(os.path.join(HERE, "arm_b_oracle_consensus.json"), "w"), indent=1)
print(f"\n[out] arm_b_oracle_consensus.json")
