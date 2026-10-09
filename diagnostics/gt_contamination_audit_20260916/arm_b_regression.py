"""Arm B2 -- the correspondence FLOOR, by regression over fit quality.

Arm B contrasted two fits. The JAB run tree holds ~100 fits of the SAME 11 specimens whose
joint error (JAB med_fk, % WL) spans ~17 % to ~45 %. For each fit we recompute R1's
consensus-vertex residual exactly as recalibrate_landmarks.py does, then regress

    consensus residual (% WL)  ~  a + b * fit joint error (% WL)

The INTERCEPT a is the estimand this whole experiment is about: the landmark residual that
would remain if the fit were perfect. a ~ 0 => R1's ~24 % is fit error. a ~ 20 => a genuine
absence of vertex-level correspondence. The CI on a says whether n separates them at all.

Caveats, stated up front:
  * the fits are not independent draws (shared specimens, arms, seeds); the CI is a
    bootstrap over FITS and understates dependence between arms.
  * this is an extrapolation below the observed range (min med_fk 17.4 %), which the fit
    family never reaches; the intercept is model-based, not measured.
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
LM_DIR = os.path.join(REPO, "annotation", "landmarks")
MESH_DIR = "/hpcwork/nao48500/worker_alt_data"
RUNS = "/hpcwork/nao48500/jab_runs"
fix = lambda v: np.array([v[0], v[2], -v[1]], float)  # noqa: E731
rng = np.random.default_rng(0)

sc = pd.read_csv(os.path.join(REPO, "diagnostics/joint_alignment_benchmark/data/scores_specimen.csv"))
fk = sc.groupby(["arm", "seed", "stage"])["med_fk"].median()

# ----------------------------------------------------------------- landmarks (once)
SIDS = sorted(json.load(open(os.path.join(
    REPO, "diagnostics/joint_alignment_benchmark/data/gt_joints_fitframe.json"))).keys())
LM = {}
for sid in SIDS:
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
    assert np.median(off) < 0.5, sid
    LM[sid] = dict(L=L, WL=float(np.linalg.norm(L["wl_anterior_r"] - L["wl_posterior_r"])))
NAMES = sorted(set().union(*[set(v["L"]) for v in LM.values()]))
print(f"[lm] {len(LM)} specimens, {len(NAMES)} landmarks")


def consensus(V, sids):
    """R1's estimator: per landmark, the template vertex minimising mean %WL distance."""
    cells, per_lm, verts = {}, {}, {}
    for k in NAMES:
        sub = [s for s in sids if k in LM[s]["L"]]
        if len(sub) < 3:
            continue
        D = np.stack([np.linalg.norm(V[sids.index(s)] - LM[s]["L"][k], axis=1) / LM[s]["WL"] * 100
                      for s in sub])
        b = int(np.argmin(D.mean(0)))
        verts[k] = b
        per_lm[k] = float(D.mean(0)[b])
        for i, s in enumerate(sub):
            cells[(k, s)] = float(D[i, b])
    return cells, per_lm, verts


rows = []
for p in sorted(glob.glob(f"{RUNS}/*/*.npz")):
    d, f = os.path.basename(os.path.dirname(p)), os.path.basename(p)[:-4]
    mm = re.match(r"(.+)_s(\d+)(_hier)?$", d)
    if not mm:
        continue
    arm, seed = mm.group(1), int(mm.group(2))
    key = (arm, seed, f)
    if key not in fk.index:
        continue
    try:
        z = np.load(p, allow_pickle=True)
        if "verts" not in z.files:
            continue
        V = np.asarray(z["verts"], float)
        sids = [str(x).replace("_processed.obj", "") for x in z["labels"]]
    except Exception as e:                                     # noqa: BLE001
        print(f"[skip] {p}: {e}")
        continue
    if sorted(sids) != SIDS or not np.isfinite(V).all() or V.std() < 1e-3:
        print(f"[skip] {p}: specimen set / integrity")
        continue
    cells, per_lm, verts = consensus(V, sids)
    rows.append(dict(arm=arm, seed=seed, stage=f, med_fk=float(fk.loc[key]),
                     cell_median=float(np.median(list(cells.values()))),
                     lm_mean=float(np.mean(list(per_lm.values()))),
                     mand_same=int(verts.get("mandibular_apex_l") == verts.get("mandibular_apex_r")),
                     per_lm=per_lm, verts=verts))
    print(f"[fit] {arm:<22} s{seed} {f:<22} med_fk {rows[-1]['med_fk']:6.2f}  "
          f"cell-median {rows[-1]['cell_median']:6.2f}  lm-mean {rows[-1]['lm_mean']:6.2f}  "
          f"L/R-mandible-collapse {rows[-1]['mand_same']}")

df = pd.DataFrame(rows)
print(f"\n[reg] {len(df)} fits, med_fk range {df.med_fk.min():.1f}-{df.med_fk.max():.1f} %WL")


def ols_ci(x, y, label):
    b, a = np.polyfit(x, y, 1)
    boot = np.array([np.polyfit(x[i], y[i], 1)
                     for i in rng.integers(0, len(x), (4000, len(x)))])
    lo, hi = np.percentile(boot[:, 1], [2.5, 97.5])
    r = np.corrcoef(x, y)[0, 1]
    print(f"[reg] {label:<12} slope {b:6.3f}  intercept {a:7.2f} %WL  "
          f"95% CI [{lo:6.2f}, {hi:6.2f}]  r {r:.3f}")
    return dict(slope=float(b), intercept=float(a), ci95=[float(lo), float(hi)], r=float(r))


out = dict(built="2026-09-16", n_fits=len(df),
           fits=[{k: v for k, v in r.items() if k != "verts"} for r in rows],
           reg_cell=ols_ci(df.med_fk.values, df.cell_median.values, "cell-median"),
           reg_lm=ols_ci(df.med_fk.values, df.lm_mean.values, "lm-mean"),
           mandible_collapse_rate=float(df.mand_same.mean()),
           collapse_by_fk=df.groupby(pd.cut(df.med_fk, [0, 20, 25, 30, 35, 100]),
                                     observed=True)["mand_same"].mean().to_dict().__str__())
print(f"[reg] L/R mandible collapse in {df.mand_same.sum()}/{len(df)} fits")
print(df.groupby(pd.cut(df.med_fk, [0, 20, 25, 30, 35, 100]), observed=True)
      .agg(n=("arm", "size"), cell_median=("cell_median", "mean"),
           collapse=("mand_same", "mean")))
json.dump(out, open(os.path.join(HERE, "arm_b_regression.json"), "w"), indent=1)
df.drop(columns=["per_lm", "verts"]).to_csv(os.path.join(HERE, "arm_b_regression.csv"), index=False)
print("\n[out] arm_b_regression.json / .csv")
