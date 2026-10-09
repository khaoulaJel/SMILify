"""R1 — rebuild the landmark->template-vertex map in the CORRECTED frame.

Replaces `annotation/landmark_indices_recalibrated.json`, whose vertices were fitted to landmarks
stored in the Blender Z-up frame (obj = (x, z, -y) of stored) and which consequently span a Weber's
length 1.540x the rule-computed distance on the template.

Method
------
For each expert specimen:
  1. correct the landmark frame, then normalise with the SAME convention the fitter uses:
     c, s = obj.mean(0), abs(obj - obj.mean(0)).max()   [r9_objective_audit.py:69]
  2. ASSERT the corrected landmark lies on the scan surface. Uncorrected landmarks sit 1.4-3% of
     the diagonal off-surface, so this assert genuinely discriminates.
  3. take the fitted model vertices (cached in the JAB production fit; no re-optimisation).

Consensus: the template vertex index minimising the MEAN distance to that landmark across all
specimens -- a global search over all 10235 vertices, not a per-specimen nearest-vertex vote.

Beyond the original: leave-one-specimen-out stability, and per-landmark dispersion, so unstable
landmarks can be excluded rather than silently used.

Ground-truth defects handled per the 2026-09-14 audit: Dolichoderus excluded (annotated on the
Discothyrea mesh); n = 11. The JAB production fit already contains exactly these 11.
"""
import json
import os

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
LM_DIR = os.path.join(REPO, "annotation", "landmarks")
MESH_DIR = "/hpcwork/nao48500/worker_alt_data"
FIT = "/hpcwork/nao48500/jab_runs/A_prod_s0/Stage_3_deform_fine.npz"

fix = lambda v: np.array([v[0], v[2], -v[1]], float)  # noqa: E731  Blender Z-up -> obj

# ----------------------------------------------------------------- load the production fit
z = np.load(FIT, allow_pickle=True)
assert "verts" in z.files, z.files
V = np.asarray(z["verts"], float)                      # (n_spec, 10235, 3), normalised frame
SIDS = [str(x).replace("_processed.obj", "") for x in z["labels"]]
NV = V.shape[1]
print(f"[fit] {FIT}")
print(f"[fit] {len(SIDS)} specimens x {NV} vertices")

# integrity: the fit must not be silently empty / degenerate
assert np.isfinite(V).all(), "non-finite vertices in fit"
assert V.std() > 1e-3, "fit vertices are degenerate"

# ----------------------------------------------------------------- per-specimen landmarks
spec = []
for sid in SIDS:
    p = os.path.join(LM_DIR, f"{sid}_traits.json")
    if not os.path.exists(p):
        print(f"[skip] no landmark file: {sid}")
        continue
    d = json.load(open(p))["landmarks"]

    m = trimesh.load(os.path.join(MESH_DIR, f"{sid}_processed.obj"), process=False)
    o = np.asarray(m.vertices, float)
    c, s = o.mean(0), np.abs(o - o.mean(0)).max()
    diag = float(np.linalg.norm(o.max(0) - o.min(0)))

    corrected, raw_off, cor_off = {}, {}, {}
    for k, v in d.items():
        pc = fix(v["original"])
        pr = np.array(v["original"], float)
        # distance to the scan surface, in % of diagonal, for corrected and raw
        cor_off[k] = float(np.linalg.norm(o - pc, axis=1).min()) / diag * 100.0
        raw_off[k] = float(np.linalg.norm(o - pr, axis=1).min()) / diag * 100.0
        corrected[k] = (pc - c) / s

    med_cor = float(np.median(list(cor_off.values())))
    med_raw = float(np.median(list(raw_off.values())))
    # THE DISCRIMINATING ASSERT: corrected landmarks must sit on the scan.
    assert med_cor < 0.5, f"{sid}: corrected landmarks {med_cor:.3f}% diag off-surface"

    wl = float(np.linalg.norm(corrected["wl_anterior_r"] - corrected["wl_posterior_r"]))
    spec.append(dict(sid=sid, L=corrected, WL=wl, med_cor=med_cor, med_raw=med_raw,
                     vi=SIDS.index(sid)))
    print(f"[lm] {sid:<46} on-surface corrected {med_cor:6.3f}%  raw {med_raw:6.3f}%  WL {wl:.4f}")

print(f"\n[lm] {len(spec)} specimens usable")
print(f"[lm] median on-surface error: corrected {np.median([a['med_cor'] for a in spec]):.4f}% diag"
      f"  |  raw {np.median([a['med_raw'] for a in spec]):.4f}% diag")

# ----------------------------------------------------------------- consensus vertex per landmark
# per-landmark availability: do NOT intersect across specimens, that silently drops landmarks
# missing on a single specimen (Cephalotes carries only 8 of 12).
names = sorted(set().union(*[set(a["L"]) for a in spec]))
out = {}
print(f"\n{'landmark':<26} {'n':>3} {'vertex':>7} {'mean %WL':>9} {'max %WL':>8} {'LOO stable':>11} {'LOO move %WL':>13}")
print("-" * 88)
for k in names:
    sub = [a for a in spec if k in a["L"]]
    if len(sub) < 3:
        print(f"{k:<26} {len(sub):>3}  SKIPPED (needs >=3 specimens)")
        continue
    # distance from every template vertex to this landmark, per specimen, in % WL
    D = np.stack([np.linalg.norm(V[a["vi"]] - a["L"][k], axis=1) / a["WL"] * 100.0
                  for a in sub])                                      # (n_sub, NV)
    mean_d = D.mean(0)
    best = int(np.argmin(mean_d))

    # leave-one-specimen-out stability of the chosen index
    loo_idx, loo_move = [], []
    for i in range(len(sub)):
        m_ = np.delete(D, i, axis=0).mean(0)
        b = int(np.argmin(m_))
        loo_idx.append(b)
        loo_move.append(float(np.linalg.norm(
            V[sub[0]["vi"]][b] - V[sub[0]["vi"]][best]) / sub[0]["WL"] * 100.0))
    stable = sum(int(b == best) for b in loo_idx)

    out[k] = dict(vertex=best, n=len(sub),
                  mean_pct_wl=float(mean_d[best]), max_pct_wl=float(D[:, best].max()),
                  per_specimen_pct_wl={a["sid"]: float(D[i, best]) for i, a in enumerate(sub)},
                  loo_stable=stable, loo_max_move_pct_wl=float(max(loo_move)))
    print(f"{k:<26} {len(sub):>3} {best:>7} {mean_d[best]:>9.2f} {D[:, best].max():>8.2f} "
          f"{stable:>6}/{len(sub)} {max(loo_move):>13.2f}")

# ----------------------------------------------------------------- comparison with the old file
old_p = os.path.join(REPO, "annotation", "landmark_indices_recalibrated.json")
old = json.load(open(old_p)) if os.path.exists(old_p) else {}
print(f"\n{'landmark':<26} {'old v':>7} {'new v':>7} {'displacement %WL':>18}")
print("-" * 62)
for k in names:
    if k not in old:
        continue
    ov, nv = int(old[k]["vertex"]), out[k]["vertex"]
    disp = float(np.median([np.linalg.norm(V[a["vi"]][ov] - V[a["vi"]][nv]) / a["WL"] * 100.0
                            for a in spec]))
    out[k]["old_vertex"] = ov
    out[k]["displacement_from_old_pct_wl"] = disp
    print(f"{k:<26} {ov:>7} {nv:>7} {disp:>18.1f}")

meta = dict(
    built="2026-09-16", fit=FIT, n_specimens=len(spec),
    specimens=[a["sid"] for a in spec],
    frame_note="landmarks corrected with obj=(x,z,-y) then normalised by (o-mean)/maxabs",
    supersedes="annotation/landmark_indices_recalibrated.json (fitted in the Blender Z-up frame)",
    landmarks=out)
dst = os.path.join(HERE, "landmark_indices_recalibrated_v2.json")
json.dump(meta, open(dst, "w"), indent=1)
print(f"\n[out] {dst}")
