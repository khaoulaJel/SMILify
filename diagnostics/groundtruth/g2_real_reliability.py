"""G2 -- recompute the report's per-feature reliability table against REAL ground truth.

WHAT §2.2 OF REPORT_MORPHOMETRICS.md CURRENTLY RESTS ON
The table that decides which measurements the report trusts -- and which drives `CORE_BLOCKS`,
and therefore what gets published as a finding -- was computed on **12 SYNTHETIC specimens
generated from the model itself**. Those are in-distribution by construction: the same trap that
made ALL_ANTS_CLEAN an invalid control in Z1. It reports head R = 0.958 and leg-distal R = 0.136.

TWO REASONS THAT NUMBER CANNOT BE COMPARED TO REALITY AS IT STANDS
1. `calib_features.py` correlates RAW measurements (verified: no `log_shape_ratios` step). On
   synthetic specimens the fit and the truth share one absolute scale, so a large ant has a long
   everything in both -- correlation is then dominated by SIZE and runs toward 1 for every feature
   whether or not the shape was recovered. Z8's real repeatability of 0.149 and §2.2's 0.958 are
   very likely not the same quantity.
2. On real specimens that method is not even computable. Each annotation lives in its own scan's
   arbitrary units (§2.1: "only proportions are available"), so there is no shared scale to
   correlate. The real calibration has to be proportion-based.

WHAT THIS DOES
Bone lengths are `|J[child] - J[parent]|`, so they are computable directly from annotated joints
with no meshing step, and `measure.py`'s own `bone_table` supplies the parent/child pairs -- the
same code path for truth and fit, so a discrepancy is a property of the fit. Both sides go through
Mosimann log-shape-ratios, which removes isometric size and leaves proportion, which is the only
thing the corpus supports.

Reported per feature and per block:
  R      correlation between fitted and true PROPORTION across the 10 real specimens
  bias   median signed error -- systematic over/under-estimation, which R cannot see
  SNR    true spread / error spread

R and bias answer different questions and the project needs both: a measurement can track
between-specimen variation well (high R) while being systematically off (bias), and a constant
bias is harmless for comparative work whereas failing to rank specimens is fatal.
"""
import glob
import json
import re
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "diagnostics", "morphometrics"))
sys.path.insert(0, REPO)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import measure as ms  # noqa: E402

# §2.2's synthetic table, for direct comparison
SYNTH_R = {"head": 0.958, "leg_prox": 0.756, "mesosoma": 0.849, "mandible": 0.608,
           "gaster": 0.597, "antenna": 0.759, "leg_distal": 0.136}


def main():
    M = ms.load_model()
    bones = ms.bone_table(M)
    names = [str(x) for x in np.asarray(M["dd"]["J_names"]).ravel()] if "dd" in M else None
    with open(os.path.join(REPO, "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl"), "rb") as fh:
        u = pickle._Unpickler(fh)
        u.encoding = "latin1"
        dd = u.load()
    names = [str(x) for x in np.asarray(dd["J_names"]).ravel()]
    Jr = dd["J_regressor"]
    Jr = np.asarray(Jr.todense() if hasattr(Jr, "todense") else Jr, dtype=np.float64)

    fitted = {}
    for p in sorted(glob.glob(os.path.join(REPO, "diagnostics/moonshot/runs/Z8_W*/"
                                                 "Stage_3_deform_fine.npz"))):
        d = np.load(p, allow_pickle=True)
        V = np.asarray(d["verts"], dtype=np.float64)
        Jf = (np.einsum("jv,nvc->njc", Jr, V) if Jr.shape[1] == V.shape[1]
              else np.einsum("vj,nvc->njc", Jr, V))
        for i, lab in enumerate(d["labels"]):
            fitted[str(lab).replace("_processed.obj", "").replace(".obj", "")] = Jf[i]

    bnames = [b["name"] for b in bones]
    G, F, used = [], [], []
    # Annotation source is a CLI argument so the SAME computation can be run against the novice
    # batch and against Fabian's expert-corrected batch. Truth and fit always go through the one
    # code path below; only the truth directory changes.
    ann_dir = sys.argv[1] if len(sys.argv) > 1 else "annotation/gt_batch1"
    print(f"ground truth: {ann_dir}")
    for f in sorted(glob.glob(os.path.join(REPO, ann_dir, "*_joints.json"))):
        gt = json.load(open(f))
        sid = gt["specimen_id"]
        # expert files carry an `_edited` suffix, and one filename doubled its own stem on export
        sid = re.sub(r"_edited$", "", sid)
        sid = re.sub(r"^(.+?)\1$", r"\1", sid)
        if sid not in fitted:
            continue
        P = {j["joint_name"]: np.array(j["position"], dtype=np.float64)
             for j in gt["joints"] if j.get("position") is not None}
        g = np.array([np.linalg.norm(P[names[b["child"]]] - P[names[b["parent"]]])
                      if names[b["child"]] in P and names[b["parent"]] in P else np.nan
                      for b in bones])
        Jf = fitted[sid]
        fl = np.array([np.linalg.norm(Jf[b["child"]] - Jf[b["parent"]]) for b in bones])
        G.append(g)
        F.append(fl)
        used.append(sid)
    G, F = np.array(G), np.array(F)
    print(f"{len(used)} specimens, {len(bnames)} bones from measure.bone_table()")

    # SIZE DIVISOR from the bones present on every specimen, so the Mosimann denominator is one
    # fixed common set. Each feature is then expressed against that divisor and correlated on the
    # specimens where it exists -- pairwise-complete rather than list-wise. Requiring all ten
    # would silently drop the head, mandible and antenna blocks (their joints are placed on 9/10),
    # i.e. exactly the blocks §2.2 rates highest.
    common = ~np.isnan(G).any(0) & (G > 0).all(0) & (F > 0).all(0)
    print(f"bones on ALL specimens (size divisor): {common.sum()} of {len(bnames)}")
    gsize = np.log(G[:, common]).mean(1, keepdims=True)
    fsize = np.log(F[:, common]).mean(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        Gz = np.log(G) - gsize
        Fz = np.log(F) - fsize
    cols = list(bnames)

    rows = []
    for k, c in enumerate(cols):
        m = np.isfinite(Gz[:, k]) & np.isfinite(Fz[:, k])
        if m.sum() < 5:
            continue
        g, f = Gz[m, k], Fz[m, k]
        R = float(np.corrcoef(f, g)[0, 1]) if g.std() > 1e-12 and f.std() > 1e-12 else np.nan
        rows.append({"feature": c, "block": ms.group_of(c.split("_", 1)[1]), "n": int(m.sum()),
                     "R": R, "bias": float(np.median(f - g)),
                     "snr": float(g.std() / max((f - g).std(), 1e-12))})

    print("\n" + "=" * 96)
    print("G2 -- per-BLOCK reliability, REAL ground truth (proportions) vs §2.2's SYNTHETIC table")
    print("=" * 96)
    print(f"{'block':<12}{'feats':>6}{'REAL R':>9}{'synth R':>10}{'drop':>8}"
          f"{'median bias':>13}{'median SNR':>12}")
    blocks = {}
    for b in sorted({r["block"] for r in rows}):
        v = [r for r in rows if r["block"] == b]
        Rr = float(np.nanmedian([r["R"] for r in v]))
        bi = float(np.nanmedian([r["bias"] for r in v]))
        sn = float(np.nanmedian([r["snr"] for r in v]))
        sy = SYNTH_R.get(b, np.nan)
        blocks[b] = {"n": len(v), "R_real": Rr, "R_synth": sy, "bias": bi, "snr": sn}
        print(f"{b:<12}{len(v):>6}{Rr:>9.3f}{sy:>10.3f}{sy-Rr:>8.3f}{bi:>13.3f}{sn:>12.2f}")

    allR = np.array([r["R"] for r in rows], dtype=float)
    print(f"\n  median R across all {len(rows)} features: {np.nanmedian(allR):.3f}")
    print(f"  features with R >= 0.5 : {int((allR >= 0.5).sum())} of {len(rows)}")
    print(f"  features with R <= 0   : {int((allR <= 0).sum())}")

    print("\n" + "=" * 96)
    print("BEST AND WORST FEATURES (real ground truth, proportion)")
    print("=" * 96)
    st = sorted(rows, key=lambda r: -(r["R"] if np.isfinite(r["R"]) else -9))
    print(f"{'BEST':<22}{'R':>7}{'bias':>8}    {'WORST':<22}{'R':>7}{'bias':>8}")
    for i in range(min(10, len(st) // 2)):
        a, b = st[i], st[-1 - i]
        print(f"{a['feature']:<22}{a['R']:>7.3f}{a['bias']:>8.3f}    "
              f"{b['feature']:<22}{b['R']:>7.3f}{b['bias']:>8.3f}")

    json.dump({"blocks": blocks, "features": rows, "specimens": used},
              open(os.path.join(HERE, "g2_results.json"), "w"), indent=2)
    print(f"\nwrote {os.path.join(HERE, 'g2_results.json')}")


if __name__ == "__main__":
    main()
