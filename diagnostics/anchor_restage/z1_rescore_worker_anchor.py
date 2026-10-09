"""Z1 -- re-score the WORKER gen/spread anchor on fits whose shape space was OPEN.

WHY THIS EXISTS
REPORT.md 6.4.1 publishes `worker registrations gen@20/spread = 0.98` and Y1 calibrated its
entire pose ladder against it. That anchor was measured on the LIM_0 / BPX_noprior / M7_handoff
family. REPORT.md 6.7 then established that those arms had the shape space CLOSED -- betas sd
0.00038-0.00050 against a model prior sd of 0.20-1.70, i.e. every specimen received essentially
the same shape -- and that removing `w_beta_prior` raises betas sd 757x. `D1_PROD.yaml`, the
shipped recipe, sets `w_beta_prior: 0.0`.

So the anchor may be a property of a pipeline that no longer exists. Nobody has scored the
metric on real worker registrations produced with an open shape space. The .npz for the
original arms are gone (metrics.csv only), so this is a re-score of DIFFERENT fits, not of
those -- which is exactly the point.

The metric functions are IMPORTED from y1_shape_space_analysis, not reimplemented, so the
number is produced by the same code that produced the Y1 ladder it will be compared against.

SCORES REPORTED PER RUN
  proxy      gen@k / spread            -- the published anchor's own quantity
  mechanism  variance split (parametric shapedirs.betas vs free-form deform_verts),
             and the fraction of centred free-form variance lying inside span(shapedirs)
             -- REFUTE1's R6 quantity, which is what separates the clean corpus (7.3%) from
             the worker corpus (0.9%). A proxy move with no mechanism move is the seventh-
             instance pattern this project now scores for by rule.
  gate       betas sd and mean |z| against the model prior -- VOIDING. If a run is in the
             frozen regime (|z| < 0.05) it is not evidence about the open-shape pipeline.
"""
import argparse
import csv
import glob
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "diagnostics", "pose_causality"))

from y1_shape_space_analysis import KS, loo_gen_over_spread, rest_space_shaped  # noqa: E402

WORKER_TAGS = ("CASENT", "OKENT")


def is_real_worker_run(run_dir):
    """True iff metrics.csv's specimens carry AntScan accession tags."""
    m = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(m):
        return False
    with open(m) as fh:
        r = csv.reader(fh)
        next(r, None)
        row = next(r, None)
    return bool(row) and any(t in row[0] for t in WORKER_TAGS)


def in_span_fraction(F, shapedirs_flat):
    """REFUTE1 R6: fraction of centred free-form variance inside span(shapedirs).

    F is (n, V*3) free-form fields. Projection uses an orthonormal basis of the shapedirs,
    so the fraction is basis-conditioning independent.
    """
    Fc = F - F.mean(axis=0, keepdims=True)
    tot = float((Fc ** 2).sum())
    if tot <= 0:
        return float("nan")
    Q, _ = np.linalg.qr(shapedirs_flat.T)          # (V*3, B) orthonormal
    proj = Fc @ Q                                   # (n, B)
    return float((proj ** 2).sum() / tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", default="diagnostics/moonshot/runs")
    ap.add_argument("--out", default="diagnostics/anchor_restage/out_Z1/z1_worker_anchor.json")
    args = ap.parse_args()

    with open(os.path.join(REPO, os.environ.get(
            "SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")), "rb") as fh:
        u = pickle._Unpickler(fh)
        u.encoding = "latin1"
        dd = u.load()
    v_template = np.asarray(dd["v_template"], dtype=np.float64)
    shapedirs = np.asarray(dd["shapedirs"], dtype=np.float64)          # (V,3,B)
    sd_prior = np.sqrt(np.diag(np.asarray(dd["shape_cov"], dtype=np.float64)))
    B = shapedirs.shape[2]
    sd_flat = shapedirs.transpose(2, 0, 1).reshape(B, -1)              # (B, V*3)

    rows = []
    for rd in sorted(glob.glob(os.path.join(REPO, args.runs_root, "*"))):
        p = os.path.join(rd, "Stage_3_deform_fine.npz")
        if not os.path.exists(p) or not is_real_worker_run(rd):
            continue
        d = np.load(p)
        if "betas" not in d or "deform_verts" not in d:
            continue
        betas = np.asarray(d["betas"], dtype=np.float64)
        dv = np.asarray(d["deform_verts"], dtype=np.float64)
        n = len(betas)

        bsd = betas.std(axis=0)
        z = float((bsd / sd_prior[:len(bsd)]).mean())

        P = rest_space_shaped(v_template, shapedirs, betas).reshape(n, -1)   # parametric only
        F = dv.reshape(n, -1)                                               # free-form only
        X = P + F                                                            # what the metric sees

        sp, gen, rat = loo_gen_over_spread(X, KS)
        _, _, rat_P = loo_gen_over_spread(P, KS)
        _, _, rat_F = loo_gen_over_spread(F, KS)

        var_P = float(((P - P.mean(0)) ** 2).sum())
        var_F = float(((F - F.mean(0)) ** 2).sum())

        rows.append({
            "run": os.path.basename(rd), "n": n,
            "betas_sd": float(bsd.mean()), "z": z,
            "frozen": bool(z < 0.05),
            "spread": sp, "ratio": rat, "ratio_P": rat_P, "ratio_F": rat_F,
            "var_parametric": var_P, "var_freeform": var_F,
            "freeform_share": var_F / (var_P + var_F),
            "in_span_frac": in_span_fraction(F, sd_flat),
        })

    if not rows:
        raise SystemExit("no real-worker runs with fits found")

    print("=" * 108)
    print("Z1 -- worker gen/spread anchor, re-scored on OPEN-shape-space fits")
    print("published anchor (closed shape space, REPORT 6.4.1): gen@10 0.99-1.00, gen@20 0.98")
    print("best closed-era worker arm BPX_noprior: gen@10 0.9353, gen@20 0.9208")
    print("=" * 108)
    print(f"{'run':<30}{'n':>4}{'betas sd':>10}{'|z|':>7}{'gate':>9}"
          + "".join(f"{'gen@'+str(k):>9}" for k in KS))
    for r in rows:
        gate = "FROZEN" if r["frozen"] else "open"
        print(f"{r['run']:<30}{r['n']:>4}{r['betas_sd']:>10.5f}{r['z']:>7.3f}{gate:>9}"
              + "".join(f"{r['ratio'][k]:>9.4f}" for k in KS))

    print()
    print("MECHANISM -- what carries the shape, and does it live in the model's span?")
    print(f"{'run':<30}{'freeform share':>16}{'in-span frac':>14}"
          f"{'gen@20 param':>14}{'gen@20 free':>13}")
    for r in rows:
        print(f"{r['run']:<30}{r['freeform_share']*100:>15.1f}%{r['in_span_frac']*100:>13.3f}%"
              f"{r['ratio_P'][20]:>14.4f}{r['ratio_F'][20]:>13.4f}")
    print("\nREFUTE1 R6 reference: worker LIM_0 0.899% in-span, ALL_ANTS_CLEAN 7.268%, "
          "CLEAN81_M7 24.442%")

    live = [r for r in rows if not r["frozen"]]
    print("\n" + "=" * 108)
    if live:
        g = [r["ratio"][20] for r in live]
        print(f"open-shape worker runs: n={len(live)}  gen@20/spread "
              f"min {min(g):.4f}  median {float(np.median(g)):.4f}  max {max(g):.4f}")
        print(f"published anchor 0.98 -> re-measured median {float(np.median(g)):.4f}")
    else:
        print("NO open-shape worker run available -- Z1 cannot be answered from existing fits")
    print("=" * 108)

    out = os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rows, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
