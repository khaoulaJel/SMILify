"""M2 -- synthetic morphometric recovery: when the fit is wrong, how wrong is the MEASUREMENT?

THE QUESTION NOTHING ELSE IN THIS PROJECT ANSWERS
    Atta tells us the measurements agree with a physical reference on 20 good fits. It cannot tell
    us how measurement error behaves when fitting goes wrong, because we have no ground truth for
    what the measurement should have been. The synthetic corpus does: `synth_power48` was GENERATED
    from the model, so every trait has an exact known value.

    The decisive question: **does Chamfer -- the thing the objective actually optimises, and the
    number everyone reports -- predict morphometric error?** R3/R4/R9 predict no. M2 measures it.

WHY THIS CORPUS IS UNUSUALLY STRONG FOR THE PURPOSE
    The synthetic specimens share the template topology, so vertex i in the fit corresponds to
    vertex i in ground truth BY CONSTRUCTION. That gives a TRUE per-vertex correspondence error,
    not a chamfer-style nearest-neighbour proxy -- the distinction this whole project turns on.
    Ground-truth betas / joint_rot / log_beta_scales are stored too, so pose and shape error are
    also exact rather than inferred.

Everything runs on existing fits. No new fitting.
"""
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "out")
sys.path.insert(0, REPO)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")
os.environ.setdefault("SMILIFY_COUPLE_JOINT_BLENDSHAPES", "0")
sys.path.insert(0, os.path.join(REPO, "diagnostics", "morphometrics"))
sys.path.insert(0, os.path.join(REPO, "diagnostics", "groundtruth"))

import trait_extract as TX          # noqa: E402
from measure import load_model      # noqa: E402

GT = os.path.join(REPO, "diagnostics/moonshot/synth_power48/ground_truth.npz")
ARMS = {"P48_zero": "no correspondence supervision (closest to production)",
        "P48_cse_all": "with CSE correspondence supervision",
        "C13_uniform": "C13 uniform tolerance arm"}
TRAITS = ("HW", "HL", "ML", "SL", "WL", "PetL", "GL", "FL", "TBL")


def chamfer(a, b):
    """Symmetric mean nearest-neighbour distance -- what the objective optimises."""
    from scipy.spatial import cKDTree
    da, _ = cKDTree(b).query(a)
    db, _ = cKDTree(a).query(b)
    return float((da.mean() + db.mean()) / 2)


def main():
    M = load_model()
    gt = np.load(GT, allow_pickle=True)
    V_gt = gt["verts"].astype(np.float64)
    gt_names = [str(x) for x in gt["names"]]
    J_gt = np.einsum("ij,njk->nik", M["Jr"], V_gt)

    results = {"corpus": "synth_power48", "n": len(gt_names), "arms": {}}

    for arm, desc in ARMS.items():
        path = os.path.join(REPO, f"diagnostics/moonshot/runs/{arm}/Stage_3_deform_fine.npz")
        if not os.path.exists(path):
            print(f"[m2] SKIP {arm}: no fit found")
            continue
        fit = np.load(path, allow_pickle=True)
        V_fit = fit["verts"].astype(np.float64)
        fit_names = [str(x).replace(".obj", "") for x in fit["labels"]]

        # ---- GATE 1: specimen alignment. Order must match or every number is meaningless. ----
        if fit_names != gt_names:
            print(f"[m2] {arm}: label order differs -- reindexing by name")
            order = [fit_names.index(n) for n in gt_names]
            V_fit = V_fit[order]
            fit = {k: (fit[k][order] if getattr(fit[k], "shape", (0,))[:1] == (len(order),)
                       else fit[k]) for k in fit.files}
        else:
            fit = {k: fit[k] for k in fit.files}

        # ---- GATE 2: frame check. A global scale/offset mismatch between GT and fitted frames
        # would bias every trait. Compare per-specimen bounding-box extent; must be ~1.0.
        ext_gt = (V_gt.max(1) - V_gt.min(1)).mean(-1)
        ext_fit = (V_fit.max(1) - V_fit.min(1)).mean(-1)
        ratio = ext_fit / ext_gt
        frame_ok = bool(abs(np.median(ratio) - 1.0) < 0.05)
        print(f"[m2] {arm}: frame ratio median={np.median(ratio):.4f} "
              f"[{ratio.min():.3f},{ratio.max():.3f}]  OK={frame_ok}")

        # ---- exact errors, available only because topology is shared ----
        vtx_err = np.linalg.norm(V_fit - V_gt, axis=-1).mean(-1)      # TRUE correspondence error
        cham = np.array([chamfer(V_fit[i], V_gt[i]) for i in range(len(gt_names))])
        J_fit = np.einsum("ij,njk->nik", M["Jr"], V_fit)
        joint_err = np.linalg.norm(J_fit - J_gt, axis=-1).mean(-1)
        pose_err = np.abs(fit["joint_rot"].astype(np.float64)
                          - gt["joint_rot"].astype(np.float64)).mean((1, 2))
        beta_err = np.abs(fit["betas"].astype(np.float64)
                          - gt["betas"].astype(np.float64)).mean(-1)
        deform = np.linalg.norm(fit["deform_verts"].astype(np.float64), axis=-1).mean(-1)
        scale_norm = ext_gt   # normaliser so errors are comparable across specimen sizes

        diags = {"chamfer": cham / scale_norm, "vertex_corr_err": vtx_err / scale_norm,
                 "joint_err": joint_err / scale_norm, "pose_err_rad": pose_err,
                 "beta_err": beta_err, "deform_mag": deform / scale_norm}

        tr_gt = TX.traits(V_gt, M=M)
        tr_ft = TX.traits(V_fit, M=M)
        trait_err, trait_err_shape = {}, {}
        for t in TRAITS:
            if t not in tr_gt:
                continue
            g, f_ = np.asarray(tr_gt[t], float), np.asarray(tr_ft[t], float)
            trait_err[t] = np.abs(f_ - g) / g * 100.0
            # SCALE-CORRECTED: divide each specimen's trait by its own frame extent, so a fit that
            # is merely the wrong SIZE no longer counts as a wrong SHAPE. Isolates the part of the
            # error the objective's scale terms cannot excuse.
            gs, fs = g / ext_gt, f_ / ext_fit
            trait_err_shape[t] = np.abs(fs - gs) / gs * 100.0

        # ---- THE decisive analysis: which diagnostic predicts morphometric error? ----
        pred = {}
        for dname, dv in diags.items():
            pred[dname] = {}
            for t, te in trait_err.items():
                ok = np.isfinite(te) & np.isfinite(dv)
                if ok.sum() < 8:
                    continue
                r, p = stats.pearsonr(dv[ok], te[ok])
                rs, ps = stats.spearmanr(dv[ok], te[ok])
                pred[dname][t] = dict(pearson_r=float(r), pearson_p=float(p),
                                      spearman_r=float(rs), spearman_p=float(ps), n=int(ok.sum()))

        results["arms"][arm] = dict(
            description=desc, frame_ratio_median=float(np.median(ratio)), frame_ok=frame_ok,
            diagnostics={k: [float(x) for x in v] for k, v in diags.items()},
            diagnostics_summary={k: dict(mean=float(np.mean(v)), median=float(np.median(v)))
                                 for k, v in diags.items()},
            trait_error_pct={t: [float(x) for x in v] for t, v in trait_err.items()},
            trait_error_summary={t: dict(mean=float(np.mean(v)), median=float(np.median(v)),
                                         p90=float(np.percentile(v, 90)), max=float(np.max(v)))
                                 for t, v in trait_err.items()},
            trait_error_shape_only={t: dict(mean=float(np.mean(v)), median=float(np.median(v)),
                                            p90=float(np.percentile(v, 90)), max=float(np.max(v)))
                                    for t, v in trait_err_shape.items()},
            predicts=pred)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "m2_results.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    for arm, r in results["arms"].items():
        print(f"\n{'='*100}\n{arm} -- {r['description']}\n{'='*100}")
        print("trait recovery error (% of true value):   [raw = incl. scale | shape = scale-corrected]")
        for t, ss in sorted(r["trait_error_summary"].items(), key=lambda kv: kv[1]["median"]):
            sh = r["trait_error_shape_only"][t]
            print(f"  {t:6s} raw median {ss['median']:6.2f}%  p90 {ss['p90']:7.2f}%   |   "
                  f"shape median {sh['median']:6.2f}%  p90 {sh['p90']:7.2f}%")
        print("\ndoes each diagnostic predict trait error?  (spearman r across 48 specimens)")
        hdr = f"  {'diagnostic':18s}" + "".join(f"{t:>9s}" for t in TRAITS if t in r["predicts"]["chamfer"])
        print(hdr)
        for dname in ("chamfer", "vertex_corr_err", "joint_err", "pose_err_rad", "beta_err",
                      "deform_mag"):
            if dname not in r["predicts"]:
                continue
            line = f"  {dname:18s}"
            for t in TRAITS:
                if t not in r["predicts"][dname]:
                    continue
                v = r["predicts"][dname][t]
                star = "*" if v["spearman_p"] < 0.05 else " "
                line += f"{v['spearman_r']:>8.2f}{star}"
            print(line)
    print(f"\nwrote {os.path.join(OUT, 'm2_results.json')}   (* = p<0.05)")


if __name__ == "__main__":
    main()
