"""Morphometric Validation Framework (MV) -- is a SMILify-extracted measurement trustworthy?

THE PRINCIPLE THIS EXISTS TO ENFORCE
    A number extracted from an articulated fitted model is NOT automatically a morphometric.
    Four independent things must hold, and they fail independently:

      B1 ANATOMICAL VALIDITY  does the measurement's endpoints sit on the structure it is named
                              after?  (needs expert landmarks; 12 specimens)
      B2 POSE INVARIANCE      does it stay constant when only articulation changes, shape fixed?
                              (needs nothing but a fit; every specimen)
      B3 BIOLOGICAL VALIDITY  does it reproduce a known biological scaling relationship?
                              (needs reference measurements; 20 Atta)
      B4 INTERNAL CONSISTENCY bilateral asymmetry |R-L|/mean. Ants are near-symmetric, so measured
                              L/R disagreement is pipeline error, not biology. Needs NO ground
                              truth -- the only axis computable on all 757 specimens, and therefore
                              the only candidate for a DEPLOYABLE confidence criterion.

    A measurement can be pose-stable and still anatomically wrong (measuring the wrong structure,
    consistently). That is exactly R3/R4's finding, and it is why one axis is not enough.

THE KEY QUESTION THE FRAMEWORK ANSWERS
    Does B4 (free, needs no ground truth) predict B2 (the failure mode that matters)? If yes,
    Fabian gets a per-specimen, per-trait trust score applicable to a corpus with no annotations.

Traits are the GLAD-standard protocol already defined in diagnostics/groundtruth/trait_extract.py
(HL ML SL WL PetL GL HW FL TBL) -- reused, NOT redefined.
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "out")
sys.path.insert(0, REPO)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")
os.environ.setdefault("SMILIFY_COUPLE_JOINT_BLENDSHAPES", "0")
sys.path.insert(0, os.path.join(REPO, "diagnostics", "morphometrics"))
sys.path.insert(0, os.path.join(REPO, "diagnostics", "groundtruth"))

import trait_extract as TX              # noqa: E402
from measure import load_model          # noqa: E402
from fitter_3d.trainer import SMAL3DFitter  # noqa: E402

ATTA_NPZ = os.path.join(REPO, "diagnostics/atta_reference/blender_bundle/ATTA20_ARM_A.npz")
ATTA_REF = os.path.join(REPO, "diagnostics/atta_reference/measurements/"
                              "ant_body_lengths_and_estimated_head_widths.csv")
RECAL = os.path.join(REPO, "annotation/landmark_indices_recalibrated.json")
TEMPL = os.path.join(REPO, "diagnostics/groundtruth/landmark_template_indices.json")

# Landmark naming differs between the rule-computed template file (used by trait_extract) and the
# expert-recalibrated file. Mapped explicitly rather than by string munging.
TEMPL_TO_RECAL = {
    "clypeal_margin_ant_mid": "clypeal_ant_mid",
    "cephalic_margin_post_mid": "cephalic_post_mid",
    "mandibular_apex_r": "mandibular_apex_r",
    "antennal_insertion_r": "antennal_insertion_r",
    "scape_apex_r": "scape_apex_r",
    "wl_anterior_r": "wl_anterior_r",
    "wl_posterior_r": "wl_posterior_r",
}


def forward_pair(npz_path, device="cpu"):
    """(verts_observed, verts_canonical) for every specimen in a fit npz.

    Canonical = the SAME specimen with joint rotation zeroed, betas / per-joint scale /
    per-joint translation / deform_verts left exactly as fitted. Isolates articulation from
    everything else, which is the only thing B2 is allowed to vary.
    """
    z = np.load(npz_path, allow_pickle=True)
    n = z["betas"].shape[0]
    f = SMAL3DFitter(batch_size=n, device=device)
    t = lambda k: torch.tensor(z[k], dtype=torch.float32, device=device)  # noqa: E731
    kw = dict(betas=t("betas"), log_beta_scales=t("log_beta_scales"),
              betas_trans=t("betas_trans"), deform_verts=t("deform_verts"))
    with torch.no_grad():
        obs = f(global_rot=t("global_rot"), joint_rot=t("joint_rot"), trans=t("trans"), **kw)
        can = f(global_rot=torch.zeros_like(t("global_rot")),
                joint_rot=torch.zeros_like(t("joint_rot")),
                trans=torch.zeros_like(t("trans")), **kw)
    labels = [str(x).replace("_processed.obj", "").replace(".obj", "") for x in z["labels"]]
    return obs.numpy().astype(np.float64), can.numpy().astype(np.float64), labels


def recal_landmarks(M):
    """Expert-recalibrated landmark indices in load_landmarks' (right, left) format."""
    d = json.load(open(RECAL))
    mir = TX.mirror_map(M["v_template"])
    out = {}
    for name, e in d.items():
        r = int(e["vertex"])
        out[name] = (r, int(mir[r]))
    return out


def b2_pose_invariance(tr_obs, tr_can):
    """Per-trait, per-specimen |Δ%| between observed and canonical articulation."""
    out = {}
    for k in tr_obs:
        o, c = np.asarray(tr_obs[k], float), np.asarray(tr_can[k], float)
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = np.abs(c - o) / o * 100.0
        out[k] = dict(mean=float(np.nanmean(pct)), max=float(np.nanmax(pct)),
                      per_specimen=[float(x) for x in pct])
    return out


def b4_asymmetry(tr):
    """|R-L| / mean, per bilateral trait per specimen. No ground truth required."""
    out = {}
    for k in TX.BILATERAL:
        if k not in tr or (k + "_l") not in tr:
            continue
        r, l = np.asarray(tr[k], float), np.asarray(tr[k + "_l"], float)
        with np.errstate(divide="ignore", invalid="ignore"):
            a = np.abs(r - l) / ((r + l) / 2.0) * 100.0
        out[k] = dict(mean=float(np.nanmean(a)), max=float(np.nanmax(a)),
                      per_specimen=[float(x) for x in a])
    return out


def b1_anatomical_validity(M):
    """How far the trait-defining landmark indices sit from expert-adjudicated consensus.

    The template file trait_extract actually uses is rule-computed and explicitly
    source='verify' (never anatomically adjudicated). The recalibrated file was fitted to 12
    expert annotations. `moved_pct_diag` is that displacement, in % of the body diagonal -- i.e.
    the anatomical error carried by every trait still computed with the template indices.
    """
    recal = json.load(open(RECAL))
    templ = json.load(open(TEMPL))["landmarks"]
    per_landmark, per_trait = {}, {}
    for tname, rname in TEMPL_TO_RECAL.items():
        e = recal.get(rname)
        if e is None:
            continue
        per_landmark[tname] = dict(
            recal_name=rname, template_vertex=templ.get(tname, {}).get("vertex"),
            recal_vertex=e["vertex"], old_vertex=e.get("old_vertex"),
            moved_pct_diag=e.get("moved_pct_diag"),
            expert_agreement_pct_diag=e.get("agreement_pct_diag"), n_experts=e.get("n"),
            adjudicated=True)
    for name in templ:
        if name not in per_landmark:
            per_landmark[name] = dict(recal_name=None, template_vertex=templ[name]["vertex"],
                                      adjudicated=False,
                                      note="no expert annotation exists for this landmark")
    for tname, (kind, spec, status, note) in TX.TRAITS.items():
        if kind == "pair":
            pts = [p for p in spec]
            adj = [per_landmark.get(p, {}).get("adjudicated", False) for p in pts]
            moved = [per_landmark.get(p, {}).get("moved_pct_diag") for p in pts]
            moved = [m for m in moved if m is not None]
            per_trait[tname] = dict(
                kind=kind, status=status, endpoints=pts, all_endpoints_adjudicated=all(adj),
                worst_moved_pct_diag=(max(moved) if moved else None),
                verdict=("ADJUDICATED" if all(adj) else "UNADJUDICATED"))
        else:
            per_trait[tname] = dict(kind=kind, status=status, endpoints=list(spec),
                                    all_endpoints_adjudicated=False,
                                    worst_moved_pct_diag=None,
                                    verdict=("RIG_JOINT_NOT_SURFACE" if kind == "joint"
                                             else "NOT_LANDMARK_BASED"))
    return per_landmark, per_trait


def b3_biological_validity(tr_obs, labels, M, joints_obs):
    """Allometric scaling of model-derived traits against Fabian's physical reference (20 Atta).

    CRITICAL FRAME CORRECTION. fitter_3d/utils.py::load_meshes normalises EVERY specimen
    independently (centre on vertex mean, divide by max|coord|), so a trait in model units has had
    its absolute size stripped -- a 1.1 mg minor and a 47 mg major are both ~unit-sized. Regressing
    raw model-unit traits against real body length therefore yields exponent ~0 for EVERYTHING,
    which is an artefact of the frame, not a biological null. (This was caught here by the HW
    positive control below contradicting the established Atta result, exactly what it is for.)

    Fix, identical to the Atta head-width replication's own protocol: restore physical scale with
    each specimen's OWN factor, scale_i = BL_mm_i / BL_model_i, taken on the same `b_t`-`b_a_5`
    joint pair the reference CSV names, then regress in mm.

    STRUCTURAL CAVEAT, stated because it changes how every number here reads: since scale_i itself
    contains BL_mm, log(T_mm) = log(T_model) - log(BL_model) + log(BL_mm), so a trait carrying no
    independent size information lands at exponent ~= 1.0, NOT 0. The null for this table is
    therefore 1.0: >1 is positive allometry, <1 negative, ~1 is "tracks size isometrically / adds
    nothing beyond body length". The published reference exponents are computed from purely
    physical measurements and carry no such structure, so a model-vs-reference comparison is a
    genuine test of whether the ratio scales the way the real animal's does.
    """
    import csv
    from scipy import stats
    ref = {}
    with open(ATTA_REF) as fh:
        for row in csv.DictReader(fh):
            ref[int(row["Shape"])] = (float(row["b_t to b_a_5 [mm]"]),
                                      float(row["head width [mm]"]), float(row["mass [mg]"]))
    jn = M["jnames"]
    bt, ba5 = jn.index("b_t"), jn.index("b_a_5")
    bl_model_all = np.linalg.norm(joints_obs[:, bt] - joints_obs[:, ba5], axis=-1)

    idx, bl, hw_ref, mass = [], [], [], []
    for i, lab in enumerate(labels):
        try:
            s = int(str(lab).split("_")[0])
        except ValueError:
            continue
        if s in ref:
            idx.append(i); bl.append(ref[s][0]); hw_ref.append(ref[s][1]); mass.append(ref[s][2])
    idx = np.array(idx); bl = np.array(bl); hw_ref = np.array(hw_ref); mass = np.array(mass)
    out = {"n": int(len(idx)), "traits": {},
           "null_exponent_vs_body_length": 1.0,
           "reference": {"head_width~body_length": 1.2352, "head_width~mass": 0.3949}}
    if len(idx) < 5:
        out["note"] = "too few matched specimens for scaling"
        return out

    scale = bl / bl_model_all[idx]          # mm per model unit, per specimen
    for k, v in tr_obs.items():
        y = np.asarray(v, float)[idx] * scale        # -> mm
        if not np.all(np.isfinite(y)) or np.any(y <= 0):
            continue
        row = {}
        for xname, x in (("body_length_mm", bl), ("mass_mg", mass)):
            s, ic, r, p, se = stats.linregress(np.log10(x), np.log10(y))
            tc = stats.t.ppf(0.975, len(idx) - 2)
            row[xname] = dict(exponent=float(s), ci_lo=float(s - tc * se),
                              ci_hi=float(s + tc * se), r2=float(r ** 2), p=float(p))
        out["traits"][k] = row

    # POSITIVE CONTROL. The reference head width is a directly measured physical quantity, so
    # regressing it against reference body length must reproduce the published exponent (1.2352).
    # If this control fails, nothing else in B3 is trustworthy and the table must not be read.
    s, ic, r, p, se = stats.linregress(np.log10(bl), np.log10(hw_ref))
    tc = stats.t.ppf(0.975, len(idx) - 2)
    out["positive_control_reference_HW_vs_BL"] = dict(
        exponent=float(s), ci_lo=float(s - tc * se), ci_hi=float(s + tc * se), r2=float(r ** 2),
        published=1.2352,
        passes=bool(abs(s - 1.2352) < 0.10))
    return out


def main():
    M = load_model()
    lm_templ = TX.load_landmarks(M)
    lm_recal_raw = recal_landmarks(M)
    # trait_extract addresses landmarks by TEMPLATE names; remap the recalibrated set onto them
    # so the SAME trait definitions can be evaluated under both index sets.
    lm_recal = dict(lm_templ)
    for tname, rname in TEMPL_TO_RECAL.items():
        if rname in lm_recal_raw:
            lm_recal[tname] = lm_recal_raw[rname]

    obs, can, labels = forward_pair(ATTA_NPZ)
    J_obs = np.einsum("ij,njk->nik", M["Jr"], obs)
    print(f"[mv] ATTA20: {obs.shape} observed, {can.shape} canonical, {len(labels)} labels")

    res = {"corpus": "ATTA20_ARM_A", "n_specimens": len(labels), "labels": labels,
           "model": os.environ["SMILIFY_SMAL_FILE"],
           "landmark_sets": {"template": TEMPL, "recalibrated": RECAL}}

    for setname, lm in (("template", lm_templ), ("recalibrated", lm_recal)):
        tr_o = TX.traits(obs, M=M, lm=lm)
        tr_c = TX.traits(can, M=M, lm=lm)
        res.setdefault("B2_pose_invariance", {})[setname] = b2_pose_invariance(tr_o, tr_c)
        res.setdefault("B4_asymmetry", {})[setname] = b4_asymmetry(tr_o)
        res.setdefault("B3_biological_validity", {})[setname] = b3_biological_validity(
            tr_o, labels, M, J_obs)
        res.setdefault("_trait_values_observed", {})[setname] = {
            k: [float(x) for x in v] for k, v in tr_o.items()}

    pl, pt = b1_anatomical_validity(M)
    res["B1_anatomical_validity"] = {"per_landmark": pl, "per_trait": pt}

    # ---- Does B4 (free) predict B2 (what matters)? Per-specimen, within trait. ----
    from scipy import stats as st
    link = {}
    for setname in ("template", "recalibrated"):
        b2, b4 = res["B2_pose_invariance"][setname], res["B4_asymmetry"][setname]
        rows = {}
        for k in b4:
            if k not in b2:
                continue
            a = np.asarray(b4[k]["per_specimen"], float)
            p = np.asarray(b2[k]["per_specimen"], float)
            ok = np.isfinite(a) & np.isfinite(p)
            if ok.sum() >= 5:
                r, pv = st.pearsonr(a[ok], p[ok])
                rs, ps = st.spearmanr(a[ok], p[ok])
                rows[k] = dict(n=int(ok.sum()), pearson_r=float(r), pearson_p=float(pv),
                               spearman_r=float(rs), spearman_p=float(ps))
        link[setname] = rows
    res["B4_predicts_B2"] = link

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "mv_results.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"[mv] wrote {os.path.join(OUT, 'mv_results.json')}")

    print("\n=== B2 pose invariance (recalibrated indices) ===")
    for k, v in sorted(res["B2_pose_invariance"]["recalibrated"].items(),
                       key=lambda kv: kv[1]["mean"]):
        print(f"  {k:8s} mean {v['mean']:7.3f}%   max {v['max']:8.3f}%")
    print("\n=== B4 bilateral asymmetry (recalibrated) ===")
    for k, v in sorted(res["B4_asymmetry"]["recalibrated"].items(), key=lambda kv: kv[1]["mean"]):
        print(f"  {k:8s} mean {v['mean']:7.3f}%   max {v['max']:8.3f}%")
    print("\n=== B4 predicts B2? (per-specimen, within trait) ===")
    for k, v in res["B4_predicts_B2"]["recalibrated"].items():
        print(f"  {k:8s} n={v['n']:3d}  pearson r={v['pearson_r']:+.3f} (p={v['pearson_p']:.3f})"
              f"   spearman r={v['spearman_r']:+.3f} (p={v['spearman_p']:.3f})")
    print("\n=== B1 anatomical validity (per trait) ===")
    for k, v in res["B1_anatomical_validity"]["per_trait"].items():
        mv = v.get("worst_moved_pct_diag")
        print(f"  {k:8s} {v['verdict']:22s} status={v['status']:18s}"
              + (f" worst_landmark_moved={mv:.2f}% diag" if mv is not None else ""))


if __name__ == "__main__":
    main()
