"""Fork of diagnostics/moonshot/hull/within_part_signal.py -- same question, same protocol,
same controls, DINOv2 in place of the Heat Kernel Signature.

WHY THIS FORK EXISTS
within_part_signal.py asked "does an INTRINSIC descriptor (HKS) recover within-part
correspondence better than the fitted pipeline?" and got a weak-but-technically-positive
answer: HKS clears its own 3x-over-chance bar (0.66% vs 0.14%) but loses to the pipeline on
placement error (12.91% vs 4.32% median) -- FINAL_REPORT's "COND -- kills intrinsic-only, not
the learned-extrinsic family". T1.0 (diagnostics/khaoula_v2/t10_diff3f_sanity.py) then showed
raw DINOv2/SD-turbo features carry real anatomical semantic signal at the COARSE (5-group)
level (ARI 0.428-0.447 vs ~0.000 chance) -- decisively, unlike HKS's marginal pass. This
script is the actual T1.1 test: does that signal survive being asked the SAME hard question
HKS was asked, restricted to the correct part, scored against the same E6 ground truth and
the same `fitted`/`spatial`/`random_within` controls?

WHAT CHANGED FROM THE ORIGINAL, PRECISELY (everything else is untouched)
  - `hks(v, f, n_eig, n_times)` -> `dino_descriptor(v, f)`. Per T1.0's finding (DINO alone:
    0.797/0.428, barely behind `combined`'s 0.799/0.447, and far cheaper than adding SD's
    VAE+UNet pass), this uses DINO ONLY, not SD, not the combined feature -- a deliberate
    scope-narrowing decision, not an oversight.
  - Coverage is NOT total. HKS is defined at every vertex (it's intrinsic, computed from the
    mesh's own connectivity). DINO here is EXTRINSIC and view-based (8 renders + backprojection,
    reusing T1.0's already-validated `render_views`/`dino_features`/`backproject`): T1.0 saw
    92.6% vertex coverage on the template alone. Every score() call below is restricted to the
    intersection of template-seen and target-seen vertices for that specimen, and the seen
    fraction is reported per specimen -- silently scoring over a shrunken, occlusion-biased
    subset without saying so would misrepresent what "% correct" means here.
  - `spatial` and `fitted` controls are UNCHANGED -- pure vertex-position geometry, no
    descriptor involved -- so they are a direct internal-consistency check: they should
    reproduce (up to ~run choice of `--run`) the original script's own `spatial`/`fitted` rows.
  - Default `--n` lowered from 6 to 4 and views/specimen fixed at 8 (T1.0's setting) -- DINO
    rendering+extraction per specimen is far cheaper per-call than HKS's eigendecomposition,
    but doing it `--n` times each requiring 8 renders adds up; 4 specimens is enough to read
    the pre-registered gate below without over-spending compute on a probe that might fail it.

PRE-REGISTERED READING, restated from the parent brief (T1.1), fixed before running:
  DINO must beat `hks_within`'s 12.91% median error by a wide margin -- NOT merely clear
  `random_within`'s chance level, which HKS already showed is an almost-worthless bar. Ideally
  it should land meaningfully below the fitted pipeline's own 4.32% (the number that would
  actually justify building T1.2's full matching pipeline). Kill condition, set before running:
  if within-part DINO does not land clearly below 12.91%, stop here -- do not scope T1.2.

ADDENDUM, added after the first run (result: DINO 10.34% vs HKS's 12.91% reference -- an
apparent win that does NOT clear the pre-registered "half of 12.91%" bar): DINO's number was
computed over only 77.0% mean coverage (vertices visible from BOTH the template's and the
posed target's 8 views -- occlusion shifts with pose, so intersection coverage is lower than
T1.0's 92.6% single-mesh figure). HKS's 12.91% reference is computed over 100% coverage
(intrinsic, defined everywhere). Occlusion is not random -- it plausibly favours easier,
more-visible geometry over cluttered leg-joint/gaster-underside regions -- so DINO's apparent
edge over HKS could be entirely a population-selection artifact: an easier vertex subset,
not a better descriptor. Fixed by also computing HKS restricted to the EXACT SAME per-specimen
joint-coverage mask DINO used (`hks_within_matched` below), alongside a full-coverage HKS
reproduction (`hks_within_full`, sanity-checking this run reproduces ~12.91% before trusting
anything computed alongside it). Only `hks_within_matched` vs `dino_within` is a fair
comparison; `dino_within` vs the original 12.91% reference is not, and must not be read as
one going forward.
"""

import argparse
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MOON = os.path.abspath(os.path.join(HERE, "..", "moonshot"))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)  # t10_diff3f_sanity.py lives next to this file
sys.path.insert(0, MOON)  # trainer_hierarchical.anatomical_groups needs REPO on path too
sys.path.insert(0, os.path.join(MOON, "hull"))  # original within_part_signal.py's hks()
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")
os.environ.setdefault("HF_HOME", "/p/scratch/cias-7/jellal1/hf_cache")

from scipy.spatial import cKDTree  # noqa: E402
from within_part_signal import hks  # noqa: E402  -- the ORIGINAL, unmodified function


DINO_CACHE = os.path.join(HERE, "out", "dino_cache")


def dino_descriptor(v_np, f_np, cache_name=None, refine_model=None):
    """Per-vertex DINOv2 feature via T1.0's 8-view render + nearest-vertex backprojection,
    optionally refined by a trained RefineNet encoder (T1.1 refinement-layer addendum).

    Returns (feat, seen): feat is (V, C) float64 (C=768 raw, or `embed_dim` if refined;
    zero rows where unseen -- callers MUST use `seen` to exclude those), seen is (V,) bool.

    cache_name: if given and `diagnostics/khaoula_v2/out/dino_cache/{cache_name}.npz`
    exists (written by train_refine_autoencoder.py or a prior run of this function),
    reuse it instead of re-rendering/re-extracting -- e.g. `template`, `synth_000`, so
    T1.1's original run and the refinement-layer eval share the same raw DINO features
    for the SAME specimens, and neither re-pays extraction the other already did.
    """
    if cache_name is not None:
        cp = os.path.join(DINO_CACHE, f"{cache_name}.npz")
        if os.path.isfile(cp):
            d = np.load(cp)
            feat, seen = d["feat"], d["seen"]
        else:
            feat, seen = _extract_dino(v_np, f_np)
            os.makedirs(DINO_CACHE, exist_ok=True)
            np.savez(cp, feat=feat, seen=seen)
    else:
        feat, seen = _extract_dino(v_np, f_np)

    if refine_model is not None:
        import torch

        with torch.no_grad():
            x = torch.tensor(feat, dtype=torch.float32, device=next(refine_model.parameters()).device)
            z, _ = refine_model(x)
            feat = z.cpu().numpy().astype(np.float64)

    return feat.astype(np.float64), seen


def _extract_dino(v_np, f_np):
    import torch
    from t10_diff3f_sanity import render_views, dino_features, backproject, DEV

    v = torch.tensor(v_np, dtype=torch.float32, device=DEV)
    c = v.mean(0)
    v = (v - c) / (v - c).abs().max()
    f = torch.tensor(f_np, dtype=torch.int64, device=DEV)

    imgs, rasterizers = render_views(v, f, n_views=8, image_size=512)
    dmaps = dino_features(imgs)
    feat, seen = backproject(v, f, rasterizers, dmaps)
    return feat.astype(np.float64), seen


# ======================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--run", default="SYN_clean")
    ap.add_argument("--gt_groups", default="16", choices=["7", "13", "16"])
    ap.add_argument("--refined", default=None,
                     help="path to a refine_net.pt from train_refine_autoencoder.py; if given, "
                          "score the REFINED descriptor (T1.1 refinement-layer addendum) instead "
                          "of raw DINO. Uses cache_name= for template/synth_000-003 so this run "
                          "and T1.1's original run share the same raw DINO extraction where the "
                          "specimen overlaps.")
    args = ap.parse_args()

    refine_model = None
    if args.refined is not None:
        import torch
        from refine_net import RefineNet

        ckpt = torch.load(args.refined, map_location="cpu")
        refine_model = RefineNet(in_dim=768, embed_dim=ckpt["embed_dim"])
        refine_model.load_state_dict(ckpt["state_dict"])
        refine_model = refine_model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        print(f"loaded refinement model from {args.refined} "
              f"(trained on {ckpt['train_specimens']}, embed_dim={ckpt['embed_dim']})")

    import config
    from fitter_3d.trainer_hierarchical import anatomical_groups

    with open(os.path.join(REPO, config.SMAL_FILE), "rb") as fh:
        u = pickle._Unpickler(fh)
        u.encoding = "latin1"
        dd = u.load()
    jn = [str(x) for x in dd["J_names"]]
    sd, sa = args.gt_groups in ("13", "16"), args.gt_groups == "16"
    jg = anatomical_groups(jn, split_distal=sd, split_anterior=sa)
    gnames = sorted(set(jg.values()))
    n2i = {nm: i for i, nm in enumerate(gnames)}
    vpart = np.array([n2i[jg[int(d)]] for d in np.asarray(dd["weights"]).argmax(1)], dtype=np.int64)

    v_tpl = np.asarray(dd["v_template"], dtype=np.float64)
    faces = np.asarray(dd["f"]).astype(np.int64)
    V = len(v_tpl)

    print(f"template {V} verts, {len(faces)} faces, {len(gnames)} anatomical groups")
    print("computing DINO descriptor on the template (8 views) ...\n", flush=True)
    d_tpl, seen_tpl = dino_descriptor(v_tpl, faces, cache_name="template", refine_model=refine_model)
    print(f"  template vertex coverage: {seen_tpl.sum()}/{V} ({100 * seen_tpl.mean():.1f}%)")
    print("computing HKS descriptor on the template (matched-coverage addendum) ...", flush=True)
    d_tpl_hks = hks(v_tpl, faces, n_eig=120)

    gt = np.load(os.path.join(MOON, "synth_clean", "ground_truth.npz"))
    gtv = gt["verts"]
    names = [str(x) for x in gt["names"]]

    # look in this session's own khaoula_v2/runs first (e.g. T04_baseline -- the honest
    # current-best D1 recipe from Tier 0, with real data), fall back to moonshot/runs
    # (the original branch's run directories, several of which -- SYN_clean, SYN_D1 -- are
    # NOT present on disk despite being referenced in reports; only their scored summary
    # survives in moonshot/out/synth_scores.npz, not the raw per-vertex npz this needs)
    fitp_local = os.path.join(HERE, "runs", args.run, "Stage_3_deform_fine.npz")
    fitp_moon = os.path.join(MOON, "runs", args.run, "Stage_3_deform_fine.npz")
    fitp = fitp_local if os.path.isfile(fitp_local) else fitp_moon
    fitd = np.load(fitp) if os.path.isfile(fitp) else None
    print(f"  fitted-control source: {fitp if fitd is not None else '(none found -- fitted row will be empty)'}")
    fit_labels = [str(x) for x in fitd["labels"]] if fitd is not None else []

    rng = np.random.default_rng(0)
    acc = {k: [] for k in ("dino_global", "dino_within", "spatial", "fitted", "random_within",
                            "hks_within_full", "hks_within_matched")}
    err = {k: [] for k in acc}
    ident = np.arange(V)
    coverage_log = []

    for si in range(min(args.n, len(names))):
        tgt = gtv[si].astype(np.float64)
        print(f"  {names[si]}: rendering + extracting DINO (8 views) ...", flush=True)
        d_tgt, seen_tgt = dino_descriptor(tgt, faces, cache_name=names[si], refine_model=refine_model)
        seen = seen_tpl & seen_tgt
        coverage_log.append(float(seen.mean()))
        print(f"  {names[si]}: computing HKS (matched-coverage addendum) ...", flush=True)
        d_tgt_hks = hks(tgt, faces, n_eig=120)

        # frame-match: both meshes normalised the way the loader does
        def norm(x):
            c = x.mean(0)
            return (x - c) / np.abs(x - c).max()

        tn = norm(tgt)
        extent = 1.0

        def score(match, key, mask):
            m = mask
            acc[key].append(float((match[m] == ident[m]).mean()))
            err[key].append(np.linalg.norm(tn[match[m]] - tn[ident[m]], axis=1) / extent)

        # --- DINO nearest neighbour, globally (restricted to jointly-seen vertices)
        m_global = np.full(V, -1, dtype=np.int64)
        idx_seen = np.nonzero(seen)[0]
        m_global[idx_seen] = idx_seen[cKDTree(d_tgt[idx_seen]).query(d_tpl[idx_seen], k=1)[1]]
        score(m_global, "dino_global", seen)

        # --- DINO nearest neighbour, restricted to the CORRECT part AND jointly-seen
        #     (the oracle-part case, same as HKS's `hks_within`)
        m_within = np.full(V, -1, dtype=np.int64)
        m_rand = np.full(V, -1, dtype=np.int64)
        within_mask = np.zeros(V, dtype=bool)
        for g in range(len(gnames)):
            idx = np.nonzero((vpart == g) & seen)[0]
            if len(idx) == 0:
                continue
            m_within[idx] = idx[cKDTree(d_tgt[idx]).query(d_tpl[idx], k=1)[1]]
            m_rand[idx] = idx[rng.integers(0, len(idx), len(idx))]
            within_mask[idx] = True
        score(m_within, "dino_within", within_mask)
        score(m_rand, "random_within", within_mask)

        # --- ADDENDUM: HKS within-part, full coverage (reproduction check -- should land
        #     near the original probe's 12.91%) and matched-coverage (restricted to the SAME
        #     `seen` mask DINO used above -- the only fair comparison to `dino_within`)
        m_hks_full = np.empty(V, dtype=np.int64)
        m_hks_matched = np.full(V, -1, dtype=np.int64)
        matched_mask = np.zeros(V, dtype=bool)
        for g in range(len(gnames)):
            idx_full = np.nonzero(vpart == g)[0]
            if len(idx_full):
                m_hks_full[idx_full] = idx_full[cKDTree(d_tgt_hks[idx_full]).query(d_tpl_hks[idx_full], k=1)[1]]
            idx_m = np.nonzero((vpart == g) & seen)[0]
            if len(idx_m):
                m_hks_matched[idx_m] = idx_m[cKDTree(d_tgt_hks[idx_m]).query(d_tpl_hks[idx_m], k=1)[1]]
                matched_mask[idx_m] = True
        score(m_hks_full, "hks_within_full", np.ones(V, dtype=bool))
        score(m_hks_matched, "hks_within_matched", matched_mask)
        assert np.array_equal(matched_mask, within_mask), "HKS/DINO matched-coverage masks diverged"

        # --- spatial NN from the template's own rest pose (a floor) -- UNCHANGED, no descriptor
        full_mask = np.ones(V, dtype=bool)
        score(cKDTree(tn).query(norm(v_tpl), k=1)[1], "spatial", full_mask)

        # --- the pipeline's own answer, same protocol -- UNCHANGED, no descriptor
        if fitd is not None and names[si] in [x[:-4] if x.endswith(".obj") else x for x in fit_labels]:
            bi = [x[:-4] if x.endswith(".obj") else x for x in fit_labels].index(names[si])
            fv = fitd["verts"][bi].astype(np.float64)
            score(cKDTree(tn).query(fv, k=1)[1], "fitted", full_mask)
        print(f"  {names[si]}  done (joint coverage {100 * seen.mean():.1f}%)", flush=True)

    print(f"\nmean joint (template x target) vertex coverage across specimens: "
          f"{100 * np.mean(coverage_log):.1f}%")

    print(f"\n{'method':<16}{'correct %':>11}{'median err':>13}{'p90 err':>10}   what it is")
    print("-" * 92)
    desc = {
        "fitted": "the pipeline's own result",
        "spatial": "nearest target vertex to the TEMPLATE rest pose",
        "random_within": "chance level inside the correct part (jointly-seen verts only)",
        "dino_global": "DINOv2 descriptor NN, whole mesh (jointly-seen verts only)",
        "dino_within": "DINOv2 descriptor NN, correct part given (jointly-seen verts only)",
        "hks_within_full": "ADDENDUM: HKS, correct part given, FULL 100% coverage (reproduction check)",
        "hks_within_matched": "ADDENDUM: HKS, correct part given, SAME jointly-seen verts as dino_within",
    }
    for k in ("fitted", "spatial", "random_within", "dino_global", "dino_within",
              "hks_within_full", "hks_within_matched"):
        if not acc[k]:
            continue
        e = np.concatenate(err[k])
        print(
            f"{k:<16}{100 * float(np.mean(acc[k])):>10.2f}%{100 * float(np.median(e)):>12.2f}%"
            f"{100 * float(np.percentile(e, 90)):>9.2f}%   {desc[k]}"
        )

    print("\nREADING")
    result = dict(pass_condition=None, dino_within_median_err=None, hks_within_reference=0.1291,
                  hks_within_full_this_run=None, hks_within_matched_this_run=None,
                  fitted_reference_from_this_run=None, matched_comparison_valid=None)
    if acc["dino_within"] and acc["random_within"]:
        hw, rw = float(np.mean(acc["dino_within"])), float(np.mean(acc["random_within"]))
        print(
            f"  within-part DINO  {100 * hw:.2f}%  vs chance {100 * rw:.2f}%  -> "
            f"{'signal present' if hw > 3 * max(rw, 1e-9) else 'NO signal above chance'}"
        )
        me = np.median(np.concatenate(err["dino_within"]))
        mf = np.median(np.concatenate(err["fitted"])) if acc["fitted"] else float("nan")
        print(f"  median within-part error: DINO {100 * me:.2f}% of extent vs pipeline {100 * mf:.2f}%")
        print(f"  {'DINO beats the pipeline' if me < mf else 'DINO does NOT beat the pipeline'} on placement error.")

        m_hf = np.median(np.concatenate(err["hks_within_full"])) if acc["hks_within_full"] else float("nan")
        m_hm = np.median(np.concatenate(err["hks_within_matched"])) if acc["hks_within_matched"] else float("nan")
        print(f"\n  ADDENDUM (matched-coverage check, requested before logging any verdict):")
        print(f"    HKS full-coverage (100%) this run: {100 * m_hf:.2f}% median err "
              f"(reproduction check vs original probe's 12.91% reference)")
        print(f"    HKS matched-coverage (same {100 * np.mean(coverage_log):.1f}% subset as DINO): "
              f"{100 * m_hm:.2f}% median err")
        print(f"    DINO on that identical subset: {100 * me:.2f}% median err")
        dino_wins_matched = me < m_hm
        print(f"  -> {'DINO beats HKS' if dino_wins_matched else 'HKS beats or ties DINO'} "
              f"on the IDENTICAL vertex population "
              f"({'the original apparent edge SURVIVES matched-coverage' if dino_wins_matched else 'the original apparent edge was a coverage-population artifact, NOT a real descriptor advantage'})")

        print(f"\n  T1.1's OWN GATE (raw DINO vs HKS, stated before T1.1 was run): within-part "
              f"median error must land clearly below HKS's reference -- evaluated against the "
              f"FAIR, matched-coverage HKS number ({100 * m_hm:.2f}%), not the original "
              f"full-coverage 12.91% reference, since that comparison was shown to be "
              f"population-biased.")
        clears_hks = me < 0.5 * m_hm  # "clearly below" = at most half the matched HKS number
        print(f"  {'refined ' if args.refined else 'raw '}DINO median err {100 * me:.2f}% vs "
              f"matched HKS median err {100 * m_hm:.2f}% -> "
              f"{'CLEARS (< half of matched HKS)' if clears_hks else 'does NOT clear'}")

        RAW_DINO_REFERENCE = 0.1034  # T1.1's own raw-DINO result, job 15520392, this exact protocol
        result.update(dino_within_median_err=float(me), fitted_reference_from_this_run=float(mf),
                       hks_within_full_this_run=float(m_hf), hks_within_matched_this_run=float(m_hm),
                       matched_comparison_valid=True, pass_condition=bool(clears_hks),
                       raw_dino_reference=RAW_DINO_REFERENCE, refined=bool(args.refined))

        if args.refined:
            print(f"\n  REFINEMENT-LAYER GATE (pre-registered in V2_CHANGELOG.md/REPORT_V2.md "
                  f"BEFORE this run, stated before its result was known): refined within-part "
                  f"median error must land AT OR BELOW HALF of raw DINO's reference "
                  f"({100 * RAW_DINO_REFERENCE:.2f}%), i.e. <= {50 * RAW_DINO_REFERENCE:.2f}%.")
            clears_refinement_gate = me <= 0.5 * RAW_DINO_REFERENCE
            print(f"  refined median err {100 * me:.2f}% vs raw-DINO reference "
                  f"{100 * RAW_DINO_REFERENCE:.2f}% (gate: <= {50 * RAW_DINO_REFERENCE:.2f}%) -> "
                  f"{'CLEARS the refinement-layer gate' if clears_refinement_gate else 'does NOT clear the refinement-layer gate'}")
            print(f"  (secondary target, the number that would actually justify T1.2: fitted "
                  f"pipeline reference {100 * mf:.2f}% -- "
                  f"{'also cleared' if me < mf else 'not cleared'})")
            result.update(clears_refinement_gate=bool(clears_refinement_gate))

    import json

    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    out = dict(
        scores={k: dict(correct=float(np.mean(v)) if v else None,
                         median_err=float(np.median(np.concatenate(err[k]))) if v else None)
                for k, v in acc.items()},
        mean_joint_coverage=float(np.mean(coverage_log)) if coverage_log else None,
        n_specimens=args.n,
        gt_groups=args.gt_groups,
        **result,
    )
    out_name = "within_part_signal_dino_refined.json" if args.refined else "within_part_signal_dino.json"
    out_path = os.path.join(HERE, "out", out_name)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
