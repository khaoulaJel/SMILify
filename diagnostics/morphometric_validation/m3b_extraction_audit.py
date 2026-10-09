"""M3b -- real-corpus extraction transfer audit. Checks C1-C5 exactly as pre-registered in
PREREGISTRATION_M3b_extraction_audit.md. C6 (failure attribution) needs renders and is a
separate manual step driven by this script's flagged list.

Panel is FROZEN. This script must not be used to change it. It reports HW/HL/WL only.
"""
import json, os, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [REPO, os.path.join(REPO, "diagnostics/groundtruth"),
                os.path.join(REPO, "diagnostics/morphometrics"),
                os.path.join(REPO, "diagnostics/morphometric_validation")]

import trait_extract as TX                      # noqa: E402
from measure import load_model, kabsch, body_frame  # noqa: E402
from mv_framework import forward_pair, recal_landmarks, TEMPL_TO_RECAL  # noqa: E402

NPZ = os.path.join(REPO, "diagnostics/moonshot/runs/AUDIT40_W0/Stage_3_deform_fine.npz")
OUT = os.path.join(REPO, "diagnostics/morphometric_validation/out")


def head_lateral_halves(verts, M):
    """|max| and |min| lateral projection of head vertices about the head centroid, in the
    template's own part frame. C4: a head that straddles the midline gives a ratio near 1."""
    V0, jn = M["v_template"], M["jnames"]
    idx = np.where(M["dominant"] == jn.index("b_h"))[0]
    ax = body_frame(M)[0]                              # 0 = lateral
    T = V0[idx] - V0[idx].mean(0)
    P = verts[:, idx]
    Pc = P - P.mean(1, keepdims=True)
    hi = np.zeros(len(P)); lo = np.zeros(len(P))
    for i in range(len(P)):
        proj = (Pc[i] @ kabsch(Pc[i], T)) @ ax
        hi[i], lo[i] = proj.max(), proj.min()
    return hi, np.abs(lo)


def ap_order(verts, M, lm):
    """C3: are the four midline landmarks in anterior->posterior order along the specimen's own
    body axis? Axis is per-specimen (gaster tip -> thorax joint), so this tests the FIT, not the
    landmark indexing (which is fixed template vertices)."""
    J = np.einsum("ij,njk->nik", M["Jr"], verts)
    jn = M["jnames"]
    ant = J[:, jn.index("b_t")] - J[:, jn.index("b_a_5")]      # posterior -> anterior
    ant /= np.linalg.norm(ant, axis=1, keepdims=True)
    chain = ["clypeal_margin_ant_mid", "cephalic_margin_post_mid",
             "petiole_ant_mid", "gaster_apex_mid"]
    proj = np.stack([np.einsum("nc,nc->n", verts[:, lm[c][0]], ant) for c in chain], axis=1)
    return chain, proj, np.all(np.diff(proj, axis=1) < 0, axis=1)   # strictly decreasing


def largest_gap_frac(x):
    """C5: largest gap between consecutive sorted values, as a fraction of the range, together
    with how much of the sample the gap separates off."""
    s = np.sort(x); d = np.diff(s); k = int(d.argmax())
    rng = s[-1] - s[0]
    minority = min(k + 1, len(s) - k - 1) / len(s)
    return d[k] / rng if rng > 0 else 0.0, minority


def main():
    os.makedirs(OUT, exist_ok=True)
    M = load_model()
    print(f"[m3b] model {os.environ.get('SMILIFY_SMAL_FILE','<config default>')} "
          f"-> {M['v_template'].shape[0]} verts, {len(M['jnames'])} joints")
    # Same construction as mv_framework.main(): start from the template set (which is what
    # trait_extract addresses landmarks by) and overlay the expert-recalibrated indices where an
    # adjudicated one exists. petiole_*/gaster_apex_mid have no expert annotation and stay on the
    # template index -- exactly as in M3, so this audit measures the SAME traits M3 validated.
    lm_templ = TX.load_landmarks(M)
    lm_raw = recal_landmarks(M)
    lm = dict(lm_templ)
    for tname, rname in TEMPL_TO_RECAL.items():
        if rname in lm_raw:
            lm[tname] = lm_raw[rname]
    obs, _can, labels = forward_pair(NPZ)
    n = len(labels)
    print(f"[m3b] {n} specimens from {NPZ}")

    tr = TX.traits(obs, M=M, lm=lm)
    HW, HL, WL = tr["HW"], tr["HL"], tr["WL"]
    hw_wl, hl_wl = HW / WL, HL / WL

    # ---- C1 completeness
    c1 = np.isfinite(HW) & np.isfinite(HL) & np.isfinite(WL) & (HW > 0) & (HL > 0) & (WL > 0)
    # ---- C2 ratio plausibility
    c2 = (hw_wl > 0.15) & (hw_wl < 2.0) & (hl_wl > 0.15) & (hl_wl < 2.0)
    # ---- C3 AP ordering
    chain, proj, c3 = ap_order(obs, M, lm)
    # ---- C4 head straddles midline
    hi, lo = head_lateral_halves(obs, M)
    ratio = hi / np.maximum(lo, 1e-12)
    c4 = (ratio > 0.5) & (ratio < 2.0)
    # ---- C5 unimodality proxy
    c5 = {}
    for nm, v in (("HW/WL", hw_wl), ("HL/WL", hl_wl)):
        g, mino = largest_gap_frac(v)
        c5[nm] = dict(largest_gap_frac=float(g), minority_frac=float(mino),
                      violates=bool(g >= 0.20 and mino >= 0.10))

    bars = {"C1": (int(c1.sum()), 40), "C2": (int(c2.sum()), 38),
            "C3": (int(c3.sum()), 36), "C4": (int(c4.sum()), 38)}
    verdict = {k: v[0] >= v[1] for k, v in bars.items()}
    verdict["C5"] = not any(d["violates"] for d in c5.values())
    overall = all(verdict.values())

    flagged = sorted({labels[i] for i in range(n)
                      if not (c1[i] and c2[i] and c3[i] and c4[i])})

    rows = []
    for i in range(n):
        rows.append(dict(specimen=labels[i], genus=labels[i].split("_")[0],
                         HW=float(HW[i]), HL=float(HL[i]), WL=float(WL[i]),
                         HW_WL=float(hw_wl[i]), HL_WL=float(hl_wl[i]),
                         head_lat_ratio=float(ratio[i]),
                         ap_proj={c: float(proj[i, j]) for j, c in enumerate(chain)},
                         C1=bool(c1[i]), C2=bool(c2[i]), C3=bool(c3[i]), C4=bool(c4[i])))

    res = dict(npz=NPZ, n=n, preregistration="PREREGISTRATION_M3b_extraction_audit.md",
               bars=bars, verdict=verdict, C5=c5, overall_pass=bool(overall),
               flagged=flagged, per_specimen=rows,
               summary={k: dict(median=float(np.median(v)), q05=float(np.quantile(v, .05)),
                                q95=float(np.quantile(v, .95)))
                        for k, v in (("HW_WL", hw_wl), ("HL_WL", hl_wl),
                                     ("head_lat_ratio", ratio))})
    p = os.path.join(OUT, "m3b_extraction_audit.json")
    json.dump(res, open(p, "w"), indent=1)

    print(f"\n[m3b] C1 completeness    {bars['C1'][0]}/40  (bar 40)   {'PASS' if verdict['C1'] else 'FAIL'}")
    print(f"[m3b] C2 ratio box       {bars['C2'][0]}/40  (bar 38)   {'PASS' if verdict['C2'] else 'FAIL'}")
    print(f"[m3b] C3 AP ordering     {bars['C3'][0]}/40  (bar 36)   {'PASS' if verdict['C3'] else 'FAIL'}")
    print(f"[m3b] C4 head midline    {bars['C4'][0]}/40  (bar 38)   {'PASS' if verdict['C4'] else 'FAIL'}")
    for nm, d in c5.items():
        print(f"[m3b] C5 {nm:8s}      gap={d['largest_gap_frac']:.3f} minority={d['minority_frac']:.2f}"
              f"   {'FAIL' if d['violates'] else 'PASS'}")
    for nm, d in res["summary"].items():
        print(f"[m3b]   {nm:15s} median={d['median']:.3f}  [{d['q05']:.3f}, {d['q95']:.3f}]")
    print(f"\n[m3b] OVERALL: {'PASS' if overall else 'FAIL'}   flagged={len(flagged)} -> {flagged}")
    print(f"[m3b] wrote {p}")


if __name__ == "__main__":
    main()
