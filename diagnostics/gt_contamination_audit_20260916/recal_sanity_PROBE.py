"""Are the expert-recalibrated landmark vertices anatomically sensible on the template?

`annotation/landmark_indices_recalibrated.json` was fitted to the 12 expert surface annotations,
whose stored `original` coords are in the broken Blender Z-up frame (see frame_contamination_PROBE).
If the recalibration consumed those raw coords, the chosen template vertices were matched to
ROTATED targets and should sit at anatomically wrong places on the template.

Test: report each landmark's position in the template's own body frame, as a fraction along the
anterior-posterior axis (0 = most anterior, 1 = most posterior) and the dorsoventral / lateral axes.
Compare recalibrated vs the rule-computed template file.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "diagnostics", "morphometrics"))
sys.path.insert(0, os.path.join(REPO, "diagnostics", "groundtruth"))

from measure import load_model, body_frame  # noqa: E402

RECAL = json.load(open(os.path.join(REPO, "annotation/landmark_indices_recalibrated.json")))
TEMPL = json.load(open(os.path.join(REPO, "diagnostics/groundtruth/landmark_template_indices.json")))["landmarks"]

TEMPL_TO_RECAL = {
    "clypeal_margin_ant_mid": "clypeal_ant_mid",
    "cephalic_margin_post_mid": "cephalic_post_mid",
    "mandibular_apex_r": "mandibular_apex_r",
    "antennal_insertion_r": "antennal_insertion_r",
    "scape_apex_r": "scape_apex_r",
    "wl_anterior_r": "wl_anterior_r",
    "wl_posterior_r": "wl_posterior_r",
}

M = load_model()
V = np.asarray(M["v_template"], float)
R = body_frame(M)
print(f"template verts {V.shape}, body_frame -> {type(R)}")
if isinstance(R, tuple):
    axes = np.asarray(R[0], float) if np.asarray(R[0]).shape == (3, 3) else np.asarray(R[1], float)
else:
    axes = np.asarray(R, float)
print("axes shape", axes.shape)

P = (V - V.mean(0)) @ axes.T if axes.shape == (3, 3) else (V - V.mean(0))
lo, hi = P.min(0), P.max(0)
frac = lambda p: (p - lo) / np.maximum(hi - lo, 1e-12)

print()
print(f"{'landmark':<28} {'templ v':>8} {'recal v':>8} | {'templ (ap,dv,lat)':>26} | {'recal (ap,dv,lat)':>26} | {'moved%diag':>10}")
print("-" * 125)
rows = {}
for tname, rname in TEMPL_TO_RECAL.items():
    e = RECAL.get(rname)
    if e is None:
        continue
    rv = int(e["vertex"])
    tv = TEMPL.get(tname, {}).get("vertex")
    ft = frac(P[int(tv)]) if tv is not None else None
    fr = frac(P[rv])
    st = "  ".join(f"{x:5.2f}" for x in ft) if ft is not None else " " * 20
    sr = "  ".join(f"{x:5.2f}" for x in fr)
    print(f"{tname:<28} {str(tv):>8} {rv:>8} | {st:>26} | {sr:>26} | {e.get('moved_pct_diag', float('nan')):>10.1f}")
    rows[tname] = dict(templ_vertex=tv, recal_vertex=rv,
                       templ_frac=None if ft is None else ft.tolist(), recal_frac=fr.tolist(),
                       moved_pct_diag=e.get("moved_pct_diag"))

print()
wa, wp = rows.get("wl_anterior_r"), rows.get("wl_posterior_r")
if wa and wp:
    d_t = np.linalg.norm(V[wa["templ_vertex"]] - V[wp["templ_vertex"]])
    d_r = np.linalg.norm(V[wa["recal_vertex"]] - V[wp["recal_vertex"]])
    diag = np.linalg.norm(V.max(0) - V.min(0))
    print(f"Weber's length on the template:  rule-computed {d_t/diag*100:.1f}% diag | "
          f"recalibrated {d_r/diag*100:.1f}% diag | ratio {d_r/d_t:.3f}")
    print("WL is the denominator of every scale-free M4 trait (HW/WL, HL/WL).")

json.dump(rows, open(os.path.join(HERE, "recal_sanity.json"), "w"), indent=1)
