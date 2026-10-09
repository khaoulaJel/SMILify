"""validate_landmarks.py -- check a placed landmark set before anything downstream uses it.

Every rule here was derived from an error actually made during placement, so this is a regression
test for landmark placement, not a style check. Run it on landmark_template_indices.json:

    python validate_landmarks.py landmark_template_indices.json

FAIL blocks use of the file. WARN is a judgement call to review with the expert.
"""
import sys, os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "diagnostics", "morphometrics"))
from measure import load_model  # noqa: E402  the model must be resolved through the same path
                                # the measurements use -- see landmark_candidates.py's header for
                                # the incident this prevents.

# part membership each landmark must satisfy (dominant skinning joint)
PART_OF = {
    "clypeal_margin_ant_mid": ["b_h"], "cephalic_margin_post_mid": ["b_h"],
    "antennal_insertion_r": ["b_h"], "mandibular_apex_r": ["ma_r"],
    "scape_apex_r": ["an_1_r"], "wl_anterior_r": ["b_t"], "wl_posterior_r": ["b_t"],
    "petiole_ant_mid": ["b_a_1"], "petiole_post_mid": ["b_a_1"],
    "gaster_apex_mid": ["b_a_3", "b_a_4"],
}
MIDLINE = ["clypeal_margin_ant_mid", "cephalic_margin_post_mid",
           "petiole_ant_mid", "petiole_post_mid", "gaster_apex_mid"]
RIGHT = ["mandibular_apex_r", "antennal_insertion_r", "scape_apex_r",
         "wl_anterior_r", "wl_posterior_r"]


def load():
    M = load_model()
    return (M["v_template"], M["dominant"], M["jnames"],
            np.asarray(M["dd"]["J"], dtype=np.float64))


def main(path):
    V, dom, jn, J = load()
    ji = {n: k for k, n in enumerate(jn)}
    d = json.load(open(path))
    LM = {k: v["vertex"] for k, v in d["landmarks"].items()}
    fails, warns, oks = [], [], []

    def P(name):
        return V[LM[name]]

    # ---- 0. completeness -------------------------------------------------
    for n, v in LM.items():
        if v is None or v < 0:
            fails.append(f"{n}: not placed")
        elif not (0 <= v < len(V)):
            fails.append(f"{n}: vertex {v} out of range")
    if fails:
        return report(fails, warns, oks)

    # ---- 1. midline ------------------------------------------------------
    for n in MIDLINE:
        if n not in LM:
            continue
        y = P(n)[1]
        (oks if abs(y) < 1e-4 else fails).append(
            f"{n}: y={y:+.5f}" + ("" if abs(y) < 1e-4 else "  NOT on the midline (needs |y|<1e-4)"))

    # ---- 2. right side ---------------------------------------------------
    for n in RIGHT:
        if n not in LM:
            continue
        y = P(n)[1]
        (oks if y < -0.01 else fails).append(
            f"{n}: y={y:+.5f}" + ("" if y < -0.01 else "  NOT on the right side (needs y<-0.01)"))

    # ---- 3. anatomical part membership -----------------------------------
    for n, parts in PART_OF.items():
        if n not in LM:
            continue
        got = jn[dom[LM[n]]]
        (oks if got in parts else fails).append(
            f"{n}: on '{got}'" + ("" if got in parts else f"  EXPECTED one of {parts}"))

    # ---- 4. same-surface rules (the two errors already made) -------------
    if "wl_anterior_r" in LM and "wl_posterior_r" in LM:
        a, p = P("wl_anterior_r"), P("wl_posterior_r")
        lat = abs(a[1] - p[1]); L = np.linalg.norm(a - p)
        (oks if lat < 0.02 else fails).append(
            f"WL same-side: lateral offset {lat:.4f} (must be <0.02, else WL absorbs mesosoma width)")
        span = np.ptp(V[dom == ji["b_t"], 0])
        (oks if L > 0.6 * span else warns).append(
            f"WL length {L:.4f} vs mesosoma span {span:.4f} ({100*L/span:.0f}%) "
            + ("" if L > 0.6 * span else " -- suspiciously short, an endpoint may not be at an extreme"))

    if "petiole_ant_mid" in LM and "petiole_post_mid" in LM:
        a, p = P("petiole_ant_mid"), P("petiole_post_mid")
        L = np.linalg.norm(a - p); vert = abs(a[2] - p[2])
        frac = vert / max(L, 1e-9)
        (oks if frac < 0.25 else fails).append(
            f"PetL same-surface: {100*frac:.0f}% of the length is vertical "
            f"(must be <25%; endpoints must share the dorsal or ventral outline)")

    # ---- 5. scape is the elbow, not the antenna tip ----------------------
    if "scape_apex_r" in LM:
        art = J[ji["an_1_r"]]
        L = np.linalg.norm(P("scape_apex_r") - art)
        allan = np.where(np.isin(dom, [ji[x] for x in ("an_1_r", "an_2_r", "an_3_r")]))[0]
        tip = np.linalg.norm(V[allan] - art, axis=1).max()
        (oks if L < 0.6 * tip else fails).append(
            f"SL {L:.4f} vs whole-antenna {tip:.4f} ({100*L/tip:.0f}%) "
            + ("" if L < 0.6 * tip else " -- this is the antenna TIP, not the scape elbow"))

    # ---- 6. HL / ML orientation sanity -----------------------------------
    if "clypeal_margin_ant_mid" in LM and "cephalic_margin_post_mid" in LM:
        a, p = P("clypeal_margin_ant_mid"), P("cephalic_margin_post_mid")
        (oks if a[0] > p[0] else fails).append(
            f"HL orientation: clypeal x={a[0]:.3f} vs occipital x={p[0]:.3f}"
            + ("" if a[0] > p[0] else "  REVERSED -- clypeal margin must be anterior"))
    if "mandibular_apex_r" in LM and "clypeal_margin_ant_mid" in LM:
        m = P("mandibular_apex_r")
        (oks if m[0] >= V[dom == ji["b_h"], 0].max() - 1e-6 or m[0] > P("clypeal_margin_ant_mid")[0]
         else warns).append(
            f"ML: mandibular apex x={m[0]:.3f} vs clypeal x={P('clypeal_margin_ant_mid')[0]:.3f} "
            "(apex should project further forward)")

    return report(fails, warns, oks)


def report(fails, warns, oks):
    for line in oks:
        print("  OK    " + line)
    for line in warns:
        print("  WARN  " + line)
    for line in fails:
        print("  FAIL  " + line)
    print()
    print(f"{len(oks)} passed, {len(warns)} warnings, {len(fails)} failures")
    if fails:
        print("\nDO NOT run the extraction layer on this file until the failures are fixed.")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1
                  else os.path.join(HERE, "landmark_template_indices.json")))
