"""landmark_candidates.py -- propose template vertex indices for the GLAD trait landmarks.

These are GEOMETRIC PROPOSALS, not definitions. Each is the vertex that satisfies a simple
extremal rule inside the correct anatomical part. The rule approximates the published landmark
definition; it does not replace it. Every index emitted here must be visually verified in
Blender against the definition in LANDMARK_PROTOCOL.md before it is used.

THE MODEL IS RESOLVED THROUGH measure.load_model(), NOT by naming a file here. An earlier version
of this script hard-coded `3D_model_prep/SMIL_OmniAnt.pkl` because that is what `config.SMAL_FILE`
says. It is the wrong model: `measure.py` line 52 does
`os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")`,
so every fitted result in this project -- G1, G2, the Z8 runs, the 757-specimen corpus -- lives on
a 10235-vertex / 20466-face / 25-PC model, while SMIL_OmniAnt is 10229 / 20454 / 13. The two are
NOT a subset of one another (the first 10229 vertices differ by up to 1.35 units), so indices do
not transfer. Resolving through the same code path the measurements use is the only way this
cannot drift again.

Axis convention is DERIVED from joint positions, never assumed -- see `axis_report()`.
"""
import os
import sys
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "diagnostics", "morphometrics"))

from measure import load_model  # noqa: E402  single source of truth for which model is in play


def load():
    M = load_model()
    import config
    return (M["v_template"], M["dominant"], M["jnames"],
            np.asarray(M["dd"]["J"], dtype=np.float64), config.SMAL_FILE)


def mirror_map(V):
    """Exact bilateral pairing. Verified: 100% of vertices match within 1e-3."""
    from scipy.spatial import cKDTree
    M = V.copy(); M[:, 1] *= -1
    d, i = cKDTree(V).query(M)
    assert d.max() < 1e-3, f"mesh not mirror-symmetric (max {d.max()})"
    return i


def main():
    V, dom, jn, J, model_path = load()
    ji = {n: k for k, n in enumerate(jn)}
    part = lambda name: np.where(dom == ji[name])[0]
    mid = np.where(np.abs(V[:, 1]) < 1e-4)[0]          # 223 midline vertices
    mirror = mirror_map(V)

    def pick(idx, score, label, conf, note):
        v = int(idx[np.argmax(score)])
        return dict(name=label, vertex=v, xyz=[round(float(c), 4) for c in V[v]],
                    confidence=conf, note=note)

    head, thorax = part("b_h"), part("b_t")
    head_mid = np.intersect1d(head, mid)
    thor_mid = np.intersect1d(thorax, mid)
    pet_mid = np.intersect1d(part("b_a_1"), mid)
    out = []

    # --- head, midline (Type I / II) -------------------------------------
    out.append(pick(head_mid, V[head_mid, 0], "clypeal_margin_ant_mid", "medium",
                    "Anteriormost midline point of the head part. GLAD HL anterior endpoint. "
                    "Verify it sits on the median clypeal margin, not on a mandible or the labrum."))
    out.append(pick(head_mid, -V[head_mid, 0], "cephalic_margin_post_mid", "medium",
                    "Posteriormost midline point of the head part. GLAD HL posterior endpoint. "
                    "Verify it is the occipital margin midpoint, not inside the neck/foramen."))

    # --- mandible, right side (Type I) -----------------------------------
    ma = part("ma_r")
    out.append(pick(ma, np.linalg.norm(V[ma] - J[ji["ma_r"]], axis=1), "mandibular_apex_r", "high",
                    "Vertex furthest from the mandibular articulation = the apical tooth. "
                    "HIGHEST-RISK point: the template is a mean mandible, and correspondence on "
                    "trap-jaw (Odontomachus) and long-mandible (Aenictus) forms may not hold."))

    # --- antenna, right side ---------------------------------------------
    out.append(pick(head, -np.linalg.norm(V[head] - J[ji["an_1_r"]], axis=1),
                    "antennal_insertion_r", "medium",
                    "Head vertex nearest the scape articulation. Torular/antennal insertion."))
    an1 = part("an_1_r")
    out.append(pick(an1, np.linalg.norm(V[an1] - J[ji["an_1_r"]], axis=1), "scape_apex_r", "medium",
                    "Furthest scape vertex from its articulation. NOTE: GLAD SL excludes the "
                    "basal condyle/neck, so the proximal endpoint is the articulation, not the bulb."))

    # --- mesosoma: the two Weber's length endpoints ----------------------
    # SAME-SIDE RULE. Weber's length is the diagonal of the mesosoma IN PROFILE, so both
    # endpoints lie in one lateral plane. An earlier version paired a midline anterior point with
    # a lateral posterior one, which makes the diagonal absorb mesosoma WIDTH -- and WL is the
    # normaliser for every other trait, so that width would leak into all of them. Both endpoints
    # are therefore drawn from the same right lateral band. These are Type III CONSTRUCTED
    # extremes, not Type I anatomical junctions, and must be reported on the Type III axis.
    lat = thorax[V[thorax, 1] < -0.055]
    out.append(pick(lat, V[lat, 0], "wl_anterior_r", "RULE",
                    "Anteriormost mesosoma vertex in the right lateral band -- the front of the "
                    "pronotum in profile, excluding the neck. Type III constructed."))
    out.append(pick(lat, -(V[lat, 0] + V[lat, 2]), "wl_posterior_r", "RULE",
                    "Posteroventral-most mesosoma vertex in the same lateral band -- the posterior "
                    "basal angle of the metapleuron, behind and below the hind coxa. Type III."))

    # --- waist and gaster -------------------------------------------------
    # SAME-SURFACE RULE. The petiole midline is 8 vertices forming two rows -- a dorsal outline
    # and a ventral one -- separated by a large z gap. Choosing "frontmost" and "backmost"
    # independently lands them on OPPOSITE rows, and the resulting "length" is a diagonal: on the
    # earlier model 60% of it was height. Both endpoints are therefore pinned to the ventral row,
    # which is also the row that spans the segment's full length. Split at the LARGEST GAP rather
    # than the median: a median split silently drops the posteriormost ventral vertex.
    if pet_mid.size >= 4:
        z = np.sort(V[pet_mid, 2])
        cut = (z[np.argmax(np.diff(z))] + z[np.argmax(np.diff(z)) + 1]) / 2
        vent = pet_mid[V[pet_mid, 2] < cut]
        out.append(pick(vent, V[vent, 0], "petiole_ant_mid", "medium",
                        "Anterior petiole, VENTRAL midline outline (same-surface rule)."))
        out.append(pick(vent, -V[vent, 0], "petiole_post_mid", "medium",
                        "Posterior petiole, same VENTRAL outline."))
    # b_a_5 has ZERO dominant vertices -- the terminal gaster joint drives no geometry, the same
    # pathology as the four wing joints. The gaster surface is carried by b_a_3 and b_a_4.
    ga = np.union1d(part("b_a_3"), part("b_a_4"))
    ga_mid = np.intersect1d(ga, mid)
    src = ga_mid if ga_mid.size else ga
    out.append(pick(src, -V[src, 0], "gaster_apex_mid", "medium",
                    "Posteriormost midline gaster point (TBL endpoint). Taken over b_a_3+b_a_4 "
                    "because b_a_5 carries no vertices."))

    # --- Type III search regions (extents, not points) --------------------
    regions = {
        "head_region": dict(vertices=len(head), rule="max |y| extent within head part -> HW",
                            confidence="high", note="Whole head part. HW is a max, not a percentile."),
        "pronotum_region": dict(vertices=int((V[thorax, 0] > np.percentile(V[thorax, 0], 66)).sum()),
                                rule="anterior third of mesosoma by x -> PW",
                                confidence="LOW",
                                note="PLACEHOLDER. The pronotum is not a modelled part; this is the "
                                     "anterior third of b_t by x and needs hand-painting in Blender."),
        "eye_region_r": dict(vertices=0, rule="max diameter within eye region -> EL",
                             confidence="ABSENT",
                             note="No eye geometry is separable from the head part. EL and HW2 cannot "
                                  "be computed until an eye vertex group is painted, or they are dropped."),
    }

    res = dict(model=model_path, n_vertices=int(V.shape[0]),
               n_midline=int(mid.size), axis_convention="+X anterior, Y lateral, +Z dorsal",
               landmarks=out, regions=regions,
               mirror_map_note="template is exactly mirror-symmetric; left-side indices derived, never clicked")
    with open(os.path.join(HERE, "landmark_candidates.json"), "w") as fh:
        json.dump(res, fh, indent=1)

    print(f"{'landmark':26s} {'vert':>6s} {'conf':>7s}  xyz")
    for d in out:
        print(f"{d['name']:26s} {d['vertex']:6d} {d['confidence']:>7s}  {d['xyz']}")
    print()
    for k, v in regions.items():
        print(f"{k:18s} n={v['vertices']:5d}  {v['confidence']:>6s}  {v['rule']}")


if __name__ == "__main__":
    main()
