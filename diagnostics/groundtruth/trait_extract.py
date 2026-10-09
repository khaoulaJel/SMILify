"""trait_extract.py -- GLAD morphometric traits as versioned functions on a fitted mesh.

WHY THIS EXISTS. `measure.py` computes two things: joint-to-joint BONE LENGTHS (rig rotation
pivots) and PART DIMENSIONS (percentile extents along PCA axes). Neither is a GLAD trait:

  * a bone length is a distance between rotation pivots, which have no anatomical definition --
    the quantity class G2 found unreliable against real ground truth (leg_prox real R = 0.032
    against a synthetic 0.756);
  * `part_dims` head width is `percentile(98) - percentile(2)` along a PCA axis, whereas GLAD HW
    is the MAXIMUM head width across a defined region. The percentile trim discards exactly the
    extremes the trait is defined as, and the PCA axis is not an anatomical direction (the
    template's head width and length differ by 0.9%, so eigenvalue order is not stable).

This module measures what myrmecologists measure, between landmarks placed under
LANDMARK_PROTOCOL.md and validated by validate_landmarks.py.

THE OPERATIONALISATION, pre-registered in LANDMARK_PROTOCOL.md section 1. Every published ant
measurement is defined in a 2D view ("full-face view", "in profile"). There is no view on a mesh,
so each trait is a direct 3D distance with NO projection. Consequence, declared rather than
discovered: our values differ systematically from microscope values, because a projected width is
always <= a 3D width. That is a constant correctable offset, and a reason to expect non-zero bias
even from a perfect fit.

POSE INVARIANCE. Type I traits are point-to-point distances and are therefore pose-invariant for
free. Type III traits are extents along a direction, which is NOT pose-invariant, so the part is
first rigidly aligned onto the template's copy of it and the extent taken along the template's own
anatomical axis -- the same construction `part_dims` uses, reused rather than reimplemented.
"""

import os
import sys
import json
import hashlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "diagnostics", "morphometrics"))

from measure import load_model, body_frame, kabsch  # noqa: E402  reuse, do not reimplement

LANDMARKS = os.path.join(HERE, "landmark_template_indices.json")

# ---------------------------------------------------------------------------
# Trait definitions. Each is (kind, spec, status, note).
#   "pair"  -> 3D distance between two landmarks
#   "joint" -> 3D distance between two rig JOINTS (flagged: rotation pivots, not surface)
#   "ext"   -> Type III extent along a body axis within a part, after rigid alignment
#   "sum"   -> sum of other traits
# ---------------------------------------------------------------------------
TRAITS = {
    "HL":   ("pair", ("clypeal_margin_ant_mid", "cephalic_margin_post_mid"), "ready",
             "Head length. Anterior clypeal margin to occipital margin midpoint."),
    "ML":   ("pair", ("mandibular_apex_r", "clypeal_margin_ant_mid"), "pending_O2",
             "Mandible length. Convention O2 unresolved: apex-to-clypeus assumed."),
    "SL":   ("pair", ("antennal_insertion_r", "scape_apex_r"), "ready",
             "Scape length. Excludes the basal condyle: proximal endpoint is the articulation."),
    "WL":   ("pair", ("wl_anterior_r", "wl_posterior_r"), "type_III",
             "Weber's length. BOTH endpoints are Type III constructed extremes, not Type I "
             "junctions -- report on the Type III axis. Normaliser for every other trait."),
    "PetL": ("pair", ("petiole_ant_mid", "petiole_post_mid"), "pending_O4",
             "Petiole length, ventral outline. O4: petiole only; postpetiole not measured."),
    "GL":   ("pair", ("petiole_post_mid", "gaster_apex_mid"), "ready",
             "Gaster length. Includes the postpetiole where present (subfamily-dependent)."),
    "HW":   ("ext", ("b_h", 0), "pending_O1",
             "Head width. Maximum mediolateral extent of the head. O1 unresolved: this includes "
             "the eyes, since no eye geometry is separable."),
    "FL":   ("joint", ("l_3_fe_r", "l_3_ti_r"), "rig_not_surface",
             "Hind femur length measured JOINT-TO-JOINT. GLAD FL is a surface length; this is a "
             "rotation-pivot distance, the class G2 found unreliable. Flagged in the certificate."),
    "TBL":  ("sum", ("ML", "HL", "WL", "PetL", "GL"), "derived",
             "Total body length."),
}

# Not computable, kept explicit so their absence is a recorded result rather than an omission.
NOT_COMPUTABLE = {
    "EL":  "No eye geometry is separable from the head part (0 vertices).",
    "HW2": "Depends on eye position; see EL. Definition also unverified (O3).",
    "PW":  "The pronotum is not a modelled structure; PART_DEFS treats the mesosoma as one blob.",
}

# Traits that exist on both sides. Left indices are derived by mirror map, never placed by hand.
BILATERAL = ("ML", "SL", "WL", "FL")


def landmark_version(path=LANDMARKS):
    """Hash of the landmark file. Recorded with every output so a trait table can never be
    silently compared against one computed from a different landmark set."""
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:12]


def mirror_map(V):
    """Exact bilateral vertex pairing. The template is mirror-symmetric to 1e-3 on all 10229
    vertices, verified; assert rather than assume, since a future template may not be."""
    from scipy.spatial import cKDTree
    Mv = V.copy()
    Mv[:, 1] *= -1
    d, i = cKDTree(V).query(Mv)
    if d.max() >= 1e-3:
        raise RuntimeError(f"template is not mirror-symmetric (max {d.max():.4g}); "
                           "left-side traits cannot be derived")
    return i


def load_landmarks(M, path=LANDMARKS):
    """Landmark name -> (right_index, left_index). Midline points map to themselves."""
    d = json.load(open(path))
    if d["n_vertices"] != M["v_template"].shape[0]:
        raise RuntimeError(f"landmark file is for a {d['n_vertices']}-vertex model, "
                           f"this model has {M['v_template'].shape[0]}")
    mir = mirror_map(M["v_template"])
    out = {}
    for name, e in d["landmarks"].items():
        r = int(e["vertex"])
        out[name] = (r, int(mir[r]))
    return out


def _ext(verts, M, part_joint, axis_k):
    """Type III: extent along one body axis within a part, after rigid alignment to the template.

    Not pose-invariant unless aligned first -- a posed head measured along a world axis mixes
    width into length. `axis_k` indexes body_frame's rows: 0 lateral, 1 antero-posterior,
    2 dorso-ventral.
    """
    V0 = M["v_template"]
    idx = np.where(M["dominant"] == M["jnames"].index(part_joint))[0]
    ax = body_frame(M)[axis_k]
    T = V0[idx] - V0[idx].mean(0)
    P = verts[:, idx]
    Pc = P - P.mean(1, keepdims=True)
    out = np.zeros(P.shape[0])
    for i in range(P.shape[0]):
        Q = Pc[i] @ kabsch(Pc[i], T)      # specimen part rotated into the template's part frame
        proj = Q @ ax
        out[i] = proj.max() - proj.min()  # MAXIMUM extent -- not a percentile, unlike part_dims
    return out


def traits(verts, M=None, lm=None):
    """verts: (n, 10229, 3). Returns {trait_name: (n,) array}. Left variants suffixed `_l`."""
    M = M or load_model()
    lm = lm or load_landmarks(M)
    verts = np.asarray(verts, dtype=np.float64)
    if verts.ndim == 2:
        verts = verts[None]
    J = np.einsum("ij,njk->nik", M["Jr"], verts)   # joints, for the rig-measured traits
    jn = M["jnames"]
    out = {}

    def pair(a, b, side):
        ia, ib = lm[a][side], lm[b][side]
        return np.linalg.norm(verts[:, ia] - verts[:, ib], axis=1)

    for name, (kind, spec, _status, _note) in TRAITS.items():
        if kind == "pair":
            out[name] = pair(spec[0], spec[1], 0)
            if name in BILATERAL:
                out[name + "_l"] = pair(spec[0], spec[1], 1)
        elif kind == "joint":
            a, b = jn.index(spec[0]), jn.index(spec[1])
            out[name] = np.linalg.norm(J[:, a] - J[:, b], axis=1)
            if name in BILATERAL:
                al, bl = jn.index(spec[0][:-2] + "_l"), jn.index(spec[1][:-2] + "_l")
                out[name + "_l"] = np.linalg.norm(J[:, al] - J[:, bl], axis=1)
        elif kind == "ext":
            out[name] = _ext(verts, M, spec[0], spec[1])
    for name, (kind, spec, _s, _n) in TRAITS.items():
        if kind == "sum":
            out[name] = np.sum([out[t] for t in spec], axis=0)
    return out


def asymmetry(t):
    """Bilateral asymmetry: |R-L| / mean. A lower bound on pipeline noise requiring NO ground
    truth -- ants are near-symmetric, so measured left-right disagreement in a fitted trait is
    error, not biology. This is the cheapest real accuracy signal available."""
    return {n: np.abs(t[n] - t[n + "_l"]) / np.maximum((t[n] + t[n + "_l"]) / 2, 1e-12)
            for n in BILATERAL if n + "_l" in t}


if __name__ == "__main__":
    M = load_model()
    lm = load_landmarks(M)
    print(f"landmark set {landmark_version()}   {len(lm)} landmarks")
    print()
    t = traits(M["v_template"][None], M, lm)
    print("On the TEMPLATE (sanity values, arbitrary units):")
    for n in TRAITS:
        st = TRAITS[n][2]
        flag = "" if st in ("ready", "derived") else f"   [{st}]"
        print(f"  {n:5s} = {t[n][0]:.4f}{flag}")
    print()
    print("  size-corrected by WL (the field's standard normalisation):")
    for n in ("HL", "HW", "ML", "SL", "PetL", "GL"):
        print(f"    {n}/WL = {t[n][0] / t['WL'][0]:.3f}")
    print()
    for n, why in NOT_COMPUTABLE.items():
        print(f"  {n:5s}   NOT COMPUTABLE -- {why}")
