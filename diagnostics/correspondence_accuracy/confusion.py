"""Core correspondence-accuracy metric: true anatomical segment x matched anatomical segment,
for one fitted specimen, at leg-level, within-leg-segment-level, and antenna-level granularity.

WHY THIS EXISTS (task framing, 2026-08-19): geometric/Chamfer loss can go down while
correspondence stays wrong -- a smoother loss landscape is not evidence that a target point
ended up matched to the anatomically correct template region. This module is the yardstick:
independent of Chamfer, it answers "which template segment did this scan point actually get
matched to, and was that the segment it truly came from?" for anatomical-constraint work
(GNC and friends) to be checked against, before and after.

CORRESPONDENCE DIRECTION: "matched segment" = the label of the FITTED-mesh vertex nearest a
given true-labeled target point. This is deliberately the same direction
`TargetPartition.update` (`fitter_3d/trainer_hierarchical.py`) and the target->source half of
`_partitioned_chamfer` use -- the metric audits the fitter's own correspondence rule, not a
reimplementation of it. (Source->target chamfer, which pulls the OTHER way, is a different
failure mode -- unmatched/unclaimed template area -- and is not what this module measures.)

GROUND TRUTH comes from area-weighted face sampling on the specimen's OWN target mesh, using
the barycentric scheme `pytorch3d.ops.sample_points_from_meshes` uses (face chosen
proportional to area, uniform barycentric coords within it) -- copied from
`diagnostics/registration_failure/probe_d2a_sample_starvation.py`/`probe_d2b_correspondence_
audit.py`, not reimplemented ad hoc. This is valid ONLY because the synthetic corpora used here
(`diagnostics/moonshot/synth_clean*`) share the template's exact face topology (posed/deformed
template, not an independent scan+remesh) -- callers MUST assert `faces.shape` matches before
trusting these labels; `run_audit.py` does this once per corpus.
"""

import numpy as np
import torch
from pytorch3d.ops import knn_points


def face_areas(verts, faces):
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)


def sample_true_labeled_points(verts, faces, face_lab, n_sample, rng):
    """Area-weighted sample from a target mesh, carrying every face_labels() field per point.

    verts, faces: the TARGET specimen mesh (same topology as the template).
    face_lab: dict of (n_faces,) arrays from labels.face_labels(TEMPLATE_faces, ...) -- valid
    here because target and template share face indexing by construction.
    """
    areas = face_areas(verts, faces)
    p = areas / areas.sum()
    face_idx = rng.choice(len(p), size=n_sample, p=p)
    v0, v1, v2 = faces[face_idx, 0], faces[face_idx, 1], faces[face_idx, 2]
    b = rng.dirichlet([1, 1, 1], size=n_sample)
    pts = b[:, 0:1] * verts[v0] + b[:, 1:2] * verts[v1] + b[:, 2:3] * verts[v2]
    point_lab = {k: v[face_idx] for k, v in face_lab.items()}
    # "true vertex" for geodesic-error purposes: the sampled face's dominant (highest-
    # barycentric-weight) TEMPLATE vertex -- the same reduction face_labels() uses for
    # categorical labels, just kept at vertex granularity instead of collapsed to a segment
    # name, so geodesic.py can measure how far a wrong match actually is in mesh distance.
    dominant_corner = b.argmax(axis=1)  # (n_sample,) in {0,1,2}
    face_verts = faces[face_idx]  # (n_sample, 3)
    true_vertex_idx = face_verts[np.arange(len(face_idx)), dominant_corner]
    point_lab["true_vertex_idx"] = true_vertex_idx
    return pts, point_lab


class ConfusionAccumulator:
    """Accumulates a true-label x matched-label count matrix over many specimens/points."""

    def __init__(self, class_names):
        self.names = list(class_names)
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.M = np.zeros((len(self.names), len(self.names)), dtype=np.int64)
        self.n_unmapped_true = 0
        self.n_unmapped_pred = 0

    def add(self, true_labels, pred_labels):
        for t, p in zip(true_labels, pred_labels):
            ti = self.idx.get(t)
            pi = self.idx.get(p)
            if ti is None:
                self.n_unmapped_true += 1
                continue
            if pi is None:
                self.n_unmapped_pred += 1
                continue
            self.M[ti, pi] += 1

    def as_dict(self):
        row_sum = self.M.sum(axis=1, keepdims=True)
        norm = np.divide(self.M, row_sum, out=np.zeros_like(self.M, dtype=float), where=row_sum > 0)
        return dict(
            names=self.names,
            counts=self.M.tolist(),
            row_normalized=norm.tolist(),
            accuracy=float(np.trace(self.M) / max(self.M.sum(), 1)),
            n_unmapped_true=self.n_unmapped_true,
            n_unmapped_pred=self.n_unmapped_pred,
        )


def nearest_fitted_vertex_index(fitted_verts_t, pts_t, restrict_mask=None):
    """For each point in pts_t (1,P,3), the GLOBAL template-vertex index of the nearest vertex
    in fitted_verts_t (1,V,3), optionally restricted to `restrict_mask` over V. This is the
    same nearest-vertex correspondence rule `TargetPartition`/`_partitioned_chamfer` use;
    `nearest_fitted_label` reduces its output to a segment name, `geodesic.py` uses the raw
    index to measure how far a WRONG match actually is in mesh distance.
    """
    src = fitted_verts_t
    idx_map = None
    if restrict_mask is not None:
        idx_map = np.where(restrict_mask)[0]
        src = fitted_verts_t[:, restrict_mask, :]
        if src.shape[1] == 0:
            return np.array([], dtype=np.int64)
    nn_idx = knn_points(pts_t, src, K=1).idx[0, :, 0].cpu().numpy()
    if idx_map is not None:
        nn_idx = idx_map[nn_idx]
    return nn_idx


def nearest_fitted_label(fitted_verts_t, pts_t, vlabel_key, fitted_vlabels, restrict_mask=None):
    """For each point in pts_t (1,P,3), find the nearest vertex in fitted_verts_t (1,V,3)
    (optionally restricted to `restrict_mask` over V), and return that vertex's label under
    `vlabel_key` (one of the per-vertex arrays returned by labels.vertex_labels).
    """
    if restrict_mask is not None and int(restrict_mask.sum()) == 0:
        return np.array([None] * pts_t.shape[1], dtype=object)
    nn_idx = nearest_fitted_vertex_index(fitted_verts_t, pts_t, restrict_mask)
    return fitted_vlabels[vlabel_key][nn_idx]


def leg_confusion(true_leg, true_is_leg, pts_t, fitted_verts_t, fitted_vlabels):
    """Leg-level confusion: unrestricted nearest-fitted-vertex search over the WHOLE mesh
    (this is what a target point actually experiences under plain chamfer / TargetPartition's
    own knn_points call -- no leg-restriction, since the fitter does not know the true leg).

    Returns (accumulator, leg_pred, pts_leg, true_leg_sub) -- the per-point predictions and the
    leg-restricted points/labels are returned so within_leg_segment_confusion can reuse them
    without repeating the true_is_leg gather or the knn call.
    """
    from labels import LEGS

    acc = ConfusionAccumulator(list(LEGS))
    pts_leg = pts_t[:, true_is_leg, :]
    true_leg_sub = true_leg[true_is_leg]
    if not true_is_leg.any():
        return acc, np.array([], dtype=object), pts_leg, true_leg_sub
    pred = nearest_fitted_label(fitted_verts_t, pts_leg, "leg_id", fitted_vlabels)
    acc.add(true_leg_sub, pred)
    return acc, pred, pts_leg, true_leg_sub


def within_leg_segment_confusion(true_seg, true_is_leg, pts_leg, true_leg_sub, leg_pred, fitted_verts_t, fitted_vlabels):
    """Segment-level confusion (co/tr/fe/ti/ta/pt), RESTRICTED to points whose leg-level match
    (`leg_pred`, from leg_confusion's nearest-fitted-vertex call) was already correct against
    that point's true leg -- isolates within-leg segment swaps (e.g. femur<->tibia) from
    cross-leg swaps, matching the split the D2b probe used, but now the FULL matrix instead of
    a single mismatch-rate number.
    """
    from labels import LEG_SEGMENTS

    acc = ConfusionAccumulator(list(LEG_SEGMENTS))
    if not true_is_leg.any() or len(leg_pred) == 0:
        return acc
    true_seg_sub = true_seg[true_is_leg]
    correct_leg_mask = leg_pred == true_leg_sub
    if not correct_leg_mask.any():
        return acc
    pts_sub = pts_leg[:, correct_leg_mask, :]
    legs_here = true_leg_sub[correct_leg_mask]
    segs_here = true_seg_sub[correct_leg_mask]
    fitted_leg_id = fitted_vlabels["leg_id"]
    preds = np.array([None] * len(legs_here), dtype=object)
    for leg_name in np.unique(legs_here):
        m = legs_here == leg_name
        restrict = fitted_leg_id == leg_name
        pred = nearest_fitted_label(fitted_verts_t, pts_sub[:, m, :], "leg_seg", fitted_vlabels, restrict_mask=restrict)
        preds[m] = pred
    acc.add(segs_here, preds)
    return acc


def within_leg_segment_geodesic(true_vertex_idx, true_is_leg, pts_leg, true_leg_sub, leg_pred, fitted_verts_t, fitted_vlabels):
    """Companion to within_leg_segment_confusion: for the SAME correct-leg-only point subset,
    return (true_vertex_idx_sub, matched_vertex_idx, leg_name_sub) so a caller can look up
    template-mesh geodesic distance between the true and matched vertex -- the categorical
    confusion matrix says WHETHER a point landed in the wrong segment, this says HOW FAR (in
    mesh distance) the miss actually was, which is what tells "still wrong but now
    adjacent-segment wrong" (partial credit) apart from "wrong by the same amount as before".
    """
    if not true_is_leg.any() or len(leg_pred) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=object)
    true_vidx_sub = true_vertex_idx[true_is_leg]
    correct_leg_mask = leg_pred == true_leg_sub
    if not correct_leg_mask.any():
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=object)
    pts_sub = pts_leg[:, correct_leg_mask, :]
    legs_here = true_leg_sub[correct_leg_mask]
    true_vidx_here = true_vidx_sub[correct_leg_mask]
    fitted_leg_id = fitted_vlabels["leg_id"]
    matched_vidx = np.full(len(legs_here), -1, dtype=np.int64)
    for leg_name in np.unique(legs_here):
        m = legs_here == leg_name
        restrict = fitted_leg_id == leg_name
        matched_vidx[m] = nearest_fitted_vertex_index(fitted_verts_t, pts_sub[:, m, :], restrict_mask=restrict)
    return true_vidx_here, matched_vidx, legs_here


def antenna_confusion(true_ant_side, true_ant_seg, true_is_ant, pts_t, fitted_verts_t, fitted_vlabels):
    """Antenna-level confusion, tracked completely separately from legs per the task spec.

    Two matrices: side (r/l cross-confusion, unrestricted nearest-vertex, mirroring
    leg_confusion) and segment (an_1/an_2/an_3, restricted to points whose side matched,
    mirroring within_leg_segment_confusion). Antenna is NOT its own TargetPartition group in
    any GNC run so far (split_anterior was never passed, per
    `fitter_3d.trainer_hierarchical.anatomical_groups`'s own docstring) -- so the fitter's
    actual correspondence step for antenna points is an UNRESTRICTED nearest-vertex search
    exactly like this function performs, not a partition-restricted one. That is precisely why
    this metric matters here: antenna correspondence is currently unconstrained by
    construction, and this is the number that would move if that changed.
    """
    from labels import ANTENNA_SEGMENTS, SIDES

    side_acc = ConfusionAccumulator(list(SIDES))
    seg_acc = ConfusionAccumulator(list(ANTENNA_SEGMENTS))
    if not true_is_ant.any():
        return side_acc, seg_acc
    pts_ant = pts_t[:, true_is_ant, :]
    true_side_sub = true_ant_side[true_is_ant]
    true_seg_sub = true_ant_seg[true_is_ant]
    side_pred = nearest_fitted_label(fitted_verts_t, pts_ant, "ant_side", fitted_vlabels)
    side_acc.add(true_side_sub, side_pred)

    correct_side_mask = side_pred == true_side_sub
    if correct_side_mask.any():
        pts_sub = pts_ant[:, correct_side_mask, :]
        sides_here = true_side_sub[correct_side_mask]
        segs_here = true_seg_sub[correct_side_mask]
        fitted_side = fitted_vlabels["ant_side"]
        preds = np.array([None] * len(sides_here), dtype=object)
        for side in np.unique(sides_here):
            m = sides_here == side
            restrict = fitted_side == side
            pred = nearest_fitted_label(fitted_verts_t, pts_sub[:, m, :], "ant_seg", fitted_vlabels, restrict_mask=restrict)
            preds[m] = pred
        seg_acc.add(segs_here, preds)
    return side_acc, seg_acc
