"""Geometry-only, correspondence-free leg-pose initialisation (Task 7, Intervention B).

WHY THIS EXISTS, AND WHY IT IS NOT `--pf_init`/`PartAnchorInit`
-----------------------------------------------------------------
`feature/registration_moonshot`'s `PartAnchorInit` (read via `git show`, never merged into this
branch -- `part_anchor_init.py` does not exist here, and `--pf_init` in `optimise_hierarchical.py`
is currently DEAD CODE that would ImportError if invoked) already tested "correspondence-free
pose initialisation from target geometry" and found it delivered only `edge_logratio +5.1%` and
"nothing else" (`diagnostics/FINAL_REPORT.md` §7.9, `moonshot/REPORT.md` §5.3), inside an arm that
also used a FROZEN, LEARNED, target-derived partition for the entire rest of the fit -- which lost
decisively on its own (chamfer -47.6%, edge_logratio -22.9%, part_leg_distal within_tau -18.5% vs
a byte-identical control). The moonshot report's own reading: "Circularity in the fit-derived
partition is a real flaw AND ALSO an error-correcting mechanism... A frozen assignment cannot
recover, so its errors are permanent." So that experiment never isolated the INIT mechanism from
the (separately, decisively bad) frozen-partition mechanism -- pf_init's +5.1%-and-nothing-else
result is confounded, not a clean test of "does target-informed initialisation help."

Two further, independent limitations of `PartAnchorInit` specifically (not of init-from-geometry
in general):
  1. It requires a TRAINED CLASSIFIER (a PointNet++-style part-field network) to get a
     target-derived partition before any fit exists. That network is exactly the frozen partition
     the moonshot report says is the wrong way to use target-derived information.
  2. Its objective (match per-part centroid + second moment) is BLIND TO ARTICULATION: centroid
     and second moment of a bent leg's points and a straight leg's points, at the same length and
     radius, are very similar. A moment-matching objective has essentially no gradient telling it
     WHERE along the chain a bend happened. Task 6 D5 needed the actual bent GROUND-TRUTH POSE to
     recover leg_distal reliability (0.144->0.604) -- a mechanism that only has power if it can
     represent per-joint bend, which moments alone barely constrain.

This module estimates a per-leg CURVE (coarse joint POSITIONS along the chain), not a moment, by
binning each leg's target points along a radial distance from an ANALYTIC coxa anchor and taking
the per-band centroid -- this carries articulation information a whole-leg moment does not,
because a bend changes which points fall in which band. It converts that curve into joint_rot via
closed-form forward-kinematics inversion ("aim" IK: solve each joint's LOCAL rotation so its
child lands at the estimated position, given the model's own known REST bone lengths -- run once,
not an optimisation loop that could itself get stuck).

LEAKAGE AUDIT -- what this module is and is not allowed to use, checked line by line
--------------------------------------------------------------------------------------
- NO ground-truth vertex/joint labels, NO ground-truth rotations, anywhere in this file.
- NO learned classifier / part-field network. `PartFieldPartition` and `PartFieldNet` are never
  imported here.
- NO fitted correspondence of the kind under test (leg- or segment-level target-point assignment
  via nearest-neighbour against the model's OWN fitted mesh, i.e. `TargetPartition` from
  `trainer_hierarchical.py`). That mechanism IS the thing the split_distal/pf_init experiments
  already showed is unreliable for distal segments (Intervention A: within-leg mismatch 0.8-1.0
  regardless of split_distal) -- using it to seed the initialiser would be circular: bootstrap the
  hard part of the problem from the very estimate that is known not to solve it.
- What IS used, and why each is legitimate:
    1. The TEMPLATE's rest-pose kinematic tree and per-joint rest offsets (`Jr @ v_template`).
       This is a fixed anatomical prior -- the same information `measure.py`'s `J_regressor`
       already uses for every scoring computation in this project, not scan-specific, not GT.
    2. H0's own fitted `global_rot`/`trans` (a single whole-body rigid+translation fit against the
       FULL target point cloud, run identically in baseline and this arm; per Task 6, H0 places
       the body/coxae reliably: 0% cross-leg confusion at pose0). This is the ONE thing pose
       initialisation for the legs is allowed to depend on, because both experimental arms
       (baseline zero-init and this geometric init) run H0 identically before either sees a leg
       joint -- it is a shared precondition, not part of what is being compared.
    3. From (1)+(2), 6 ANALYTIC coxa anchor POSITIONS are computed purely algebraically:
       `coxa_pos_k = trans + R(global_rot) @ (J_rest[coxa_k] - J_rest[root])`. This is arithmetic
       on known constants and H0's two rigid-body parameters, NOT a nearest-neighbour search
       against the scan and NOT `TargetPartition`.
    4. Leg-LEVEL target-point assignment (which of the 6 legs a point belongs to) is done by
       nearest ANALYTIC coxa anchor (3-of-6 argmin over the 6 positions from step 3), not by
       nearest fitted mesh vertex. This is deliberately a weaker, purely geometric substitute for
       `TargetPartition` at this one stage, precisely so the initialiser cannot inherit whatever
       correctness or error the mesh-based partition has.
    5. Within a leg, distance-along-the-chain is approximated by Euclidean radial distance from
       that leg's own analytic coxa anchor (not a mesh geodesic, not a learned feature), binned
       against the TEMPLATE's known rest cumulative bone lengths. This is the one approximation in
       the method that is not exact even in principle (a folded-back leg breaks the monotonic
       radial-distance-implies-chain-position assumption) -- flagged, not hidden, and it is
       validated against ground truth as a diagnostic (never as an input) before any fit uses it.
  Net effect: the initialiser touches the target point cloud only through (a) H0's already-shared
  whole-body rigid fit and (b) fixed anatomical constants. It has no access to, and cannot recover,
  the specific distal correspondence solution the rest of the pipeline is trying to find -- if it
  did, evaluating it against GT (below) would trivially show ~perfect distal joint recovery, which
  it does not (see PROBE_B0 results, reported before any use in a fitting run).

Falls back to the REST direction (net zero local rotation for that segment, i.e. baseline's own
zero-init behaviour) wherever a distance band has too few points to estimate -- so in the worst
case (leg entirely unrepresented, e.g. severe drop60 damage) this degrades toward baseline for
that segment specifically, not to something arbitrarily worse.

FIXED, NOT TUNED, HYPERPARAMETERS -- chosen by geometric argument BEFORE seeing any of the four
test conditions' results, and never adjusted afterward:
  - `min_pts_per_band=4`: matches `part_moments`' own minimum in the (unrelated, moonshot-branch)
    PartAnchorInit code and this project's general practice of requiring >=4 points before trusting
    a 3D centroid/covariance estimate (a 3-point centroid has no redundancy against outliers).
  - band half-width = 0.5x that step's OWN rest segment length: `estimate_leg_curve` is a
    SEQUENTIAL walk (search radius is measured from the CURRENT, already-estimated joint, not a
    fixed anchor -- an earlier cumulative-distance-from-coxa design was tried and rejected before
    any GPU run, because a rest-pose-only smoke test showed even ZERO articulation gives a large
    spurious rotation: ant leg segments are not collinear at rest, so distance-from-coxa is not a
    valid chain-position proxy even without bending). 0.5x gives one full segment-length of slack
    for a genuine bend before a point systematically falls outside the search radius -- a
    structural, not fitted, choice.
"""

import numpy as np
import torch
from pytorch3d.transforms import matrix_to_axis_angle

LEG_SEGMENTS = ("co", "tr", "fe", "ti", "ta", "pt")  # 6 joints per leg chain
MIN_PTS_PER_BAND = 4
BAND_PAD_FRAC = 0.5


def leg_chains(jnames):
    """{'l{k}_{side}': [global joint idx co,tr,fe,ti,ta,pt]} for all 6 legs."""
    chains = {}
    for j, nm in enumerate(jnames):
        if not nm.startswith("l_"):
            continue
        bits = nm.split("_")
        k, seg, side = bits[1], bits[2], bits[-1]
        key = f"l{k}_{side}"
        chains.setdefault(key, [None] * 6)[LEG_SEGMENTS.index(seg)] = j
    for key, chain in chains.items():
        assert None not in chain, f"incomplete leg chain for {key}: {chain}"
    return chains


def rest_joint_positions(Jr, v_template):
    """(J,3) rest-pose joint positions, template units. Jr: (J,V) regressor, v_template: (V,3)."""
    Jr_t = torch.as_tensor(np.asarray(Jr), dtype=torch.float32)
    vt_t = torch.as_tensor(np.asarray(v_template), dtype=torch.float32)
    return Jr_t @ vt_t


def axis_angle_to_matrix_np_safe(aa):
    """(3,) axis-angle -> (3,3) rotation matrix, torch, batched-friendly single vector."""
    from pytorch3d.transforms import axis_angle_to_matrix

    return axis_angle_to_matrix(aa.unsqueeze(0))[0]


def analytic_coxa_anchors(global_rot_aa, trans, rest_J, chains, root_idx=0):
    """6 coxa POSITIONS from H0's fitted rigid params + template rest geometry ONLY.

    No target point cloud, no mesh, no nearest-neighbour search -- pure arithmetic. See the
    module docstring's LEAKAGE AUDIT item 3.

    Matches `smal_model.batch_lbs.batch_global_rigid_transformation` EXACTLY: the root joint's
    own transform (`A0`) is `[R_root | Js[:,0]]`, i.e. global rotation pivots the skeleton about
    the root's OWN rest position (not the coordinate origin), and every descendant's absolute
    position is `parent_pivot + R_root @ (child_rest - parent_rest)` for a direct child of the
    root (no intermediate joints between root and coxa). `trans` is then added on top, exactly as
    `SMAL3DFitter.forward` does (`verts = verts + trans`, after the whole FK pass).

    global_rot_aa: (3,) axis-angle. trans: (3,). rest_J: (J,3). chains: leg_chains() output.
    Returns {leg_key: (3,) coxa position}, and the (3,3) global rotation matrix (reused by the
    per-leg chain solver so it is computed once).
    """
    R_root = axis_angle_to_matrix_np_safe(global_rot_aa)
    root_rest = rest_J[root_idx].to(trans.device)
    out = {}
    for key, chain in chains.items():
        coxa_idx = chain[0]
        offset = rest_J[coxa_idx].to(trans.device) - root_rest
        out[key] = trans + root_rest + R_root @ offset
    return out, R_root


def assign_points_to_legs_coxa(target_pts, coxa_anchors):
    """Nearest-of-6-ANALYTIC-ANCHOR assignment. NOT `TargetPartition` -- see LEAKAGE AUDIT item 4.

    target_pts: (P,3). coxa_anchors: {leg_key: (3,)}.
    Returns {leg_key: (P_k,3) subset of target_pts nearest that leg's coxa anchor}.

    KNOWN DEFECT (2026-08-20, found validating a raw-scan anatomical-labeling adapter):
    representing an entire articulated limb by a single coxa point systematically starves the
    middle (mesothoracic) legs. Validated on synth_clean against ground-truth skinning-weight
    labels (which are near-balanced across all 6 legs, 670-923 verts each): this function
    assigns l2_r/l2_l only 490/493 points on the same specimen where l1/l3 get 1468-2897 -- a
    ~4-6x starvation, not scan noise, reproduced from a single specimen with known correspondence.
    Mechanism: a coxa is a single point; a middle leg's extended surface can curve geometrically
    closer to a NEIGHBORING leg's coxa point than to its own under pure nearest-anchor-point
    competition. Kept here, UNCHANGED, for historical auditability (this is what
    `cheap_anatomical_init.py`/Arm B of the Anatomical Initialization Ceiling Test actually used
    -- that result's mechanistic attribution to "the IK heuristic" alone should be treated as
    unconfirmed; the point-to-leg assignment stage may also be contributing). New code should
    use `assign_points_to_legs_chain` instead -- see its docstring and
    `diagnostics/anatomical_pose_init/validate_leg_assignment.py` for the head-to-head comparison.
    """
    keys = list(coxa_anchors.keys())
    anchors = torch.stack([coxa_anchors[k] for k in keys])  # (6,3)
    d2 = (target_pts[:, None, :] - anchors[None, :, :]).pow(2).sum(-1)  # (P,6)
    nearest = d2.argmin(-1)  # (P,)
    return {k: target_pts[nearest == i] for i, k in enumerate(keys)}


# Backward-compat alias: every existing caller (cheap_anatomical_init.py,
# simple_leg_heuristic.py) keeps getting EXACTLY the coxa-only behavior it always got, so
# fixing the primitive does not silently change historical results. New code should call
# assign_points_to_legs_coxa or assign_points_to_legs_chain explicitly.
assign_points_to_legs = assign_points_to_legs_coxa


def rest_chain_points(rest_J, chains, global_rot_aa, trans, root_idx=0, n_samples_per_segment=8):
    """Per-leg REST-POSE joint chain (co,tr,fe,ti,ta,pt), rigidly transformed by the SAME
    global (rotation, translation) `analytic_coxa_anchors` uses -- i.e. straight/unbent,
    exactly as much articulation information as is available before any per-joint pose is
    estimated (that's what we're trying to compute). Additionally densifies each of the 5
    inter-joint segments with `n_samples_per_segment` linearly interpolated points, so the
    "chain" used for point-to-limb distance is a piecewise-linear polyline (a "distance to
    nearest bone segment" -- the standard primitive in skeletal-rigging point assignment, e.g.
    RigNet's volumetric-distance-to-nearest-bone features), not just 6 isolated joint points.

    Returns {leg_key: (6 + 5*n_samples_per_segment, 3)} tensor of polyline points per leg,
    in the SAME rigidly-transformed frame as `target_pts`/`coxa_anchors`.
    """
    R_root = axis_angle_to_matrix_np_safe(global_rot_aa)
    root_rest = rest_J[root_idx].to(trans.device)

    out = {}
    for key, chain in chains.items():
        joint_pos = []
        for j in chain:
            offset = rest_J[j].to(trans.device) - root_rest
            joint_pos.append(trans + root_rest + R_root @ offset)
        joint_pos = torch.stack(joint_pos)  # (6,3)
        pts = [joint_pos]
        for i in range(len(joint_pos) - 1):
            t = torch.linspace(0, 1, n_samples_per_segment + 2, device=joint_pos.device)[1:-1]
            seg = joint_pos[i][None, :] * (1 - t[:, None]) + joint_pos[i + 1][None, :] * t[:, None]
            pts.append(seg)
        out[key] = torch.cat(pts, dim=0)
    return out


def assign_points_to_legs_chain(target_pts, chain_polylines):
    """Nearest-of-6-ARTICULATED-CHAIN assignment: for each target point, distance to a leg is
    the minimum distance to ANY point on that leg's rest-pose polyline (`rest_chain_points`),
    not just the coxa. Fixes the mesothoracic-starvation defect in
    `assign_points_to_legs_coxa` -- see that function's docstring. Standard "distance to
    nearest bone segment" skeletal-rigging primitive (Kavan et al.-style skinning-by-distance;
    RigNet uses the same volumetric-distance-to-nearest-bone idea for skinning-weight
    prediction: https://ar5iv.labs.arxiv.org/html/2210.09463).

    target_pts: (P,3). chain_polylines: {leg_key: (M,3)} from `rest_chain_points`.
    Returns {leg_key: (P_k,3) subset of target_pts nearest that leg's polyline}.
    """
    keys = list(chain_polylines.keys())
    d2_per_leg = []
    for k in keys:
        poly = chain_polylines[k]  # (M,3)
        d2 = (target_pts[:, None, :] - poly[None, :, :]).pow(2).sum(-1)  # (P,M)
        d2_per_leg.append(d2.min(dim=1).values)  # (P,)
    d2_stack = torch.stack(d2_per_leg, dim=1)  # (P,6)
    nearest = d2_stack.argmin(-1)
    return {k: target_pts[nearest == i] for i, k in enumerate(keys)}


# Body-core rig chain: head -> thorax(root) -> 5 gaster segments. Same JOINT_NAMES convention
# as fitter_3d/part_groups.py.
BODY_CORE_JOINT_NAMES = ["b_h", "b_t", "b_a_1", "b_a_2", "b_a_3", "b_a_4", "b_a_5"]


def body_core_chain_points(rest_J, jnames, global_rot_aa, trans, n_samples_per_segment=8):
    """Body-core polyline: head -> thorax(root) -> gaster segments, rigidly transformed by the
    SAME global (rotation, translation) leg chains use (`rest_chain_points`). A single point
    (root/thorax) badly under-represents an elongated body -- validated on synth_clean
    (2026-08-20, `diagnostics/anatomical_pose_init/validate_7way_soft.py`): single-point
    body-core recall was 0.366 (most true body-core points, e.g. gaster tip, are far from the
    root and get stolen by leg chains); the chain fixes this to 0.663 recall, 7-way macro-F1
    0.608->0.697 -- the same "chain beats point" mechanism `assign_points_to_legs_chain` fixed
    for legs, now applied to the body axis.

    rest_J: (J,3) template rest joint positions (row 0 = root). jnames: full J_names list (root
    included). Returns (6 + 5*n_samples_per_segment, 3) polyline points, in the SAME
    rigidly-transformed frame as `rest_chain_points`/leg assignment.
    """
    name_to_idx = {n: i for i, n in enumerate(jnames)}
    R_root = axis_angle_to_matrix_np_safe(global_rot_aa)
    root_rest = rest_J[0]
    positions = []
    for nm in BODY_CORE_JOINT_NAMES:
        idx = name_to_idx[nm]
        rest_pos = rest_J[idx]
        pos = trans + root_rest + R_root @ (rest_pos - root_rest)
        positions.append(pos)
    joint_pos = torch.stack(positions)
    pts = [joint_pos]
    for i in range(len(joint_pos) - 1):
        t = torch.linspace(0, 1, n_samples_per_segment + 2, device=joint_pos.device)[1:-1]
        seg = joint_pos[i][None, :] * (1 - t[:, None]) + joint_pos[i + 1][None, :] * t[:, None]
        pts.append(seg)
    return torch.cat(pts, dim=0)


def assign_points_to_parts_chain(target_pts, chain_polylines, core_polyline):
    """7-way (body-core + 6 legs) nearest-chain assignment, unifying `assign_points_to_legs_chain`
    with `body_core_chain_points` rather than a separate arbitrary body-core distance threshold
    (validated on synth_clean: no single global threshold cleanly separates the two classes --
    see validate_leg_assignment_v2.py's body-core threshold sweep).

    target_pts: (P,3). chain_polylines: {leg_key: (M,3)}. core_polyline: (M',3).
    Returns (labels (P,) int in {0..6}, 0=body_core, 1..6=leg index in chain_polylines' key
    order), matching the convention of `train_leg_pose_regressor.body_core_and_leg_masks`.
    """
    leg_keys = list(chain_polylines.keys())
    d2_leg = torch.stack([
        (target_pts[:, None, :] - chain_polylines[k][None, :, :]).pow(2).sum(-1).min(dim=1).values
        for k in leg_keys
    ], dim=1)  # (P,6)
    d2_core = (target_pts[:, None, :] - core_polyline[None, :, :]).pow(2).sum(-1).min(dim=1).values[:, None]  # (P,1)
    d2_all = torch.cat([d2_core, d2_leg], dim=1)  # (P,7)
    return d2_all.argmin(dim=1), leg_keys


def _align_rotation(a, b, eps=1e-8):
    """Minimal (no-twist) rotation matrix R with R @ a_hat = b_hat, batched.

    a, b: (...,3), need not be normalised. Standard Rodrigues shortest-arc formula. Degenerates
    (returns identity) when a_hat and b_hat are exactly parallel or antiparallel -- a
    measure-zero event for continuous-valued target points; left unhandled by design rather than
    with an arbitrary perpendicular-axis tiebreak, since any such tiebreak would itself be an
    unjustified extra assumption.
    """
    a_hat = a / a.norm(dim=-1, keepdim=True).clamp_min(eps)
    b_hat = b / b.norm(dim=-1, keepdim=True).clamp_min(eps)
    v = torch.cross(a_hat, b_hat, dim=-1)
    c = (a_hat * b_hat).sum(-1)  # cos(theta)
    s = v.norm(dim=-1)  # sin(theta)
    K = torch.zeros(*v.shape[:-1], 3, 3, dtype=a.dtype, device=a.device)
    K[..., 0, 1] = -v[..., 2]
    K[..., 0, 2] = v[..., 1]
    K[..., 1, 0] = v[..., 2]
    K[..., 1, 2] = -v[..., 0]
    K[..., 2, 0] = -v[..., 1]
    K[..., 2, 1] = v[..., 0]
    eye = torch.eye(3, dtype=a.dtype, device=a.device).expand_as(K)
    coef = torch.where(s > eps, (1 - c) / s.clamp_min(eps) ** 2, torch.zeros_like(s))
    return eye + K + coef[..., None, None] * (K @ K)


KNN_K = 6  # fixed, standard Isomap/geodesic-graph default; not tuned to any result


def _graph_distance_from_anchor(anchor, pts, k=KNN_K):
    """Geodesic-like distance from `anchor` to every point in `pts`, via shortest path over a
    symmetric k-NN graph (Euclidean edge weights) that includes `anchor` as an extra node.

    WHY: straight-line (Euclidean) distance from a single fixed point is NOT a valid proxy for
    "distance along the chain" once the chain bends -- confirmed directly by this module's own
    rest-pose smoke test, which failed with plain radial distance even at ZERO articulation
    (ant leg segments are not collinear at rest). Graph distance instead follows the point
    cloud's own local connectivity, which does track the true chain even through a bend, as
    long as consecutive segments are locally denser than the gap between non-adjacent ones --
    the same assumption every Isomap/geodesic-embedding method makes, standard and unrelated to
    any ground truth or learned model. `k=6` is scipy's own common default context for k-NN
    graphs of this size and is fixed before any of the four test conditions were examined.

    anchor: (3,). pts: (P,3). Returns (P,) graph distance (np.inf where disconnected).
    """
    import numpy as _np
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    P = pts.shape[0]
    all_pts = torch.cat([anchor[None, :], pts], dim=0).detach().cpu().numpy().astype(_np.float64)  # (P+1,3)
    n = all_pts.shape[0]
    kk = min(k, n - 1)
    if kk < 1:
        return torch.full((P,), float("inf"), dtype=pts.dtype, device=pts.device)
    d = _np.linalg.norm(all_pts[:, None, :] - all_pts[None, :, :], axis=-1)  # (n,n)
    nn_idx = _np.argsort(d, axis=1)[:, 1 : kk + 1]  # exclude self
    rows = _np.repeat(_np.arange(n), kk)
    cols = nn_idx.reshape(-1)
    weights = d[rows, cols]
    graph = csr_matrix((weights, (rows, cols)), shape=(n, n))
    graph = graph.maximum(graph.T)  # symmetrize (mutual or one-directional kNN both become edges)
    dist = dijkstra(graph, directed=False, indices=0)  # (n,), from node 0 = anchor
    return torch.as_tensor(dist[1:], dtype=pts.dtype, device=pts.device)  # drop the anchor itself


def estimate_leg_curve(coxa_pos, leg_pts, seg_len, min_pts_per_band=MIN_PTS_PER_BAND):
    """Estimate 5 joint positions (tr,fe,ti,ta,pt) for one leg from its target points.

    Bins `leg_pts` by GRAPH (geodesic-like) distance from `coxa_pos` -- see
    `_graph_distance_from_anchor` -- against the template's known CUMULATIVE rest bone length,
    and takes each band's centroid.

    Two earlier designs were tried and rejected before any GPU run, using this module's own
    rest-pose self-consistency check (a target built FROM the rest pose, zero articulation,
    must recover ~zero rotation -- a structural correctness check, independent of any of the
    four real test conditions):
      1. Cumulative EUCLIDEAN radial distance from the coxa. Failed the rest-pose check: ant leg
         segments are not collinear even at rest (coxa/trochanter/femur already bend), so
         straight-line distance from one fixed point is not a valid "distance along the chain"
         proxy even with zero articulation.
      2. A sequential walk re-searching from each newly estimated joint, still by Euclidean
         distance. Also failed: with all of a leg's points pooled together (not pre-split by
         segment), points from a LATER segment can be Euclidean-closer to an EARLIER joint than
         the correct band radius, once the chain bends, contaminating that band's centroid.
    Graph distance fixes both: it follows the point cloud's own local connectivity rather than
    straight-line distance, so it tracks the true chain through a bend as long as consecutive
    segments are locally denser than the gap to non-adjacent ones (the same assumption every
    Isomap-style geodesic estimate makes).

    coxa_pos: (3,) analytic coxa anchor (target/world frame).
    leg_pts: (P,3) target points assigned to this leg by `assign_points_to_legs`.
    seg_len: (5,) PER-SEGMENT rest bone length co-tr, tr-fe, fe-ti, ti-ta, ta-pt (cumulative sum
    taken internally against graph distance, which -- unlike Euclidean distance -- IS a valid
    cumulative-arc-length quantity).
    Returns (est (5,3), valid (5,) bool) -- invalid entries are NOT filled in (caller falls back
    to the rest direction for those segments, see module docstring).
    """
    est = torch.zeros(5, 3, dtype=leg_pts.dtype, device=leg_pts.device)
    valid = torch.zeros(5, dtype=torch.bool, device=leg_pts.device)
    if leg_pts.shape[0] < min_pts_per_band:
        return est, valid
    d = _graph_distance_from_anchor(coxa_pos, leg_pts)
    finite = torch.isfinite(d)
    cumlen = torch.cumsum(seg_len, dim=0)
    edges = torch.cat([torch.zeros(1, dtype=seg_len.dtype, device=seg_len.device), cumlen])
    for i in range(5):
        lo, hi = edges[i], edges[i + 1]
        pad = BAND_PAD_FRAC * (hi - lo).clamp_min(1e-6)
        m = finite & (d >= lo - pad) & (d <= hi + pad)
        if int(m.sum()) >= min_pts_per_band:
            est[i] = leg_pts[m].mean(0)
            valid[i] = True
        # else: leave valid[i]=False -- caller's IK solve falls back to the rest direction
    return est, valid


def solve_chain_rotations(global_rot_mat, coxa_pos, rest_J, chain_idx, est_curve, valid):
    """Chain "aim" IK: local axis-angle rotations for [co,tr,fe,ti,ta] placing the estimated
    curve, given the already-known (H0-fitted) global_rot and analytic coxa position. Falls back
    to a ZERO local rotation (rest direction preserved) for any segment `valid` marks unusable.

    rest_J: (55,3) rest joint positions (template units, i.e. the SAME frame `Jr @ verts` uses).
    chain_idx: [co,tr,fe,ti,ta,pt] global joint indices for this leg.
    est_curve: (5,3) estimated positions for tr,fe,ti,ta,pt (target/world frame).
    Returns axis_angle (5,3) for joints co,tr,fe,ti,ta (row order matches chain_idx[:5]).
    """
    device = coxa_pos.device
    rest_dirs = torch.stack(
        [rest_J[chain_idx[i + 1]] - rest_J[chain_idx[i]] for i in range(5)]
    ).to(device)  # (5,3), co->tr, tr->fe, fe->ti, ti->ta, ta->pt

    R_global_prev = global_rot_mat  # R_global(coxa)'s PARENT is the body root
    joint_pos_prev = coxa_pos
    out_local = torch.zeros(5, 3, dtype=coxa_pos.dtype, device=device)
    for i in range(5):
        if valid[i]:
            desired_dir = est_curve[i] - joint_pos_prev
            R_global_i = _align_rotation(rest_dirs[i : i + 1], desired_dir[None, :])[0]
            R_local_i = R_global_prev.transpose(-1, -2) @ R_global_i
            joint_pos_next = est_curve[i]
        else:
            R_local_i = torch.eye(3, dtype=coxa_pos.dtype, device=device)
            R_global_i = R_global_prev
            joint_pos_next = joint_pos_prev + R_global_prev @ rest_dirs[i]
        out_local[i] = matrix_to_axis_angle(R_local_i.unsqueeze(0))[0]
        R_global_prev = R_global_i
        joint_pos_prev = joint_pos_next
    return out_local


def init_joint_rot_for_specimen(global_rot_aa, trans, rest_J, jnames, target_pts):
    """End-to-end: (N_POSE,3) joint_rot init for ALL joints, geometric legs + zero elsewhere.

    target_pts: (P,3) this specimen's target point sample, IN THE SAME FRAME as `trans`/rest_J
    (i.e. after H0, before H1 -- no ground truth, no learned model, see LEAKAGE AUDIT).
    """
    n_pose = len(jnames) - 1
    out = torch.zeros(n_pose, 3, dtype=trans.dtype, device=trans.device)
    chains = leg_chains(jnames)
    anchors, R_root = analytic_coxa_anchors(global_rot_aa, trans, rest_J, chains)
    assigned = assign_points_to_legs(target_pts, anchors)
    for key, chain_idx in chains.items():
        seg_len = torch.stack(
            [(rest_J[chain_idx[i + 1]] - rest_J[chain_idx[i]]).norm() for i in range(5)]
        ).to(trans.device)
        est, valid = estimate_leg_curve(anchors[key], assigned[key], seg_len)
        local_aa = solve_chain_rotations(R_root, anchors[key], rest_J, chain_idx, est, valid)
        for i in range(5):  # co,tr,fe,ti,ta -- pretarsus (chain_idx[5]) keeps zero, see docstring
            row = chain_idx[i] - 1  # joint_rot row = global joint idx - 1 (root has no row)
            out[row] = local_aa[i]
    return out


# ============================================================================================
# Genuine constrained IK (2026-08-25 basin-map follow-up, per literature sweep item Kim et al.
# 2015 "Tracking the joints of arthropod legs using multiple images and inverse kinematics").
#
# WHY THIS IS A DIFFERENT METHOD FROM `solve_chain_rotations` ABOVE, NOT A RENAME OF IT:
# `solve_chain_rotations` is a GREEDY, ONE-PASS "aim" heuristic: it walks root-to-tip, and at
# each segment rotates that segment's REST direction to point at an already-estimated
# intermediate curve position, then commits and moves on -- there is no mechanism to revisit an
# earlier joint once a later one reveals it was wrong, and no consultation of this model's
# authored per-axis joint limits (`fitter_3d/joint_limits.py`) at all. This was tested
# end-to-end, under the name "cheap anatomical" (Arm B, `out_ceiling_20260820/RESULTS.md`) and
# found a genuine FAILURE: mean pose error 28.52 deg (worse than zero-init's 23.09 deg), leg_acc
# 0.732 (vs zero-init 0.857) -- numbers the project's own D/E/F follow-up later recognised as
# matching the signature of a PROXIMAL-concentrated error (E: leg_acc 0.768), i.e. consistent
# with "an early greedy commitment at the coxa/root propagating downstream," the single most
# damaging error pattern this whole investigation has found. `init_joint_rot_for_specimen`
# above was never actually run through this failing test (grep confirms it has zero callers
# anywhere in the repo outside this file) -- it is a more carefully leakage-audited rewrite that
# was built but never taken through an actual D1 fit.
#
# `solve_chain_ik` below is a real end-effector-reaching inverse-kinematics solve: given ONLY
# the leg's root (coxa, already reliably known from H0's shared rigid fit) and tip position
# (estimated once, not five separate intermediate curve points), it optimises ALL FIVE joints'
# rotations JOINTLY against a single objective -- reach the tip -- so an error at any one joint
# can be compensated by the others, subject to this model's ACTUAL authored per-axis joint
# limits (`fitter_3d/joint_limits.py`, the same tensors `--limit` already uses in the real
# fitter, not invented bounds), with a minimum-norm (rest-pose-seeking) regulariser to pick a
# sensible point among the redundant solutions (5 joints x 3 axes = 15 unknowns, only 3 position
# constraints from the tip). This directly targets the two most plausible causes of Arm B's
# failure: no joint-limit awareness, and no ability to correct an early mistake using later
# degrees of freedom.
# ============================================================================================


def solve_chain_ik(
    global_rot_mat,
    coxa_pos,
    rest_J,
    chain_idx,
    target_tip,
    min_limits,
    max_limits,
    n_iters=300,
    lr=0.08,
    w_limit=1.0,
    w_reg=0.02,
    theta_init=None,
):
    """Joint IK for one leg: local axis-angle rotations [co,tr,fe,ti,ta] such that forward
    kinematics places the chain's end (pretarsus/tip, chain_idx[5]) at `target_tip`, subject to
    this model's authored per-axis joint limits.

    global_rot_mat: (3,3), coxa_pos: (3,) -- both from `analytic_coxa_anchors`, IDENTICAL
    preconditions to `solve_chain_rotations`, so this is a fair substitution of the rotation-
    solving step alone.
    rest_J: (55,3) rest joint positions, same frame as `Jr @ verts`.
    chain_idx: [co,tr,fe,ti,ta,pt] global joint indices for this leg.
    target_tip: (3,) target/world-frame position for the chain's end (chain_idx[5]).
    min_limits, max_limits: (5,3) axis-angle box constraints for THIS leg's five joints
    (co,tr,fe,ti,ta), sliced from `joint_limits.joint_limit_tensors`'s full (N_POSE,3) output by
    the caller -- see `init_joint_rot_for_specimen_ik`.
    n_iters/lr: fixed, not tuned to any result -- 300 Adam steps on a smooth 15-D least-squares
    objective converges the position residual to numerical noise well before that, verified by
    the oracle self-consistency check this module ships alongside (see
    `diagnostics/anatomical_pose_init/ik_init_probe.py`).
    w_limit/w_reg: joint-limit hinge weight (same functional form as the real fitter's own
    `--limit`, via `fitter_3d.joint_limits.limit_hinge`) and a minimum-norm regulariser toward
    rest pose (theta=0), breaking ties among the redundant solutions in the direction the rest
    of this project's evaluation convention already treats as "no information, no rotation."

    Returns (5,3) axis-angle, one row per co,tr,fe,ti,ta (same row convention as
    `solve_chain_rotations`, drop-in compatible with its caller).
    """
    from pytorch3d.transforms import axis_angle_to_matrix

    from fitter_3d.joint_limits import limit_hinge

    device = coxa_pos.device
    rest_dirs = torch.stack(
        [rest_J[chain_idx[i + 1]] - rest_J[chain_idx[i]] for i in range(5)]
    ).to(device)  # (5,3), co->tr, tr->fe, fe->ti, ti->ta, ta->pt -- SAME convention as
    # `solve_chain_rotations`: already scaled to segment length, not unit vectors.

    if theta_init is None:
        theta = torch.zeros(5, 3, device=device, requires_grad=True)
    else:
        theta = theta_init.detach().clone().to(device).requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    target_tip = target_tip.detach().to(device)

    for _ in range(n_iters):
        opt.zero_grad()
        R_local = axis_angle_to_matrix(theta)  # (5,3,3)
        R_g = global_rot_mat
        pos = coxa_pos
        for i in range(5):
            R_g = R_g @ R_local[i]
            pos = pos + R_g @ rest_dirs[i]
        pos_loss = ((pos - target_tip) ** 2).sum()
        limit_loss = limit_hinge(theta, min_limits, max_limits)
        reg_loss = (theta**2).mean()
        loss = pos_loss + w_limit * limit_loss + w_reg * reg_loss
        loss.backward()
        opt.step()

    return theta.detach()


def estimate_leg_tip(coxa_pos, leg_pts, min_pts=MIN_PTS_PER_BAND):
    """Single tip-position estimate for one leg: the assigned point with the LARGEST graph
    (geodesic-like) distance from the coxa anchor -- the most-distal point this leg's own
    assigned points actually contain, not a fixed-band centroid.

    Deliberately requires only ONE reliable fact (which assigned point is furthest along the
    chain) rather than `estimate_leg_curve`'s five separately-banded centroids -- the distal
    segments this project's own `joint_limits.py` notes hold ~2.4% of a leg's surface area are
    exactly where per-band centroid estimation is most starved of points; a single "furthest
    point" estimate only needs there to BE a most-distal point among however many the leg has,
    not four-plus points in each of five separate bands.

    Returns (tip (3,), valid bool). valid=False (tip left at coxa_pos) if too few points --
    caller must handle this the same way `solve_chain_rotations`'s `valid=False` path does.
    """
    if leg_pts.shape[0] < min_pts:
        return coxa_pos.clone(), False
    d = _graph_distance_from_anchor(coxa_pos, leg_pts)
    finite = torch.isfinite(d)
    if int(finite.sum()) < min_pts:
        return coxa_pos.clone(), False
    idx = torch.argmax(torch.where(finite, d, torch.full_like(d, -1.0)))
    return leg_pts[idx], True


def estimate_leg_waypoint(coxa_pos, leg_pts, target_cumlen, min_pts=MIN_PTS_PER_BAND):
    """One intermediate point estimate: the assigned point whose graph distance from the coxa is
    CLOSEST to `target_cumlen` (a rest-pose cumulative bone length up to some joint along the
    chain). Companion to `estimate_leg_tip` -- together they give a 2-constraint IK problem
    (root implicit, one waypoint, one tip) instead of 1 (tip only) or 5 (`estimate_leg_curve`'s
    full banded curve). Deliberately the weakest possible second constraint: needs only the
    SINGLE nearest point to exist, not >=4 points in a band, so it degrades as gracefully as
    `estimate_leg_tip` does under sparse coverage.

    Returns (point (3,), valid bool). valid=False if the leg has too few points at all to trust
    ANY graph-distance-based statistic (same min_pts gate as `estimate_leg_tip`, for consistency
    -- this function does not additionally require the nearest point to be close to
    `target_cumlen`, since "no leg point that ATTEMPTS to be a mid-chain point but is a somewhat
    poor match" is still informative for IK's overdetermination; if the assignment is bad enough
    that this is actively misleading, that is a genuine failure mode this arm should not hide).
    """
    if leg_pts.shape[0] < min_pts:
        return coxa_pos.clone(), False
    d = _graph_distance_from_anchor(coxa_pos, leg_pts)
    finite = torch.isfinite(d)
    if int(finite.sum()) < min_pts:
        return coxa_pos.clone(), False
    diff = torch.where(finite, (d - target_cumlen).abs(), torch.full_like(d, float("inf")))
    idx = torch.argmin(diff)
    return leg_pts[idx], True


def solve_chain_ik_multi(
    global_rot_mat,
    coxa_pos,
    rest_J,
    chain_idx,
    constraints,
    min_limits,
    max_limits,
    n_iters=300,
    lr=0.08,
    w_limit=1.0,
    w_reg=0.02,
):
    """Generalisation of `solve_chain_ik` to MULTIPLE position constraints along the same chain,
    solved JOINTLY (not sequentially/greedily -- see the module-level note above
    `solve_chain_ik` for why sequential solving is the specific thing that made Arm B fail).

    constraints: list of (seg_idx, target_pos) pairs. seg_idx in {0..4}: the world position AFTER
    applying segment `seg_idx` (0 = position of the joint after co's rotation, i.e. `tr`; 4 = the
    chain's end, i.e. the tip/pretarsus -- matching `solve_chain_ik`'s single-tip case exactly
    when called with constraints=[(4, target_tip)]). Every extra constraint reduces the
    redundancy this module's oracle probes found dominates the single-tip case: 5 joints x 3
    axes = 15 unknowns; one tip constraint gives 3 equations (heavily underdetermined, confirmed
    empirically in `ik_init_oracle_disambiguation_PROBE.py`); a second constraint at a different
    point along the chain gives 6, still redundant but less so, and -- unlike two points that
    happened to be adjacent -- a point roughly mid-chain and the tip jointly constrain more of
    the chain's shape than either alone.

    Returns (5,3) axis-angle, same row convention as `solve_chain_ik`.
    """
    from pytorch3d.transforms import axis_angle_to_matrix

    from fitter_3d.joint_limits import limit_hinge

    device = coxa_pos.device
    rest_dirs = torch.stack(
        [rest_J[chain_idx[i + 1]] - rest_J[chain_idx[i]] for i in range(5)]
    ).to(device)

    theta = torch.zeros(5, 3, device=device, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    constraints = [(seg_idx, tgt.detach().to(device)) for seg_idx, tgt in constraints]

    for _ in range(n_iters):
        opt.zero_grad()
        R_local = axis_angle_to_matrix(theta)  # (5,3,3)
        R_g = global_rot_mat
        pos = coxa_pos
        pos_loss = 0.0
        for i in range(5):
            R_g = R_g @ R_local[i]
            pos = pos + R_g @ rest_dirs[i]
            for seg_idx, tgt in constraints:
                if seg_idx == i:
                    pos_loss = pos_loss + ((pos - tgt) ** 2).sum()
        limit_loss = limit_hinge(theta, min_limits, max_limits)
        reg_loss = (theta**2).mean()
        loss = pos_loss + w_limit * limit_loss + w_reg * reg_loss
        loss.backward()
        opt.step()

    return theta.detach()


def init_joint_rot_for_specimen_ik(global_rot_aa, trans, rest_J, jnames, target_pts, min_limits_all, max_limits_all, n_iters=300):
    """End-to-end IK-based (N_POSE,3) joint_rot init for ALL joints, geometric legs + zero
    elsewhere -- the `solve_chain_ik`/`estimate_leg_tip` analogue of
    `init_joint_rot_for_specimen`, with the SAME leakage-audit preconditions (H0's shared rigid
    fit + fixed template constants only, no ground truth, no learned classifier) but using the
    FIXED chain-based point assignment (`assign_points_to_legs_chain`, not the coxa-only default
    alias `assign_points_to_legs`) since the coxa-only assignment's mesothoracic-leg starvation
    defect is a known, already-fixed confound this arm should not reinherit.

    min_limits_all, max_limits_all: (N_POSE,3) from `fitter_3d.joint_limits.joint_limit_tensors`
    -- the model's REAL authored per-axis limits, sliced per-leg internally.
    """
    n_pose = len(jnames) - 1
    out = torch.zeros(n_pose, 3, dtype=trans.dtype, device=trans.device)
    chains = leg_chains(jnames)
    anchors, R_root = analytic_coxa_anchors(global_rot_aa, trans, rest_J, chains)
    assigned = assign_points_to_legs_chain(
        target_pts, rest_chain_points(rest_J, chains, global_rot_aa, trans)
    )
    for key, chain_idx in chains.items():
        rows = [chain_idx[i] - 1 for i in range(5)]
        min_l = min_limits_all[rows].to(trans.device)
        max_l = max_limits_all[rows].to(trans.device)
        tip, valid = estimate_leg_tip(anchors[key], assigned[key])
        if valid:
            local_aa = solve_chain_ik(
                R_root, anchors[key], rest_J, chain_idx, tip, min_l, max_l, n_iters=n_iters
            )
        else:
            local_aa = torch.zeros(5, 3, dtype=trans.dtype, device=trans.device)
        for i in range(5):
            out[rows[i]] = local_aa[i]
    return out


def init_joint_rot_for_specimen_ik2(global_rot_aa, trans, rest_J, jnames, target_pts, min_limits_all, max_limits_all, n_iters=300, waypoint_seg_idx=2):
    """Two-constraint variant of `init_joint_rot_for_specimen_ik`: tip PLUS one mid-chain
    waypoint (default `waypoint_seg_idx=2`, the position after fe->ti -- the middle joint of the
    6-joint chain, a structural choice not fit to any result), via `solve_chain_ik_multi`. See
    that function's docstring for why a second constraint matters: the oracle probes found the
    single-tip version underdetermined (~21 deg mean per-joint gap from GT even with the TRUE
    tip position), not a solver bug -- this variant tests whether one extra, weak, single-point
    constraint meaningfully closes that gap. Same preconditions/leakage-audit as the tip-only
    version otherwise.
    """
    n_pose = len(jnames) - 1
    out = torch.zeros(n_pose, 3, dtype=trans.dtype, device=trans.device)
    chains = leg_chains(jnames)
    anchors, R_root = analytic_coxa_anchors(global_rot_aa, trans, rest_J, chains)
    assigned = assign_points_to_legs_chain(
        target_pts, rest_chain_points(rest_J, chains, global_rot_aa, trans)
    )
    for key, chain_idx in chains.items():
        rows = [chain_idx[i] - 1 for i in range(5)]
        min_l = min_limits_all[rows].to(trans.device)
        max_l = max_limits_all[rows].to(trans.device)
        seg_len = torch.stack(
            [(rest_J[chain_idx[i + 1]] - rest_J[chain_idx[i]]).norm() for i in range(5)]
        ).to(trans.device)
        cumlen = torch.cumsum(seg_len, dim=0)
        tip, tip_valid = estimate_leg_tip(anchors[key], assigned[key])
        wpt, wpt_valid = estimate_leg_waypoint(anchors[key], assigned[key], cumlen[waypoint_seg_idx])
        if tip_valid and wpt_valid:
            constraints = [(waypoint_seg_idx, wpt), (4, tip)]
            local_aa = solve_chain_ik_multi(
                R_root, anchors[key], rest_J, chain_idx, constraints, min_l, max_l, n_iters=n_iters
            )
        elif tip_valid:
            local_aa = solve_chain_ik(
                R_root, anchors[key], rest_J, chain_idx, tip, min_l, max_l, n_iters=n_iters
            )
        else:
            local_aa = torch.zeros(5, 3, dtype=trans.dtype, device=trans.device)
        for i in range(5):
            out[rows[i]] = local_aa[i]
    return out


# ============================================================================================
# PCA-coherent initializer (2026-08-25, candidate pool expansion). GT-free, deployable on real
# scans exactly like `init_joint_rot_for_specimen_ik`, but mechanistically distinct: a bilateral-
# symmetry-averaged candidate was considered and REJECTED before being built -- an empirical
# check (`mirror`'s own achieved error) found true left/right leg-pose distances in synth_clean
# average 33.7 deg (range 27.8-42.2), comparable to or worse than this project's own
# `proximal_30deg` condition already shown to be damaging, so specimens are NOT close to
# bilaterally symmetric and averaging with a mirrored counterpart would inject a false premise.
#
# This candidate instead directly exploits the basin map's own strongest finding: `coherent`
# (root-only rotation, downstream joints left at rest) was far less damaging than the same
# magnitude of independent per-joint noise, and `solve_chain_ik`'s minimum-norm redundant pick
# has no mechanism to prefer a coherent solution over an incoherent one (confirmed by the oracle
# disambiguation probe). This estimates the leg's overall pointing direction via PCA on its own
# assigned scan points (closed-form, no optimization loop -- a genuinely different mechanism from
# constrained IK), and applies it ONLY at the coxa, leaving trochanter->pretarsus at exactly rest
# -- i.e. constructing a `coherent`-STRUCTURED initializer from real geometry instead of from a
# perturbed ground truth.
# ============================================================================================


def estimate_leg_pca_direction(coxa_pos, leg_pts, min_pts=MIN_PTS_PER_BAND):
    """Leg's overall pointing direction from the coxa, via PCA (top principal component) on its
    own assigned scan points. Sign resolved to point AWAY from the coxa (toward the points'
    centroid), since PCA alone leaves the axis's sign ambiguous. Returns (direction (3,) unit
    vector, valid bool); direction is None (a zero-rotation fallback) when invalid."""
    if leg_pts.shape[0] < min_pts:
        return None, False
    centered = leg_pts - leg_pts.mean(dim=0, keepdim=True)
    _, _, Vt = torch.linalg.svd(centered)
    direction = Vt[0]
    centroid_dir = leg_pts.mean(dim=0) - coxa_pos
    if torch.dot(direction, centroid_dir) < 0:
        direction = -direction
    return direction / direction.norm().clamp_min(1e-8), True


def init_joint_rot_for_specimen_coherent(global_rot_aa, trans, rest_J, jnames, target_pts, direction_fn):
    """Shared skeleton behind every coxa-only 'coherent' candidate (PCA / cluster-axis / tip-
    direction, 2026-08-25 multi-start pool): rotates ONLY each leg's coxa (via `_align_rotation`,
    closed-form, no optimization loop) toward whatever direction `direction_fn(coxa_pos, leg_pts)`
    estimates; trochanter->pretarsus left at EXACTLY rest -- the `coherent` family's own
    definition (basin map, 2026-08-25: coherent error is far less damaging than the same
    magnitude of incoherent per-joint noise), applied to a real-scan estimate instead of a
    perturbed ground truth. `direction_fn` is the ONLY thing that differs between candidates;
    factored out here per this project's "unify repeated code" convention rather than copy-pasting
    this loop for each new estimator.

    direction_fn: (coxa_pos (3,), leg_pts (P,3)) -> (direction (3,) unit vector or None, valid bool).
    Same leakage-audit preconditions as `init_joint_rot_for_specimen_ik` (H0's shared rigid fit +
    fixed template constants only), using the fixed chain-based point assignment.
    """
    n_pose = len(jnames) - 1
    out = torch.zeros(n_pose, 3, dtype=trans.dtype, device=trans.device)
    chains = leg_chains(jnames)
    anchors, R_root = analytic_coxa_anchors(global_rot_aa, trans, rest_J, chains)
    assigned = assign_points_to_legs_chain(
        target_pts, rest_chain_points(rest_J, chains, global_rot_aa, trans)
    )
    for key, chain_idx in chains.items():
        row_co = chain_idx[0] - 1
        rest_dir0 = rest_J[chain_idx[1]] - rest_J[chain_idx[0]]
        direction, valid = direction_fn(anchors[key], assigned[key])
        if valid:
            R_global_co = _align_rotation(rest_dir0.unsqueeze(0), direction.unsqueeze(0))[0]
            R_local_co = R_root.transpose(-1, -2) @ R_global_co
            out[row_co] = matrix_to_axis_angle(R_local_co.unsqueeze(0))[0]
        # else: leave at zero -- same fallback convention as every other arm here.
    return out


def init_joint_rot_for_specimen_pca_coherent(global_rot_aa, trans, rest_J, jnames, target_pts):
    """PCA-direction coherent candidate. Thin wrapper over `init_joint_rot_for_specimen_coherent`
    with `estimate_leg_pca_direction` -- kept as its own named function for backward compatibility
    with existing callers (`generate_pca_coherent_init.py`)."""
    return init_joint_rot_for_specimen_coherent(
        global_rot_aa, trans, rest_J, jnames, target_pts, estimate_leg_pca_direction
    )


# ============================================================================================
# Multi-start candidate pool, part 2 (2026-08-25): two MORE independent direction estimators,
# so a per-leg 'coherent' candidate can be generated from three mechanistically distinct sources
# -- PCA's top variance axis, a near/far cluster split, and the tip-finding estimate `solve_
# chain_ik` already uses -- rather than trusting a single point estimate (which is exactly why
# PCA-coherent alone underperformed: coherent-BY-CONSTRUCTION does not fix an estimate that is
# simply wrong, only an estimate that is noisy in a way multiple independent guesses can outvote
# via GT-free selection). A bilateral-symmetry-derived 4th candidate was considered and rejected
# already (see note above `estimate_leg_pca_direction`) -- these two are genuinely new geometric
# mechanisms, not variations on symmetry.
# ============================================================================================


def estimate_leg_cluster_axis_direction(coxa_pos, leg_pts, min_pts=MIN_PTS_PER_BAND):
    """Second independent direction estimate, mechanistically distinct from PCA's variance-
    maximizing axis: split the leg's assigned points into near/far halves by GRAPH (geodesic-
    like) distance from the coxa (median split, reusing `_graph_distance_from_anchor` -- the
    same distance notion `estimate_leg_curve`/`estimate_leg_tip` already validated as the correct
    chain-position proxy, see their docstrings for why plain Euclidean distance fails even at
    rest), then take the direction from the near-half centroid to the far-half centroid. This
    directly encodes chain progression (near points vs far points) rather than overall point-
    cloud variance, so it can disagree with PCA in a genuinely informative way (e.g. a leg
    L-bent in a plane PCA's top axis would happily point along the WRONG arm of the bend).

    Returns (direction (3,) unit vector, valid bool). Requires >=2*min_pts total (>=min_pts on
    each side of the median split) -- stricter than the single-estimate functions above, since a
    degenerate split (e.g. all points on one side) would make the two centroids nearly coincident
    and the direction numerically meaningless.
    """
    if leg_pts.shape[0] < 2 * min_pts:
        return None, False
    d = _graph_distance_from_anchor(coxa_pos, leg_pts)
    finite = torch.isfinite(d)
    if int(finite.sum()) < 2 * min_pts:
        return None, False
    d_fin = d[finite]
    pts_fin = leg_pts[finite]
    med = torch.median(d_fin)
    near_mask = d_fin <= med
    far_mask = ~near_mask
    if int(near_mask.sum()) < min_pts or int(far_mask.sum()) < min_pts:
        return None, False
    near_c = pts_fin[near_mask].mean(dim=0)
    far_c = pts_fin[far_mask].mean(dim=0)
    direction = far_c - near_c
    return direction / direction.norm().clamp_min(1e-8), True


def estimate_leg_tip_direction(coxa_pos, leg_pts, min_pts=MIN_PTS_PER_BAND):
    """Third independent direction estimate: reuses `estimate_leg_tip` (the same most-distal-
    point estimate `solve_chain_ik`'s tip constraint is built from) but only as a DIRECTION
    (coxa -> tip, normalised), not as an IK target -- i.e. this is 'IK's direction' without
    running IK's 300-step optimiser, since a coherent candidate only ever uses the coxa rotation
    anyway (downstream joints stay at rest by the `coherent` family's own definition). Cheap and
    mechanistically distinct from both PCA (variance axis, uses ALL assigned points) and the
    cluster-axis split (near/far centroids): this depends on exactly ONE point, the single most-
    distal one, so it disagrees with the other two specifically when that one point is an outlier
    or when the leg's true shape is not well summarised by "coxa toward furthest point" (e.g. a
    leg bent back toward the body, where the furthest point may not lie in the true overall
    pointing direction).

    Returns (direction (3,) unit vector, valid bool).
    """
    tip, valid = estimate_leg_tip(coxa_pos, leg_pts, min_pts=min_pts)
    if not valid:
        return None, False
    direction = tip - coxa_pos
    return direction / direction.norm().clamp_min(1e-8), True


def init_joint_rot_for_specimen_cluster_coherent(global_rot_aa, trans, rest_J, jnames, target_pts):
    """Cluster-axis coherent candidate. Thin wrapper over `init_joint_rot_for_specimen_coherent`
    with `estimate_leg_cluster_axis_direction`."""
    return init_joint_rot_for_specimen_coherent(
        global_rot_aa, trans, rest_J, jnames, target_pts, estimate_leg_cluster_axis_direction
    )


def init_joint_rot_for_specimen_tipdir_coherent(global_rot_aa, trans, rest_J, jnames, target_pts):
    """Tip-direction coherent candidate. Thin wrapper over `init_joint_rot_for_specimen_coherent`
    with `estimate_leg_tip_direction`."""
    return init_joint_rot_for_specimen_coherent(
        global_rot_aa, trans, rest_J, jnames, target_pts, estimate_leg_tip_direction
    )
