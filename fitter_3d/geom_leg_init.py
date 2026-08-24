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
