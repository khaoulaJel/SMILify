"""Correspondence-accuracy audit: true anatomical segment x matched anatomical segment,
confusion matrices at leg-level, within-leg-segment-level, and antenna-level, run against every
converged arm that shares the `synth_clean` ground-truth corpus (topology-identical to the
template by construction -- see `diagnostics/moonshot/SYNTHETIC_ROUNDTRIP.md`).

This is the metric the task asked to build BEFORE touching the correspondence mechanism itself:
independent of Chamfer/edge/etc., so "did the anatomical constraint actually fix
correspondence" can be checked directly instead of inferred from a smoother loss curve.

Run: `python diagnostics/correspondence_accuracy/run_audit.py` (needs the pytorch3d conda env).
"""

import json
import os
import sys

import numpy as np
import torch
from pytorch3d.io import load_obj

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MOON = os.path.join(REPO, "diagnostics", "moonshot")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "diagnostics", "morphometrics"))
sys.path.insert(0, HERE)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import measure as ms  # noqa: E402
import labels as lb  # noqa: E402
import confusion as cf  # noqa: E402
import geodesic as geo  # noqa: E402

N_SAMPLE = 8000  # matches HierarchicalStage / optimise_moonshot's own chamfer sample size
DEVICE = "cpu"

# (run_name, corpus_dir, npz_path) -- every arm evaluated here shares the synth_clean corpus,
# so results are directly comparable to each other and are not confounded by different ground
# truth. npz_path=None defaults to the moonshot handoff layout (diagnostics/moonshot/runs/
# <run>/Stage_3_deform_fine.npz); pass an explicit path for arms audited straight off
# optimise_hierarchical.py's own output (which has no moonshot handoff at all).
RUNS = [
    ("SYN_clean_w5", "synth_clean", None),  # stock baseline, no GNC / anatomical constraint
    ("SYN_clean_pose25_gnclegonly_w5", "synth_clean", None),  # GNC, leg-only partitioning
    ("SYN_clean_pose25_splitdistal_w5", "synth_clean", None),  # --split_distal (Task 6 candidate)
    ("SYN_clean_pose25_gtinit_w5", "synth_clean", None),  # ground-truth pose init (Task 6 ceiling)
    ("SYN_clean_pose25_stratq05_w5", "synth_clean", None),  # stratified target sampling, q=0.05
    ("SYN_clean_pose25_stratq10_w5", "synth_clean", None),  # stratified target sampling, q=0.10
    ("SYN_clean_pose25_stratq20_w5", "synth_clean", None),  # stratified target sampling, q=0.20
    ("SYN_clean_pose25_c5protected_w5", "synth_clean", None),  # protected small-group robust kernel
    # Hierarchical-stage-only arms (no moonshot handoff -- see submit_split_segments_probe.sbatch).
    # MoonshotStage (fitter_3d/trainer_moonshot.py) has no TargetPartition/partitioned-chamfer
    # mechanism at all, so every *_w5 row above was already past the point where any partition
    # (including split_distal) could still act by the time Stage_3_deform_fine was captured --
    # these two are audited straight off HierarchicalStage's own final H3_deform output instead,
    # so the partitioned mechanism being changed is actually what's being measured.
    (
        "HIER_baseline",
        "synth_clean",
        os.path.join(HERE, "hier_runs", "baseline", "H3_deform.npz"),
    ),
    (
        "HIER_split_segments",
        "synth_clean",
        os.path.join(HERE, "hier_runs", "split_segments", "H3_deform.npz"),
    ),
    # Anatomical Initialization Ceiling Test (2026-08-20, diagnostics/anatomical_pose_init/).
    # Arm A/C reuse the SYN_clean_w5 / SYN_clean_pose25_gtinit_w5 rows above (same corpus, same
    # D1 recipe) rather than re-fitting -- ACI_zero and ACI_gt are deliberately NOT separate
    # rows, to avoid double-counting the same underlying data under two names.
    ("ACI_cheap", "synth_clean", None),  # cheap_anatomical init (simple_leg_heuristic.py)
    ("ACI_learned", "synth_clean", None),  # learned init (train_leg_pose_regressor.py, 25pc corpus)
    # Basin-structure follow-up (2026-08-20): magnitude-matched (~23deg) structured perturbations
    # of GT pose, see generate_structured_perturbation_init.py. Isolates error STRUCTURE from
    # error MAGNITUDE, since ACI_learned's advantage over zero-init came at near-identical
    # magnitude (22.79 vs 23.09deg).
    ("ACI_D_random", "synth_clean", None),
    ("ACI_E_proximal", "synth_clean", None),
    ("ACI_F_distal", "synth_clean", None),
    # Pose-error sweep (2026-08-20 follow-up, pose_noise_sweep.py): GT pose + fixed-magnitude
    # random-axis noise on the 36 leg joints only, everything else at exact GT. Level 0 is the
    # existing SYN_clean_pose25_gtinit_w5 row above, reused, not duplicated here.
    ("ACI_noise5", "synth_clean", None),
    ("ACI_noise10", "synth_clean", None),
    ("ACI_noise15", "synth_clean", None),
    ("ACI_noise20", "synth_clean", None),
    ("ACI_noise25", "synth_clean", None),
    ("ACI_noise30", "synth_clean", None),
    # Capacity-ceiling arm: GT-pose init + scale_cap=0.052 (D1_PROD.yaml), compared against the
    # existing SYN_clean_pose25_gtinit_w5 (GT-pose + free/uncapped segment scale, D1_low.yaml).
    ("ACI_gtcapacitycap", "synth_clean", None),
    # Phase 2 basin map (2026-08-25, diagnostics/anatomical_pose_init/CHAIN_OF_THOUGHT_20260824_WSL.md
    # continuation): extends ACI_D/E/F (random/proximal/distal) across a 15/23/30deg magnitude
    # sweep, plus two new perturbation families -- coherent (root-joint-only rigid leg rotation)
    # and swap (nearest-other-specimen's real GT leg pose, no synthetic noise, single natural
    # magnitude) -- see generate_structured_perturbation_init.py for full construction and
    # basin_map_manifest.csv for the achieved per-specimen initial error every row here should be
    # joined against.
    ("BASIN_random_15deg", "synth_clean", None),
    ("BASIN_random_23deg", "synth_clean", None),
    ("BASIN_random_30deg", "synth_clean", None),
    ("BASIN_proximal_15deg", "synth_clean", None),
    ("BASIN_proximal_23deg", "synth_clean", None),
    ("BASIN_proximal_30deg", "synth_clean", None),
    ("BASIN_distal_15deg", "synth_clean", None),
    ("BASIN_distal_23deg", "synth_clean", None),
    ("BASIN_distal_30deg", "synth_clean", None),
    ("BASIN_coherent_15deg", "synth_clean", None),
    ("BASIN_coherent_23deg", "synth_clean", None),
    ("BASIN_coherent_30deg", "synth_clean", None),
    ("BASIN_swap", "synth_clean", None),
    ("BASIN_mirror", "synth_clean", None),
    # A1 corrected-sampler basin-map rerun (2026-08-25, generate_correlated_chain_parameters,
    # rho=0.6): identical 13-condition basin map, but built from
    # anatomical_pose_init_corr_rho06 (lag-1 AR(1) whole-chain-coupled GT poses) instead of
    # synth_clean (i.i.d. GT poses) -- corpus_dir differs accordingly, since GT/meshes differ.
    ("BASINCORR_random_15deg", "anatomical_pose_init_corr_rho06", None),
    ("BASINCORR_random_23deg", "anatomical_pose_init_corr_rho06", None),
    ("BASINCORR_random_30deg", "anatomical_pose_init_corr_rho06", None),
    ("BASINCORR_proximal_15deg", "anatomical_pose_init_corr_rho06", None),
    ("BASINCORR_proximal_23deg", "anatomical_pose_init_corr_rho06", None),
    ("BASINCORR_proximal_30deg", "anatomical_pose_init_corr_rho06", None),
    ("BASINCORR_distal_15deg", "anatomical_pose_init_corr_rho06", None),
    ("BASINCORR_distal_23deg", "anatomical_pose_init_corr_rho06", None),
    ("BASINCORR_distal_30deg", "anatomical_pose_init_corr_rho06", None),
    ("BASINCORR_coherent_15deg", "anatomical_pose_init_corr_rho06", None),
    ("BASINCORR_coherent_23deg", "anatomical_pose_init_corr_rho06", None),
    ("BASINCORR_coherent_30deg", "anatomical_pose_init_corr_rho06", None),
    ("BASINCORR_swap", "anatomical_pose_init_corr_rho06", None),
    # WSL-fresh zero-init reference (same code version/machine as the BASIN_* and IK_* arms
    # above -- the original cluster SYN_clean_w5 row is unreachable, see HANDOFF_20260824.md).
    ("SYN_clean_zero_wsl", "synth_clean", None),
    # IK-based init arms (2026-08-25), see generate_ik_init.py.
    ("IK_tip_only", "synth_clean", None),
    ("IK_tip_waypoint", "synth_clean", None),
    # PCA-coherent candidate (2026-08-25), see generate_pca_coherent_init.py.
    ("PCA_coherent", "synth_clean", None),
    # Multi-start coherent pool, part 2 (2026-08-25), see generate_multistart_coherent_init.py.
    ("Cluster_coherent", "synth_clean", None),
    ("Tipdir_coherent", "synth_clean", None),
    # Correspondence-oracle test (2026-08-25, Phase 10 design doc step 1): zero-init pose +
    # PERFECT (ground-truth) partition, via --oracle_gt_partition_from. Ceiling test, not a
    # deployable arm -- see PHASE10_DESIGN_correspondence_network_20260825.md.
    ("Oracle_GT_partition", "synth_clean", None),
    # Dense per-vertex correspondence oracle (2026-08-25, Phase 10 step 1 follow-up): fixes
    # BOTH between-part (leg-level) and within-part (segment-level) correspondence, unlike
    # Oracle_GT_partition above which only fixes the former. See PHASE10 design doc.
    ("Dense_GT_oracle", "synth_clean", None),
    # Metric-ceiling check (2026-08-25): "fitted" = ground truth itself (zero error, by
    # construction). Establishes seg_acc's own ceiling below 1.0, if any, from near-boundary
    # point-sampling ambiguity alone -- before treating 1.0 as the reference for "residual gap."
    ("GT_as_fitted_ceiling", "synth_clean", os.path.join(HERE, "out", "gt_as_fitted.npz")),
    # Weight-dominance check (2026-08-25): w_dense_gt_correspondence=50.0 (~50x the smoke-tested
    # default), pushing edge/laplacian/sym toward negligible relative pull. Resolves whether
    # Dense_GT_oracle's residual seg_acc gap is a tuning artifact or a real capacity limit.
    ("Dense_GT_oracle_dominant", "synth_clean", None),
    # C2 (2026-08-26, Phase 10 B1/B2/C2): trained SMILCorrespondenceNet's dense per-point
    # predictions converted into a pose candidate via geom_leg_init.solve_chain_ik_multi (reused,
    # not new IK code) -- see generate_network_correspondence_init_20260826.py. This is the
    # LEARNED counterpart to Oracle_GT_partition/Dense_GT_oracle above: same mechanism family,
    # estimated instead of given. Evaluated against the B3 pre-registered bar
    # (out_B2_correspondence_net_20260825/PREREGISTRATION_B3_success_bar_20260825.md).
    ("Network_correspondence", "synth_clean", None),
    # Disentangling companion to Network_correspondence (2026-08-26): identical
    # solve_chain_ik_multi conversion pipeline, but fed TRUE per-point segment labels instead of
    # the network's predictions (--oracle_gt_labels). Isolates the IK-conversion/centroid-noise
    # ceiling (Part 1c's known ~14-21deg IK floor) from the network's own prediction accuracy --
    # raw pose error was nearly identical to Network_correspondence's (32.46 vs 33.42deg), so this
    # checks whether that near-parity holds at the actual fitter-outcome level too.
    ("GTLabel_IK_oracle", "synth_clean", None),
    # C4 (2026-08-27): the CSE embedding head's PREDICTED dense correspondence, fed to the fitter
    # as DIRECT vertex identity (--cse_correspondence_from), deliberately bypassing the
    # centroid/IK conversion that Network_correspondence and GTLabel_IK_oracle both used and that
    # was measured to discard ~41% of the achievable gain. Restricted to tr/fe, the only segments
    # whose HELD-OUT direct-vertex retrieval beat chance by a margin
    # (eval_C3_vs_B2_retrieval_20260827); coverage is therefore partial by design.
    # Compare against SYN_clean_zero_wsl (same zero-init recipe, no correspondence term) for the
    # effect, and against Dense_GT_oracle for how much of the oracle's gain it recovers.
    ("CSE_correspondence", "synth_clean", None),
    # C5 (2026-08-27): same CSE head, but correspondences filtered by CYCLE CONSISTENCY instead of
    # cosine similarity, and emitted for ALL segments rather than the tr/fe whitelist. Cosine was
    # measured to RAISE mean retrieval error at top-50% (0.726 -> 1.044) while cycle consistency
    # LOWERS it (0.726 -> 0.449) -- see confidence_signal_comparison_20260827.py. Because the filter
    # needs no held-out ground truth, this arm is the deployable one; coverage rises ~10% -> ~28%
    # of template vertices. Compare against CSE_correspondence for the filter+coverage effect.
    ("CSE_cycle_all", "synth_clean", None),
    # C5b/C5c (2026-08-27): deconfounding cells. C4 changed neither filter nor breadth from the
    # tr/fe whitelist; C5 changed BOTH at once. C5b isolates the FILTER (cycle, tr/fe only) and C5c
    # isolates the useful part of the BREADTH (cycle, tr/fe/co).
    # Pre-registered prediction, made before any of these landed: C5c > C5, because segment-
    # conditional survivor error measured against each segment's OWN random baseline shows co is a
    # genuine gain (0.285 vs 0.490 random) while ti (0.476 vs 0.339) and ta (0.936 vs 0.363) remain
    # WORSE THAN RANDOM even after cycle filtering -- so the all-segments arm injects net-harmful
    # correspondences. See out_confidence_probe_20260827/confidence_signals.json.
    ("CSE_cycle_trfe", "synth_clean", None),
    ("CSE_cycle_trfeco", "synth_clean", None),
    # C7 (2026-08-27): SHUFFLE CONTROL for what ti/ta actually contribute. Adding ti/ta HELPED
    # (+0.0203) despite per-point retrieval worse than random, which falsified the exclusion
    # rationale. Three-way test, all on synth_clean with the identical D1 recipe:
    #   CSE_cycle_all      real ti/ta correspondence
    #   CSE_shuf_tita      ti/ta vertices replaced by RANDOM same-segment vertices  <-- this row
    #   CSE_cycle_trfeco   no ti/ta constraint at all
    # shuffled ~= real          -> correspondence IDENTITY does not matter, only a soft regional pull
    # shuffled ~= no-constraint -> the soft-prior story is ALSO wrong; the gain is not from ti/ta
    # shuffled between the two  -> identity matters somewhat, coverage matters somewhat
    ("CSE_shuf_tita", "synth_clean", None),
    # C6 (2026-08-27): POWER run. At n=12 all four CSE configurations were statistically
    # indistinguishable from each other (every head-to-head n.s.), so "ship the numerically
    # highest" was not a defensible basis for committing scarce bench50 budget. These arms repeat
    # the configuration comparison on an INDEPENDENT 48-specimen synthetic corpus (seed 20260827,
    # not a superset of synth_clean's 20260806), where synthetic fits are the cheap resource.
    ("P48_zero", "synth_power48", None),
    ("P48_cse_all", "synth_power48", None),
    ("P48_cse_trfeco", "synth_power48", None),
    ("P48_cse_trfe", "synth_power48", None),
    # C12 (2026-08-27): did hard-negative mining change `ti`'s ROLE? C7 showed shuffling ti/ta
    # within-segment cost nothing (+0.0010, sign p=1.0) -- coarse regional pull, not correspondence.
    # C11 moved ti direct retrieval from at-chance to beating chance. These three arms differ ONLY
    # in how `ti` is treated, so the shuffle contrast isolates whether identity now matters.
    # See PREREGISTRATION_C12_ti_role_20260827.md for the readings fixed in advance.
    ("C12_c11_all", "synth_clean", None),
    ("C12_c11_shufti", "synth_clean", None),
    ("C12_c11_noti", "synth_clean", None),
    # C13 (2026-08-28): does weighting the dense correspondence term by inverse MEASURED inter-leg
    # tolerance reallocate gradient to the coxa? The coxa is the best-placed segment in absolute
    # terms but sits at risk = placement_error / inter-leg_tolerance = 0.995 vs ~0.21 elsewhere.
    # Arms differ ONLY in the per-vertex weight shape. See PREREGISTRATION_C13_tolerance_weighting_20260828.md.
    ("C13_uniform", "synth_power48", None),
    ("C13_invtol", "synth_power48", None),
    ("C13_invtol2", "synth_power48", None),
    # C14 PREREQUISITE PROBE (2026-08-28): CEILING on any pose initializer under the current
    # recipe -- joint_rot seeded from the corpus's own ground truth. Reads GT, so these are
    # ceilings and can never be reported as deployable results. With P48_zero and C13_uniform
    # already listed above these complete a 2x2 (init x correspondence); the gap
    # C14p_gtinit_cse - C13_uniform is the headroom a perfect init has ON TOP of correspondence,
    # and is what decides whether C14 gets built at all.
    # See DESIGN_C14_kinematic_init_20260828.md.
    ("C14p_gtinit_cse", "synth_power48", None),
    ("C14p_gtinit_nocse", "synth_power48", None),
    # E0 (2026-08-28): the METRIC'S OWN FLOOR. "Fit" = the ground-truth mesh itself, so placement
    # is exactly correct by construction. No fit is run. This calibrates what a co leg-level error
    # of 0.41 means: if E0 reads ~0, the coxa is trivially representable (P48 targets are
    # generated FROM the model) and every coxal result so far is an optimization finding. If E0
    # reads high, the metric itself is saturated on the coxa and the whole coxal thread is
    # mis-framed. Written before E1/E2 because it decides whether they are interpretable.
    ("E0_identity", "synth_power48",
     "diagnostics/anatomical_pose_init/out_E0_identity_20260828/identity.npz"),
    # Hier-only (H2_joint) controls. The E1/E2 freeze arms CANNOT run the D1 moonshot stage --
    # optimise_moonshot has no freeze flags and would unfreeze pose -- so they are hier-only, and
    # need hier-only baselines to pair against. Same runs as the rows above, read one stage earlier.
    ("C13_uniform_hier_H2", "synth_power48",
     "diagnostics/moonshot/runs/C13_uniform_hier/H2_joint.npz"),
    ("C14p_gtinit_nocse_hier_H2", "synth_power48",
     "diagnostics/moonshot/runs/C14p_gtinit_nocse_hier/H2_joint.npz"),
    # E1/E2 (2026-08-28): pose HARD-frozen at ground truth, shape free (E1); plus log_beta_scales
    # hard-frozen at ground truth (E2). Hier-only by necessity -- optimise_moonshot has no freeze
    # flags. Bar fixed in PREREGISTRATION_E1E2_frozen_pose_20260828.md before these ran.
    ("E1_frozen_pose", "synth_power48",
     "diagnostics/moonshot/runs/E1_frozen_pose_hier/H2_joint.npz"),
    ("E2_frozen_pose_gtscale", "synth_power48",
     "diagnostics/moonshot/runs/E2_frozen_pose_gtscale_hier/H2_joint.npz"),
    # D1 thread, step 1 (2026-08-28): the full stage trajectory of ONE arm (C13_uniform), so the
    # co=0.2564 -> 0.4115 regression can be localised to a stage before any term is toggled.
    # Free -- every npz already exists. See OPEN_THREAD_D1_coxal_degradation_20260828.md.
    ("TRAJ_C13u_H0", "synth_power48", "diagnostics/moonshot/runs/C13_uniform_hier/H0_body.npz"),
    ("TRAJ_C13u_H1", "synth_power48", "diagnostics/moonshot/runs/C13_uniform_hier/H1_legs.npz"),
    ("TRAJ_C13u_H3", "synth_power48", "diagnostics/moonshot/runs/C13_uniform_hier/H3_deform.npz"),
    ("TRAJ_C13u_D1_coarse", "synth_power48",
     "diagnostics/moonshot/runs/C13_uniform/Stage_2_deform_coarse.npz"),
    # H_B (2026-08-28): one existing D1 term zeroed per arm, all re-running D1 from the SAME
    # C13_uniform_hier/H2_joint.npz handoff. Prediction recorded in the pre-registration before
    # these ran: none clears the 70% bar (co <= 0.3030).
    ("D1_no_edge", "synth_power48", None),
    ("D1_no_sym", "synth_power48", None),
    ("D1_no_offset", "synth_power48", None),
    ("D1_no_midline", "synth_power48", None),
    ("D1_no_lap", "synth_power48", None),
    ("D1_no_deform", "synth_power48", None),
    # H_A (2026-08-28): the correspondence term carried THROUGH D1. Port verified byte-identical
    # when off (20/20 arrays, both stages, n=48) before this ran.
    ("D1_with_cse", "synth_power48", None),
    # H_A2 (2026-08-28): same as D1_with_cse but scheme:pose, removing deform_verts. Tests whether
    # ANY of H_A's 127% recovery is anatomical. Bar in PREREGISTRATION_HA2_deform_off_20260828.md.
    ("D1_with_cse_nodeform", "synth_power48", None),
    # H_A3 (2026-08-28): fresh independent replicate, --seed 1. PRIMARY reading for the corrected
    # mechanism check; D1_with_cse_nodeform is the already-seen supporting replicate.
    ("D1_with_cse_nodeform_s1", "synth_power48", None),
    # H_A4 (2026-08-28): closing arm of the coxa thread -- _s1 plus w_beta_prior 0.002.
    ("D1_cse_nodeform_betaprior", "synth_power48", None),
]


def audit_run(run, corpus, npz_path, template_faces, face_lab, vlabels, n_verts_template, geo_lookup, rng, n_specimens=None):
    d = np.load(npz_path)
    spec_labels = [str(x) for x in d["labels"]]
    fitted = d["verts"].astype(np.float64)
    assert fitted.shape[1] == n_verts_template, (
        f"{run}: fitted vertex count {fitted.shape[1]} != template {n_verts_template} -- "
        "fitted mesh does not share template topology, per-vertex labels invalid"
    )

    leg_acc = cf.ConfusionAccumulator(list(lb.LEGS))
    seg_acc = cf.ConfusionAccumulator(list(lb.LEG_SEGMENTS))
    ant_side_acc = cf.ConfusionAccumulator(list(lb.SIDES))
    ant_seg_acc = cf.ConfusionAccumulator(list(lb.ANTENNA_SEGMENTS))
    # geodesic error (mesh-surface distance, template units), pooled by TRUE segment and
    # separately by (true_seg, matched_seg) pair -- lets pt->ta / ti->ta be read off directly
    # instead of re-deriving them from the confusion matrix's row/col ordering.
    geo_by_true_seg = {s: [] for s in lb.LEG_SEGMENTS}
    geo_by_pair = {}  # (true_seg, matched_seg) -> list of distances
    per_specimen_leg_acc = []  # [(stem, leg_acc)] -- for trajectory plots (protocol section 4)
    per_specimen_per_leg_acc = []  # [(stem, {leg_name: accuracy})] -- row-normalized diagonal of
    # THIS specimen's own leg confusion matrix, i.e. per-leg (not just per-specimen-aggregate)
    # accuracy -- needed to test whether cross-estimator direction spread (multi-start coherent
    # candidates, 2026-08-25) predicts WHICH legs go wrong, not just which specimens.
    per_specimen_per_segment_leg_acc = []  # [(stem, {leg_seg: leg-level accuracy})] -- the
    # LEG-level match rate broken down by the point's TRUE leg segment, i.e. "of the points that
    # truly lie on the coxa, what fraction were matched to the correct leg at all". This is the
    # primary endpoint C13's bar was written against (co leg-level error = 1 - this['co']), and it
    # is NOT derivable from the three metrics above: per_specimen_per_leg_acc splits by which LEG,
    # this splits by which SEGMENT, and per_specimen_seg_acc is conditioned on the leg already
    # being right. Costs nothing extra -- it reuses leg_confusion's existing predictions.
    per_specimen_seg_acc = []  # [(stem, seg_acc)] -- WITHIN-part (segment-level, co/tr/fe/ti/ta/
    # pt) accuracy per specimen, restricted to points already correctly leg-assigned -- the
    # per-specimen distribution of the 83.3%-within-part error (why_partitions_null.py), pulled
    # BEFORE looking at the dense-GT-oracle result (2026-08-25, Phase 10 step 1 follow-up), so a
    # pattern in what moves can be checked against a pattern already known about the corpus.

    n = n_specimens or len(spec_labels)
    for i, spec_lab in enumerate(spec_labels[:n]):
        stem = spec_lab[:-4] if spec_lab.endswith(".obj") else spec_lab
        obj_path = os.path.join(MOON, corpus, f"{stem}.obj")
        ov, of, _ = load_obj(obj_path, load_textures=False)
        tgt_verts, tgt_faces = ov.numpy(), of.verts_idx.numpy()
        assert tgt_faces.shape == template_faces.shape, (
            f"{run}/{stem}: target face topology {tgt_faces.shape} != template "
            f"{template_faces.shape} -- ground-truth face labels are invalid for this specimen"
        )

        pts, point_lab = cf.sample_true_labeled_points(tgt_verts, tgt_faces, face_lab, N_SAMPLE, rng)
        pts_t = torch.tensor(pts, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        # Gate on the leg_id/ant_seg field itself, not the independently-voted "region" field:
        # a face straddling two DIFFERENT legs (rare, only near adjacent coxae) can have a
        # 3-way tie among its vertices' leg_id ('l1_r' / 'l2_r' / None-from-a-body-vertex),
        # which majority-vote resolves to None even though "region" majority-votes to 'leg'
        # (2 of 3 vertices are leg vertices, just not the SAME leg). Gating on leg_id directly
        # drops these genuinely-ambiguous boundary points instead of assigning them a
        # meaningless true label.
        true_is_leg = point_lab["leg_id"] != None  # noqa: E711
        true_is_ant = point_lab["ant_seg"] != None  # noqa: E711

        fitted_i = torch.tensor(fitted[i], dtype=torch.float32, device=DEVICE).unsqueeze(0)

        leg_acc_i, leg_pred, pts_leg, true_leg_sub = cf.leg_confusion(
            point_lab["leg_id"], true_is_leg, pts_t, fitted_i, vlabels
        )
        seg_acc_i = cf.within_leg_segment_confusion(
            point_lab["leg_seg"], true_is_leg, pts_leg, true_leg_sub, leg_pred, fitted_i, vlabels
        )
        side_acc_i, antseg_acc_i = cf.antenna_confusion(
            point_lab["ant_side"], point_lab["ant_seg"], true_is_ant, pts_t, fitted_i, vlabels
        )
        true_vidx_here, matched_vidx, legs_here = cf.within_leg_segment_geodesic(
            point_lab["true_vertex_idx"], true_is_leg, pts_leg, true_leg_sub, leg_pred, fitted_i, vlabels
        )

        leg_acc_i_dict = leg_acc_i.as_dict()
        per_specimen_leg_acc.append((stem, leg_acc_i_dict["accuracy"]))
        row_norm = np.array(leg_acc_i_dict["row_normalized"])
        per_leg = {name: float(row_norm[j, j]) for j, name in enumerate(leg_acc_i_dict["names"])}
        per_specimen_per_leg_acc.append((stem, per_leg))
        per_specimen_seg_acc.append((stem, seg_acc_i.as_dict()["accuracy"]))

        # Per-TRUE-SEGMENT leg-level accuracy. leg_pred and true_leg_sub are both over the
        # true_is_leg subset (confusion.leg_confusion), so leg_seg gathers with the same mask.
        true_seg_sub = point_lab["leg_seg"][true_is_leg]
        if len(leg_pred):
            _correct = np.asarray(leg_pred == true_leg_sub)
            per_seg_leg = {
                _s: float(_correct[true_seg_sub == _s].mean())
                for _s in lb.LEG_SEGMENTS if (true_seg_sub == _s).any()
            }
        else:
            per_seg_leg = {}
        per_specimen_per_segment_leg_acc.append((stem, per_seg_leg))

        leg_acc.M += leg_acc_i.M
        leg_acc.n_unmapped_true += leg_acc_i.n_unmapped_true
        seg_acc.M += seg_acc_i.M
        ant_side_acc.M += side_acc_i.M
        ant_seg_acc.M += antseg_acc_i.M

        if len(true_vidx_here):
            # true_vertex_idx is the SAMPLED POINT's own dominant-barycentric-weight corner,
            # computed independently of the face-level leg_id majority vote true_is_leg/
            # legs_here rely on -- on a boundary face they can disagree (majority vote says
            # 'leg', the specific point's nearest corner happens to be the one non-leg vertex),
            # which would look up a vertex absent from that leg's geodesic table. Same
            # genuinely-ambiguous-boundary-point situation as the leg_id/region gating earlier
            # in this function; drop the inconsistent few rather than crash or mislabel.
            consistent = vlabels["leg_id"][true_vidx_here] == legs_here
            true_vidx_here, matched_vidx, legs_here = (
                true_vidx_here[consistent], matched_vidx[consistent], legs_here[consistent]
            )
        if len(true_vidx_here):
            dists = geo_lookup.dist(legs_here, true_vidx_here, matched_vidx)
            true_seg_of_point = vlabels["leg_seg"][true_vidx_here]
            matched_seg_of_point = vlabels["leg_seg"][matched_vidx]
            for ts, ms_, dd_ in zip(true_seg_of_point, matched_seg_of_point, dists):
                if ts is None or np.isnan(dd_):
                    continue
                geo_by_true_seg[ts].append(dd_)
                geo_by_pair.setdefault((ts, ms_), []).append(dd_)

    def _stats(xs):
        xs = np.asarray(xs, dtype=np.float64)
        if len(xs) == 0:
            return dict(n=0, mean=float("nan"), median=float("nan"))
        return dict(n=int(len(xs)), mean=float(xs.mean()), median=float(np.median(xs)))

    geodesic_summary = dict(
        by_true_segment={s: _stats(v) for s, v in geo_by_true_seg.items()},
        by_pair={f"{t}->{p}": _stats(v) for (t, p), v in geo_by_pair.items()},
    )

    return dict(
        run=run,
        corpus=corpus,
        n_specimens=n,
        leg_confusion=leg_acc.as_dict(),
        within_leg_segment_confusion=seg_acc.as_dict(),
        antenna_side_confusion=ant_side_acc.as_dict(),
        antenna_segment_confusion=ant_seg_acc.as_dict(),
        geodesic=geodesic_summary,
        per_specimen_leg_acc=per_specimen_leg_acc,
        per_specimen_per_leg_acc=per_specimen_per_leg_acc,
        per_specimen_seg_acc=per_specimen_seg_acc,
        per_specimen_per_segment_leg_acc=per_specimen_per_segment_leg_acc,
    )


def main():
    M = ms.load_model()
    template_faces = np.asarray(M["dd"]["f"])
    vlabels = lb.vertex_labels(M["jnames"], M["dominant"])
    face_lab = lb.face_labels(template_faces, vlabels)
    n_verts_template = len(M["dominant"])
    rng = np.random.default_rng(1)

    v_template = np.asarray(M["v_template"], dtype=np.float64)
    geo_tables = geo.load_or_build_leg_tables(v_template, template_faces, vlabels)
    geo_lookup = geo.LegGeodesicLookup(geo_tables)

    # MERGE with whatever is already on disk, rather than overwrite it -- most historical fit
    # outputs (diagnostics/moonshot/runs/, gitignored, multi-GB) are NOT present on every machine
    # this script runs on (confirmed 2026-08-26: 40/61 RUNS entries had no backing .npz on a fresh
    # cluster checkout, including SYN_clean_zero_wsl/Oracle_GT_partition/Dense_GT_oracle). Silently
    # overwriting the tracked JSON with only the subset computable HERE would permanently discard
    # every other row's recorded result -- the small out/*.json IS the durable record for a run
    # whose multi-GB npz is gone, so a row already present but not recomputable this run is kept
    # as-is, not dropped.
    out_path = os.path.join(HERE, "out", "correspondence_confusion.json")
    existing_by_run = {}
    if os.path.isfile(out_path):
        with open(out_path) as f:
            existing_by_run = {r["run"]: r for r in json.load(f)}

    results = []
    for run, corpus, npz_path in RUNS:
        if npz_path is None:
            npz_path = os.path.join(MOON, "runs", run, "Stage_3_deform_fine.npz")
        if not os.path.isfile(npz_path):
            if run in existing_by_run:
                print(f"[keep] {run}: {npz_path} not found here, keeping previously recorded result")
                results.append(existing_by_run[run])
            else:
                print(f"[skip] {run}: {npz_path} not found, and no prior result on record")
            continue
        print(f"[run] {run} (corpus={corpus}) ...")
        r = audit_run(run, corpus, npz_path, template_faces, face_lab, vlabels, n_verts_template, geo_lookup, rng)
        results.append(r)

    print(f"\n{'run':<34}{'leg_acc':>9}{'xleg_err':>10}{'seg_acc':>9}{'ant_side':>10}{'ant_seg':>9}")
    for r in results:
        leg_acc = r["leg_confusion"]["accuracy"]
        seg_acc = r["within_leg_segment_confusion"]["accuracy"]
        side_acc = r["antenna_side_confusion"]["accuracy"]
        antseg_acc = r["antenna_segment_confusion"]["accuracy"]
        print(f"{r['run']:<34}{leg_acc:>9.3f}{1 - leg_acc:>10.3f}{seg_acc:>9.3f}{side_acc:>10.3f}{antseg_acc:>9.3f}")

    print(f"\n{'run':<34}{'pt->ta n':>10}{'pt->ta mean':>13}{'ti->ta n':>10}{'ti->ta mean':>13}{'pt mean(any)':>14}")
    for r in results:
        bp = r["geodesic"]["by_pair"]
        bt = r["geodesic"]["by_true_segment"]
        pt_ta = bp.get("pt->ta", dict(n=0, mean=float("nan")))
        ti_ta = bp.get("ti->ta", dict(n=0, mean=float("nan")))
        pt_any = bt.get("pt", dict(n=0, mean=float("nan")))
        print(
            f"{r['run']:<34}{pt_ta['n']:>10}{pt_ta['mean']:>13.4f}"
            f"{ti_ta['n']:>10}{ti_ta['mean']:>13.4f}{pt_any['mean']:>14.4f}"
        )

    OUT = os.path.join(HERE, "out")
    os.makedirs(OUT, exist_ok=True)
    out_path = os.path.join(OUT, "correspondence_confusion.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
