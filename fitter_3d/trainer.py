"""Introduces Stage class - representing a Stage of optimising a batch of SMBLD meshes to target meshes"""

from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
from pytorch3d.structures import Meshes

from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from fitter_3d.utils import plot_meshes, SDF_distance, sample_points_from_meshes_and_SDF, try_mkdir
import numpy as np
import os
import config
import pickle as pkl

from smal_model.smal_torch import SMAL
from smal_fitter.utils import eul_to_axis
from smal_fitter.priors.joint_limits_prior import _ranges_from_joint_limits
from fitter_3d.joint_limits import scale_barrier, trans_barrier
from fitter_3d.penetration_loss import penetration_loss_batched, _build_part_faces
from fitter_3d.gwn_penetration_loss import precompute_capped_topology, gwn_penetration_loss_batched
from fitter_3d.part_groups import get_part_vertex_indices, get_non_adjacent_pairs, PART_GROUPS_COARSE
from fitter_3d.eval_metrics import f_score
from fitter_3d.eval_metrics_part_precision import per_part_precision
from fitter_3d.local_smoothness import (
    build_vertex_adjacency,
    khop_expand_mask,
    random_touched_mask,
    weighted_edge_loss,
    weighted_laplacian_loss,
)
import csv

nn = torch.nn

default_weights = dict(
    w_chamfer=1.0, w_edge=1.0, w_normal=0.01, w_laplacian=0.1, w_sdf=0.5, w_limit=0.0,
    w_penetration=0.0,  # off by default; set >0 per-stage in yaml (deform stages only -- see
    # ants_cfg_penetration_gentle.yaml)
    # penetration_loss_batched hyperparameters -- these defaults are numerically a
    # no-op when left unset in a stage's yaml loss_weights.
    penetration_tau_fraction=0.03,
    penetration_max_depth_fraction=0.08,
    penetration_ramp_iters=200,
    w_penetration_gwn=0.0,  # off by default -- see fitter_3d/gwn_penetration_loss.py. Additive
    # alternative to w_penetration, not a replacement: both can be active at once (not
    # recommended together untested), or w_penetration_gwn alone to test the GWN-based
    # signal in isolation. Only has an effect if Stage was constructed with
    # gwn_penetration_pairs set (see Stage.__init__) -- otherwise silently a no-op even
    # if this weight is > 0, since there's no capped topology to evaluate it against.
    penetration_gwn_ramp_iters=0,  # 0 = no ramp (matches the first, miscalibrated run's
    # behavior -- flagged as a real gap, not an oversight, in the post-mortem of that run).
    # Set >0 per-stage in yaml for the same reason penetration_ramp_iters exists: an
    # undertrained pose can be badly self-intersecting purely from initialisation.
    w_penetration_bvh=0.0,  # off by default -- see fitter_3d/bvh_penetration_loss.py. A
    # THIRD, additive alternative to w_penetration/w_penetration_gwn: exact BVH triangle
    # collision detection + SMPLify-X's conical distance field push, the field-standard
    # approach, reused (not reimplemented) from a confirmed-working compiled extension.
    # Only has an effect if Stage was constructed with bvh_penetration_pairs set (see
    # Stage.__init__) -- otherwise silently a no-op even if this weight is > 0, same
    # convention as w_penetration_gwn/gwn_penetration_pairs.
    bvh_penetration_ramp_iters=0,  # 0 = no ramp -- same rationale as penetration_ramp_iters
    # and penetration_gwn_ramp_iters: set >0 per-stage in yaml if an undertrained pose's
    # initial self-intersection would otherwise dominate the loss from iteration 0.
    w_offset=0.0,  # off by default; L2 penalty on ||deform_verts||, ported from
    # trainer_moonshot.py (its only prior home) so it can run alongside this trainer's
    # penetration/limit machinery -- see the penetration-joint-study TASK 2 structural-
    # constraint arm. D1 setting validated on the moonshot pipeline: 5.0 (coarse stage) /
    # 2.0 (fine stage) -- diagnostics/moonshot/cfg/D1_low.yaml.
    w_scale=0.0,  # off by default; scale_barrier (fitter_3d/joint_limits.py) on
    # log_beta_scales, same name/semantics as trainer_moonshot.py's w_scale so a weight
    # tuned there means the same here. Calibrated value 0.052 (T0.4, full-corpus A/B,
    # diagnostics/EXECUTION_PLAN.md sec 5) -- validated on the moonshot/hierarchical
    # genus-classification pipeline, never before run on this (fitter_3d/trainer.py)
    # penetration-study pipeline.
    w_trans=0.0,  # off by default; trans_barrier on betas_trans, same name/semantics as
    # trainer_moonshot.py's w_trans. UNCALIBRATED -- no validated value exists anywhere in
    # the branch (diagnostics/SHIPPED_RECIPE.md, diagnostics/INVENTORY_V2.md T0.4b): the
    # only value ever tried (1.0) left the barrier loss ~0 throughout, not meaningfully
    # engaged under D1's other regularizers. Deprioritized, not shipped -- included here
    # for completeness, not as a validated lever.
)  # Added SDF distance weight; w_limit off by default (issue #97)
# Want to vary learning ratios between parameters,
default_lr_ratios = []


def get_meshes(verts, faces, device="cuda"):
    """Returns Meshes object of all SMAL meshes."""
    meshes = Meshes(verts=verts, faces=faces).to(device)
    return meshes


def _joint_limit_tensors_from_dd(dd, device):
    """Build (min_limits, max_limits, error) tensors of shape (N_POSE, 3) from an
    already-loaded SMAL .pkl dict. `error` is the deferred ValueError/TypeError (or
    None) if 'joint_limits' was present but malformed - validation is never raised
    here, only reported, so construction never crashes a consumer that doesn't use
    w_limit."""

    def _safe_ranges(dd):
        try:
            return _ranges_from_joint_limits(dd), None
        except (ValueError, TypeError) as e:
            # Fall back to wide-open ranges. Build the fallback from a *copy*
            # of dd with 'joint_limits' stripped - dd itself is never mutated.
            dd_no_limits = {k: v for k, v in dd.items() if k != "joint_limits"}
            return _ranges_from_joint_limits(dd_no_limits), e

    ranges, error = _safe_ranges(dd)
    if error is not None:
        print(
            f"WARNING: invalid 'joint_limits' in the model file: {error} "
            "Using wide-open ranges; the limit loss (w_limit > 0) will raise."
        )

    joint_names = dd["J_names"]
    n_pose = len(joint_names) - 1  # not including the root/global rotation
    min_values = np.array([ranges[j][ax][0] for j in joint_names for ax in range(3)])
    max_values = np.array([ranges[j][ax][1] for j in joint_names for ax in range(3)])

    # exclude root joint (its 3 values come first), reshape to match joint_rot
    min_limits = torch.FloatTensor(min_values[3:]).view(n_pose, 3).to(device)
    max_limits = torch.FloatTensor(max_values[3:]).view(n_pose, 3).to(device)
    return min_limits, max_limits, error


class SMAL3DFitter(nn.Module):
    def __init__(self, batch_size=1, device="cuda", shape_family=-1):
        super(SMAL3DFitter, self).__init__()

        self.device = device
        self.batch_size = batch_size
        self.n_betas = config.N_BETAS

        with open(config.SMAL_FILE, "rb") as f:
            u = pkl._Unpickler(f)
            u.encoding = "latin1"
            dd = u.load()

        if config.ignore_hardcoded_body:
            try:
                model_covs = dd["shape_cov"]
                self.mean_betas = torch.FloatTensor(dd["shape_mean_betas"]).to(device)
            except KeyError:
                print(
                    "No shape_cov or shape_mean_betas found in SMAL_FILE; "
                    "falling back to identity-covariance prior sized to config.N_BETAS"
                )
                self.mean_betas = torch.zeros(config.N_BETAS).to(device)
                model_covs = np.eye(config.N_BETAS)

        else:
            with open(config.SMAL_DATA_FILE, "rb") as f:
                u = pkl._Unpickler(f)
                u.encoding = "latin1"
                smal_data = u.load()

                self.shape_family_list = np.array(shape_family)

                model_covs = np.array(smal_data["cluster_cov"])[[shape_family]][0]
                self.mean_betas = torch.FloatTensor(smal_data["cluster_means"][[shape_family]][0])[: config.N_BETAS].to(
                    device
                )

        if config.DEBUG:
            print("MODEL COVS", model_covs)

        invcov = np.linalg.inv(model_covs + 1e-5 * np.eye(model_covs.shape[0]))
        prec = np.linalg.cholesky(invcov)

        self.betas_prec = torch.FloatTensor(prec)[: config.N_BETAS, : config.N_BETAS].to(device)

        if config.DEBUG:
            print("MEAN BETAS", self.mean_betas)

        self.betas = nn.Parameter(self.mean_betas.unsqueeze(0).repeat(batch_size, 1))

        # Reuses the dd loaded above rather than re-opening config.SMAL_FILE.
        self.kintree_table = torch.tensor(dd["kintree_table"]).to(device)

        # Get number of joints from kintree
        self.n_joints = self.kintree_table.shape[1]

        # Initialize log_beta_scales with proper shape: batch_size x n_joints x 3
        # Starting with ones means no scaling initially (since exp(0) = 1)
        self.log_beta_scales = nn.Parameter(torch.zeros(self.batch_size, self.n_joints, 3).to(device))

        # Initialize betas_trans with proper shape: batch_size x n_joints x 3
        # Starting with zeros means no translation offsets initially
        self.betas_trans = nn.Parameter(torch.zeros(self.batch_size, self.n_joints, 3).to(device))

        global_rotation_np = eul_to_axis(np.array([0, 0, 0]))
        global_rotation = (
            torch.from_numpy(global_rotation_np).float().to(device).unsqueeze(0).repeat(batch_size, 1)
        )  # Global Init (Head-On)
        self.global_rot = nn.Parameter(global_rotation)

        trans = torch.FloatTensor([0.0, 0.0, 0.0])[None, :].to(device).repeat(batch_size, 1)  # Trans Init
        self.trans = nn.Parameter(trans)

        default_joints = torch.zeros(batch_size, config.N_POSE, 3).to(device)
        self.joint_rot = nn.Parameter(default_joints)

        # Joint rotation limits (issue #97): read authored 'joint_limits' from the
        # model .pkl and build a hinge-loss prior, mirroring smal_fitter/fitter.py's
        # SMALFitter.__init__. Built from the local `dd` loaded above (not
        # LimitPrior()/config.dd) so this doesn't depend on config.dd staying in sync
        # with config.SMAL_FILE for this class.
        #
        # A malformed 'joint_limits' in the model file must not take down every run
        # that constructs a SMAL3DFitter (this class is built unconditionally, before
        # any stage's loss_weights are even looked at - see optimise.py). So we defer
        # the error to the first forward pass with w_limit > 0 (Stage.forward()) and
        # fall back to wide-open ranges meanwhile.
        if config.ignore_hardcoded_body:
            self.min_limits, self.max_limits, self._joint_limits_error = _joint_limit_tensors_from_dd(dd, device)
        else:
            # Legacy hardcoded-body mode: joint_limits/N_POSE-per-name mapping doesn't
            # apply here. Keep the attributes present (None) so Stage.forward()'s
            # _joint_limits_error check never hits AttributeError; max_limits/
            # min_limits are also None, so w_limit > 0 in this mode raises a clear
            # error at first use rather than silently doing nothing.
            self._joint_limits_error = None
            self.max_limits = None
            self.min_limits = None

        # Use this to restrict global rotation if necessary
        self.global_mask = torch.ones(1, 3).to(device)
        # self.global_mask[:2] = 0.0

        # Can be used to prevent certain joints rotating.
        # Can be useful depending on sequence.
        self.rotation_mask = torch.ones(config.N_POSE, 3).to(device)  # by default all joints are free to rotate
        # self.rotation_mask[25:32] = 0.0 # e.g. stop the tail moving

        # setup SMAL skinning & differentiable renderer
        self.smal_model = SMAL(device, shape_family_id=shape_family)
        self.faces = self.smal_model.faces.unsqueeze(0).repeat(batch_size, 1, 1)

        """
        # vertex offsets for deformations
        self.deform_verts = nn.Parameter(torch.zeros(batch_size, *self.smal_model.v_template.shape)).to(device)
        """

        # Initialize deform_verts as a leaf tensor
        self.deform_verts = nn.Parameter(
            torch.zeros(batch_size, *self.smal_model.v_template.shape).to(device), requires_grad=True
        )

    def get_joint_scales(self, log_beta_scales_arg=None):
        """
        Compute the final scale for each joint taking into account the kinematic chain.
        A joint's scale is influenced by its own scale parameters and all its parents.

        Args:
            log_beta_scales_arg (optional): Tensor of shape (batch_size, n_joints, 3)
                                            to use instead of self.log_beta_scales.
        """
        current_log_beta_scales = log_beta_scales_arg if log_beta_scales_arg is not None else self.log_beta_scales
        # Start with base scales from log_beta_scales
        joint_scales = torch.exp(current_log_beta_scales)  # Convert from log space

        # For each joint
        for joint_idx in range(self.n_joints):
            parent_idx = self.kintree_table[0, joint_idx]

            # If this joint has a parent (parent_idx != joint_idx)
            # and the parent_idx is valid (within bounds of joint_scales)
            if parent_idx != joint_idx and parent_idx < joint_scales.shape[1]:
                # Accumulate parent's scale
                joint_scales[:, joint_idx] = joint_scales[:, joint_idx] * joint_scales[:, parent_idx]

        return joint_scales

    def forward(
        self,
        betas=None,
        global_rot=None,
        joint_rot=None,
        trans=None,
        log_beta_scales=None,
        betas_trans=None,
        deform_verts=None,
        return_joints=False,
    ):
        """
        Forward pass for the SMAL model.
        Can accept optional parameters to override the internal nn.Parameter attributes.

        Args:
            betas (optional): Shape parameters, tensor of shape (batch_size, N_BETAS).
            global_rot (optional): Global rotation in axis-angle, tensor of shape (batch_size, 3).
            joint_rot (optional): Joint rotations in axis-angle, tensor of shape (batch_size, N_POSE, 3).
            trans (optional): Global translation, tensor of shape (batch_size, 3).
            log_beta_scales (optional): Logarithm of joint scales, tensor of shape (batch_size, n_joints, 3).
            betas_trans (optional): Joint translation offsets, tensor of shape (batch_size, n_joints, 3).
            deform_verts (optional): Vertex offsets, tensor of shape (batch_size, n_template_verts, 3).
            return_joints (bool): Whether to return joints.

        Returns:
            verts: Predicted vertices, tensor of shape (batch_size, n_verts, 3).
            joints: Predicted joints, tensor of shape (batch_size, n_joints, 3), if return_joints is True.
        """

        # Determine which parameters to use (passed argument or self.attribute)
        _betas = betas if betas is not None else self.betas
        _global_rot = global_rot if global_rot is not None else self.global_rot
        _joint_rot = joint_rot if joint_rot is not None else self.joint_rot
        _trans = trans if trans is not None else self.trans
        # Use provided log_beta_scales if available, otherwise use the internal one.
        # This will be passed to get_joint_scales and directly to smal_model if needed.
        _log_beta_scales_to_use = log_beta_scales if log_beta_scales is not None else self.log_beta_scales
        _betas_trans = betas_trans if betas_trans is not None else self.betas_trans
        _deform_verts = deform_verts if deform_verts is not None else self.deform_verts

        # The original get_joint_scales uses self.log_beta_scales.
        # We don't need joint_scales for the smal_model call directly,
        # as smal_model itself handles the betas_logscale.
        # The get_joint_scales method itself is not directly used in the smal_model call here,
        # but it's good practice to make it consistent if it were to be used externally
        # or if smal_model's interface changes.
        # For the current self.smal_model call, we pass _log_beta_scales_to_use directly.

        # The SMAL model expects betas_logscale to be of shape (batch_size, n_joints, 3).
        # self.log_beta_scales is already initialized with this shape.
        # If log_beta_scales is passed as an argument, it should also conform to this shape.
        # The reshape operation in the original code was:
        # `betas_logscale = self.log_beta_scales.reshape(self.batch_size, -1, 3)`
        # This is effectively a no-op if self.log_beta_scales is already (bs, n_joints, 3).

        verts, joints, Rs, v_shaped = self.smal_model(
            _betas,
            torch.cat(
                [
                    _global_rot.unsqueeze(1),  # _global_rot is (bs, 3) -> (bs, 1, 3)
                    _joint_rot,
                ],
                dim=1,
            ),  # _joint_rot is (bs, N_POSE, 3)
            betas_logscale=_log_beta_scales_to_use,  # Pass the determined log_beta_scales
            betas_trans=_betas_trans,
        )  # Pass the joint translation offsets

        verts = verts + _trans.unsqueeze(1)  # _trans is (bs, 3) -> (bs, 1, 3)
        joints = joints + _trans.unsqueeze(1)

        verts = verts + _deform_verts  # _deform_verts is (bs, n_template_verts, 3)

        if return_joints:
            return verts, joints
        else:
            return verts


class SMALParamGroup:
    """Object building on model.parameters, with modifications such as variable learning rate"""

    param_map = {
        "init": ["global_rot", "trans"],
        "init_rot_lock": ["trans", "log_beta_scales"],
        "init_rot_lock_trans": ["trans", "betas_trans"],
        "init_rot_lock_trans_scale": ["trans", "betas_trans", "log_beta_scales"],
        "default": ["global_rot", "joint_rot", "trans", "betas", "log_beta_scales"],
        "default_with_betas_trans": ["global_rot", "joint_rot", "trans", "betas", "log_beta_scales", "betas_trans"],
        "shape": ["global_rot", "trans", "betas", "log_beta_scales", "betas_trans"],
        "pose": ["global_rot", "trans", "joint_rot", "betas", "log_beta_scales", "betas_trans"],
        "deform": ["deform_verts"],
        "all": ["global_rot", "trans", "joint_rot", "betas", "log_beta_scales", "betas_trans", "deform_verts"],
    }  # map of param_type : all attributes in SMAL used in optim

    def __init__(self, model, group="smbld", lrs=None):
        """
        :param lrs: dict of param_name : custom learning rate
        """

        self.model = model

        self.group = group
        assert group in self.param_map, f"Group {group} not in list of available params: {list(self.param_map.keys())}"

        self.lrs = {}
        if lrs is not None:
            for k, lr in lrs.items():
                self.lrs[k] = lr

    def __iter__(self):
        """Return iterable list of all parameters"""
        out = []

        for param_name in self.param_map[self.group]:
            param = [getattr(self.model, param_name)]
            d = {"params": param}
            if param_name in self.lrs:
                d["lr"] = self.lrs[param_name]

            out.append(d)

        return iter(out)


class Stage:
    """Defines a stage of optimisation, the optimisation parameters for the stage, ..."""

    def __init__(
        self,
        nits: int,
        scheme: str,
        smal_3d_fitter: SMAL3DFitter,
        target_meshes: Meshes,
        mesh_names=[],
        name="optimise",
        loss_weights=None,
        lr=1e-3,
        out_dir="static_fits_output",
        custom_lrs=None,
        device="cuda",
        plot_normals=False,
        sample_size=1000,
        sdf_values=None,
        source_sdf_values=None,
        visualize_sdf_loss=False,
        sdf_vis_frequency=10,
        local_downweight=None,
        gwn_penetration_pairs=None,
        penetration_train_pairs=None,
        symmetric_chamfer_sampling=False,
        bvh_penetration_pairs=None,
        bvh_max_collisions=8,
    ):
        """
        nits = integer, number of iterations in stage
        parameters = list of items over which to be optimised
        get_mesh = function that returns Mesh object for identifying losses
        name = name of stage
        visualize_sdf_loss = whether to visualize SDF loss contribution
        sdf_vis_frequency = how often to visualize SDF loss (every N iterations)
        local_downweight = optional dict enabling the neighbor-drag diagnostic
            experiment (fitter_3d/local_smoothness.py). None (default) leaves
            edge/laplacian loss exactly as before -- every existing yaml config
            is unaffected. When set:
                "mode": "local" or "global"
                "k": int, hop radius for "local" mode (primary k; see also
                    "log_extra_k" below)
                "factor": float in [0, 1], down-weight multiplier applied to
                    touched vertices/edges (1.0 = untouched, e.g. 0.1 = 90%
                    reduction)
                "log_extra_k": list of additional k values to LOG coverage for
                    (not used for the actual down-weighting, "local" mode only)
                "global_schedule": required for mode="global" -- (n_iterations, B)
                    int array/tensor of the exact n_touched vertex COUNT to
                    replicate each iteration, normally the recorded counts from
                    a prior "local" run on the SAME specimens in the SAME
                    order. "global" mode draws a FRESH random subset of that
                    exact size every iteration and applies "factor" there (rest
                    at 1.0) -- matched to "local" on count and per-vertex
                    weight, differing only in WHERE the down-weighted vertices
                    sit (scattered vs. clustered around penetration). Also
                    reads "factor" (same key as "local" mode).
                "seed": int, default 0 -- seeds the "global" mode random draws
                    for reproducibility.
        gwn_penetration_pairs = optional list of (part_a, part_b) tuples enabling
            fitter_3d/gwn_penetration_loss.py's differentiable winding-number-based
            penetration test (see Step 6 of scripts/penetration_joint_study/FINDINGS.md
            for why: the proximity test's inside/outside verdict was confirmed to
            disagree with a true GWN signal specifically in whichever direction of a
            pair fails to improve under training). None (default) does no capped-
            topology precompute and w_penetration_gwn is silently a no-op even if
            set > 0. Deliberately an explicit pair list, not "every non-adjacent
            pair" -- this is a targeted test of the one pair already validated
            (gaster, legs), not a full rollout.
        penetration_train_pairs = optional list of (part_a, part_b) tuples
            restricting which pairs the w_penetration LOSS (the proximity
            test, not the GWN one above) actually trains on. None (default)
            trains on every anatomically non-adjacent pair, unchanged from
            prior behaviour. Evaluation/diagnostics (compute_eval_metrics's
            per-pair CSV) are UNAFFECTED by this and always cover the full
            pair set regardless -- this only narrows what the optimizer sees
            a gradient for, so collateral effects on untrained pairs remain
            visible for scoring.
        bvh_penetration_pairs = optional list of (part_a, part_b) tuples enabling
            fitter_3d/bvh_penetration_loss.py's exact-BVH-collision + conical-distance-
            field penetration signal (SMPLify-X's own approach, reused from a confirmed-
            working compiled extension -- see that module's docstring for provenance and
            why this is a real, not-reimplemented, use of the field-standard method).
            None (default) does no face-part-id precompute and w_penetration_bvh is
            silently a no-op even if set > 0. Requires `mesh_intersection` (the compiled
            bvh_cuda extension) importable -- NOT on the default Python path, since it's
            a local build under custom_processing/external/, not an installed package;
            raises ImportError with a clear message if bvh_penetration_pairs is set but
            it isn't importable, rather than failing confusingly deeper in forward().
        bvh_max_collisions = int, default 8. Passed straight through to
            mesh_intersection.bvh_search_tree.BVH's own max_collisions -- the maximum
            number of colliding faces recorded per query face before older ones are
            dropped. Only relevant if bvh_penetration_pairs is set.
        symmetric_chamfer_sampling = bool, default False (unchanged prior behaviour).
            The chamfer term compares an area-weighted 3000-point sample of the
            TARGET against the SOURCE's raw, unsampled template vertices
            (src_mesh.verts_padded(), ~10229 of them) -- an asymmetry, since raw
            vertex density varies non-uniformly across the template (legs are
            2.36x denser than gaster per scripts/penetration_joint_study/FINDINGS.md
            step 3), so a densely-meshed part gets pulled toward the target harder
            than an equally-sized sparse part gets. fitter_3d/trainer_moonshot.py
            already fixes this (area-weighted-samples both sides, unconditionally).
            When True, ports that fix here: the SOURCE is also sampled via
            sample_points_from_meshes (same 3000-point budget as the target, so
            this changes only WHICH points are compared, not how many). False
            (default) is byte-identical to prior behaviour.

        lr_decay = factor by which lr decreases at each it"""

        self.n_it = nits
        self.name = name
        self.out_dir = out_dir
        self.target_meshes = target_meshes
        self.mesh_names = mesh_names
        self.smal_3d_fitter = smal_3d_fitter
        self.device = device
        self.plot_normals = plot_normals
        self.loss_weights = default_weights.copy()
        if loss_weights is not None:
            for k, v in loss_weights.items():
                self.loss_weights[k] = v

        # Parameter for vertex sampling
        self.sample_size = sample_size

        # Store SDF values if provided
        self.sdf_values = sdf_values
        self.source_sdf_values = source_sdf_values

        # SDF loss visualization parameters
        self.visualize_sdf_loss = visualize_sdf_loss
        self.sdf_vis_frequency = sdf_vis_frequency

        self.losses_to_plot = []  # Store losses for review later

        if custom_lrs is not None:
            for attr in custom_lrs:
                assert hasattr(smal_3d_fitter, attr), f"attr '{attr}' not in SMAL."

        self.param_group = SMALParamGroup(smal_3d_fitter, scheme, custom_lrs)

        self.scheduler = None

        self.optimizer = torch.optim.Adam(self.param_group, lr=lr)
        self.src_verts = smal_3d_fitter().detach()  # original verts, detach from autograd
        self.faces = smal_3d_fitter.faces.detach()
        self.src_mesh = get_meshes(self.src_verts, self.faces, device=device)
        self.n_verts = self.src_verts.shape[1]

        # Precomputed once for the penetration loss: template topology (both
        # vertex-to-part labels and part-to-part face subsets) never changes
        # between iterations or specimens, only vertex positions do.
        self.part_vertex_indices = get_part_vertex_indices(PART_GROUPS_COARSE)
        self.non_adjacent_pairs = get_non_adjacent_pairs(PART_GROUPS_COARSE)
        faces_np = self.faces[0].cpu().numpy()  # shared template topology, same for every specimen
        self.part_faces = _build_part_faces(faces_np, self.part_vertex_indices)
        # Pairs the w_penetration LOSS trains on -- defaults to the full
        # self.non_adjacent_pairs (unchanged prior behaviour). compute_eval_metrics
        # always scores the full pair set regardless (see its docstring), so
        # narrowing this is purely a training-scope change, not a scoring change.
        self.penetration_train_pairs = (
            penetration_train_pairs if penetration_train_pairs is not None else self.non_adjacent_pairs
        )

        # Neighbor-drag diagnostic experiment (see local_smoothness.py docstring).
        # None (the default) means edge/laplacian below behave exactly as before.
        self.local_downweight = local_downweight
        self.local_downweight_log = []  # one row per (iteration, specimen), coverage fractions at every logged k
        self._current_penetrating_vertex_mask = None  # (B, n_verts) bool, refreshed each forward() call
        if self.local_downweight is not None:
            self.vertex_adjacency = build_vertex_adjacency(faces_np, self.n_verts, device)
            # Seeded so the "global" control's per-iteration random subset draws are
            # reproducible across reruns -- explicit, not left to whatever the ambient
            # torch RNG state happened to be.
            self._local_downweight_rng = torch.Generator(device=device).manual_seed(
                self.local_downweight.get("seed", 0)
            )

        # GWN-based penetration test (see fitter_3d/gwn_penetration_loss.py). Capped
        # topology (boundary loops + oriented cap faces) is derived ONCE from the
        # REST-POSE template -- fixed for all specimens/iterations, since SMIL's
        # topology never changes under deformation (see Step 6 of FINDINGS.md).
        self.gwn_penetration_pairs = gwn_penetration_pairs
        if self.gwn_penetration_pairs is not None:
            v_template = smal_3d_fitter.smal_model.v_template.detach().cpu().numpy()
            self.gwn_capped_topology = precompute_capped_topology(v_template, self.part_faces)

        # BVH-based penetration signal (see fitter_3d/bvh_penetration_loss.py). Imported
        # lazily, only when actually requested -- mesh_intersection (the compiled bvh_cuda
        # extension) is a local build under custom_processing/external/, not on the default
        # Python path or an installed package, so every other config must keep working
        # without it importable.
        self.bvh_penetration_pairs = bvh_penetration_pairs
        if self.bvh_penetration_pairs is not None:
            try:
                from mesh_intersection.bvh_search_tree import BVH
                from mesh_intersection.loss import DistanceFieldPenetrationLoss
                from fitter_3d.bvh_penetration_loss import build_face_part_ids, bvh_penetration_loss_batched
            except ImportError as exc:
                raise ImportError(
                    "bvh_penetration_pairs was set but `mesh_intersection` (the compiled "
                    "bvh_cuda extension) is not importable. It's a local build, not an "
                    "installed package -- add its directory to PYTHONPATH/sys.path before "
                    "constructing this Stage (see fitter_3d/bvh_penetration_loss.py's "
                    "docstring for which fork/build is confirmed working)."
                ) from exc
            self.bvh_face_part_id_np, self.bvh_part_name_to_idx = build_face_part_ids(
                faces_np, self.part_vertex_indices
            )
            self.bvh_face_part_id = torch.as_tensor(
                self.bvh_face_part_id_np, dtype=torch.long, device=device
            )
            self.bvh_faces = torch.as_tensor(faces_np, dtype=torch.long, device=device)
            self.bvh_module = BVH(max_collisions=bvh_max_collisions)
            self.dfp_loss_module = DistanceFieldPenetrationLoss()
            # Stashed as an instance attribute, not imported at module level in this file --
            # fitter_3d.bvh_penetration_loss itself imports mesh_intersection at import time,
            # so importing it unconditionally here would force the same hard dependency on
            # every config, exactly what the lazy try/except above exists to avoid.
            self._bvh_penetration_loss_batched = bvh_penetration_loss_batched

        self.consider_loss = lambda loss_name: (
            self.loss_weights[f"w_{loss_name}"] > 0
        )  # function to check if loss is non-zero

        # See symmetric_chamfer_sampling's docstring above. False (default) leaves
        # the chamfer term exactly as before -- every existing config unaffected.
        self.symmetric_chamfer_sampling = symmetric_chamfer_sampling

    def forward(self, src_mesh, iteration=0):
        loss = 0
        loss_components = {}

        # Sample from target meshes
        target_verts = sample_points_from_meshes(self.target_meshes, 3000)

        if self.consider_loss("chamfer"):
            if self.symmetric_chamfer_sampling:
                src_verts_for_chamfer = sample_points_from_meshes(src_mesh, 3000)
            else:
                src_verts_for_chamfer = src_mesh.verts_padded()
            loss_chamfer, _ = chamfer_distance(target_verts, src_verts_for_chamfer)
            loss_components["chamfer"] = loss_chamfer
            loss += self.loss_weights["w_chamfer"] * loss_chamfer

        # Penetration is computed here -- ahead of its old position further down --
        # whenever local_downweight needs this iteration's penetrating-vertex mask to
        # build the edge/laplacian down-weighting below. Loss accumulation via `loss +=`
        # is order-independent, so this changes nothing for every existing config where
        # local_downweight is None (the mask is simply never requested/used).
        if self.consider_loss("penetration"):
            need_mask = self.local_downweight is not None
            penetration_out = penetration_loss_batched(
                verts_padded=src_mesh.verts_padded(),
                part_vertex_indices=self.part_vertex_indices,
                part_faces=self.part_faces,
                non_adjacent_pairs=self.penetration_train_pairs,
                proximity_tau_fraction=self.loss_weights.get("penetration_tau_fraction", 0.03),
                max_depth_fraction=self.loss_weights.get("penetration_max_depth_fraction", 0.08),
                iteration=iteration,
                n_ramp_iters=self.loss_weights.get("penetration_ramp_iters", 200),
                return_diagnostics=need_mask,
            )
            if need_mask:
                loss_penetration_per_specimen, penetration_diag = penetration_out
                self._current_penetrating_vertex_mask = penetration_diag["penetrating_vertex_mask"]
            else:
                loss_penetration_per_specimen = penetration_out
            loss_penetration = loss_penetration_per_specimen.mean()
            loss_components["penetration"] = loss_penetration
            loss += self.loss_weights["w_penetration"] * loss_penetration
        else:
            self._current_penetrating_vertex_mask = None

        if self.gwn_penetration_pairs is not None and self.consider_loss("penetration_gwn"):
            loss_penetration_gwn_per_specimen = gwn_penetration_loss_batched(
                verts_padded=src_mesh.verts_padded(),
                part_vertex_indices=self.part_vertex_indices,
                capped_topology=self.gwn_capped_topology,
                pairs=self.gwn_penetration_pairs,
                iteration=iteration,
                n_ramp_iters=self.loss_weights.get("penetration_gwn_ramp_iters", 0),
                return_diagnostics=False,
            )
            loss_penetration_gwn = loss_penetration_gwn_per_specimen.mean()
            loss_components["penetration_gwn"] = loss_penetration_gwn
            loss += self.loss_weights["w_penetration_gwn"] * loss_penetration_gwn

        if self.bvh_penetration_pairs is not None and self.consider_loss("penetration_bvh"):
            loss_penetration_bvh_per_specimen = self._bvh_penetration_loss_batched(
                verts_padded=src_mesh.verts_padded(),
                faces=self.bvh_faces,
                face_part_id=self.bvh_face_part_id,
                part_name_to_idx=self.bvh_part_name_to_idx,
                train_pairs=self.bvh_penetration_pairs,
                dfp_loss_module=self.dfp_loss_module,
                bvh_module=self.bvh_module,
                iteration=iteration,
                n_ramp_iters=self.loss_weights.get("bvh_penetration_ramp_iters", 0),
                return_diagnostics=False,
            )
            loss_penetration_bvh = loss_penetration_bvh_per_specimen.mean()
            loss_components["penetration_bvh"] = loss_penetration_bvh
            loss += self.loss_weights["w_penetration_bvh"] * loss_penetration_bvh

        vertex_weight = None
        if self.local_downweight is not None:
            vertex_weight = self._compute_local_downweight_vertex_weight(iteration)

        if self.consider_loss("edge"):
            if vertex_weight is not None:
                loss_edge = weighted_edge_loss(src_mesh, vertex_weight)
            else:
                loss_edge = mesh_edge_loss(src_mesh)  # and (b) the edge length of the predicted mesh
            loss_components["edge"] = loss_edge
            loss += self.loss_weights["w_edge"] * loss_edge

        if self.consider_loss("normal"):
            loss_normal = mesh_normal_consistency(src_mesh)  # mesh normal consistency
            loss_components["normal"] = loss_normal
            loss += self.loss_weights["w_normal"] * loss_normal

        if self.consider_loss("laplacian"):
            if vertex_weight is not None:
                loss_laplacian = weighted_laplacian_loss(src_mesh, vertex_weight, self.vertex_adjacency)
            else:
                loss_laplacian = mesh_laplacian_smoothing(src_mesh, method="uniform")  # mesh laplacian smoothing
            loss_components["laplacian"] = loss_laplacian
            loss += self.loss_weights["w_laplacian"] * loss_laplacian

        # Add SDF distance loss if SDF values are provided
        if self.consider_loss("sdf") and self.sdf_values is not None and self.source_sdf_values is not None:
            # Sample points from source mesh for SDF calculation
            src_verts, src_sdf = sample_points_from_meshes_and_SDF(src_mesh, self.source_sdf_values, 10000)
            target_verts, target_sdf = sample_points_from_meshes_and_SDF(self.target_meshes, self.sdf_values, 10000)

            # Determine if we should visualize on this iteration
            visualize_now = self.visualize_sdf_loss and (
                iteration % self.sdf_vis_frequency == 0 or iteration == self.n_it - 1
            )

            # Create visualization directory if needed
            if visualize_now:
                vis_dir = os.path.join(self.out_dir, "sdf_visualization", self.name)
                try_mkdir(vis_dir)
            else:
                vis_dir = "sdf_visualization"

            # Calculate SDF distance
            loss_sdf = SDF_distance(
                src_verts,  # source points
                target_verts,  # target points
                src_sdf,  # source SDF values
                target_sdf,  # target SDF values
                k=50,  # number of nearest neighbors
                batch_reduction="mean",
                point_reduction="mean",
                norm=2,
                single_directional=False,
                visualize=visualize_now,
                output_dir=vis_dir,
                title=f"{self.name}_iteration{iteration}",
                mesh_names=self.mesh_names,  # Pass mesh names for better file naming
            )
            loss_components["sdf"] = loss_sdf
            loss += self.loss_weights["w_sdf"] * loss_sdf

        if self.consider_loss("limit"):
            if self.smal_3d_fitter._joint_limits_error is not None:
                raise ValueError(
                    "Limit loss is enabled (w_limit > 0) but the model's 'joint_limits' "
                    f"is invalid: {self.smal_3d_fitter._joint_limits_error}"
                )
            if self.smal_3d_fitter.max_limits is None or self.smal_3d_fitter.min_limits is None:
                raise ValueError(
                    "Limit loss is enabled (w_limit > 0) but no joint limits are available "
                    "for this model (legacy hardcoded-body mode does not support joint_limits)."
                )
            joint_rot = self.smal_3d_fitter.joint_rot
            max_limits = self.smal_3d_fitter.max_limits
            min_limits = self.smal_3d_fitter.min_limits
            zeros = torch.zeros_like(joint_rot)
            loss_limit = torch.mean(torch.max(joint_rot - max_limits, zeros) + torch.max(min_limits - joint_rot, zeros))
            loss_components["limit"] = loss_limit
            loss += self.loss_weights["w_limit"] * loss_limit

        if self.consider_loss("offset"):
            # L2 on the free-form vertex offsets -- ported verbatim from
            # trainer_moonshot.py:488-491 (the only place this was validated: E8,
            # correctness 4.81% -> 6.69% on the ground-truth corpus at 5.0/2.0).
            loss_offset = self.smal_3d_fitter.deform_verts.pow(2).sum(-1).mean()
            loss_components["offset"] = loss_offset
            loss += self.loss_weights["w_offset"] * loss_offset

        if self.consider_loss("scale"):
            # ported verbatim from trainer_moonshot.py:513-516 -- same call, same
            # raw (uncomposed) log_beta_scales, so a weight tuned there means the
            # same here (see w_scale's default_weights comment for the calibrated value).
            loss_scale = scale_barrier(self.smal_3d_fitter.log_beta_scales)
            loss_components["scale_barrier"] = loss_scale
            loss += self.loss_weights["w_scale"] * loss_scale

        if self.consider_loss("trans"):
            # ported verbatim from trainer_moonshot.py:518-521 -- see w_trans's
            # default_weights comment: uncalibrated, included for completeness.
            loss_trans = trans_barrier(self.smal_3d_fitter.betas_trans)
            loss_components["trans_barrier"] = loss_trans
            loss += self.loss_weights["w_trans"] * loss_trans

        return loss, loss_components

    def _compute_local_downweight_vertex_weight(self, iteration):
        """
        Returns (B, n_verts) float32 vertex_weight for the current iteration,
        per self.local_downweight["mode"]:
          "local":  1.0 (untouched) or "factor" (touched -- within k mesh-hops
                    of a currently-penetrating vertex, from
                    self._current_penetrating_vertex_mask). Also logs coverage
                    at k and every self.local_downweight["log_extra_k"] value.
          "global": the count-matched, spatially-scattered control. A FRESH
                    random subset of vertices is drawn EVERY iteration (not a
                    fixed subset chosen once) -- same size, same "factor"
                    weight, same 1.0 elsewhere as "local" mode used AT THIS
                    EXACT ITERATION (from self.local_downweight["global_schedule"],
                    normally the n_touched counts recorded from a prior "local"
                    run on the same specimens). This isolates spatial
                    clustering vs. scattering while holding count and per-
                    vertex weight identical -- deliberately at the cost of
                    giving the control NO temporal persistence at any one
                    location (a fixed-once random subset would instead test a
                    different question: consistently-scattered vs.
                    consistently-local).
        """
        cfg = self.local_downweight
        mode = cfg["mode"]
        device = self.device
        B = self.src_verts.shape[0]
        V = self.n_verts

        if mode == "global":
            counts = cfg["global_schedule"][iteration]  # (B,) int -- n_touched to replicate, per specimen
            factor = cfg["factor"]
            masks = [
                random_touched_mask(int(counts[b]), V, device, generator=self._local_downweight_rng)
                for b in range(B)
            ]
            mask = torch.stack(masks, dim=0)  # (B, V)
            return torch.where(mask, torch.full((B, V), factor, device=device), torch.ones(B, V, device=device))

        # mode == "local"
        factor = cfg["factor"]
        k_primary = cfg["k"]
        mask = self._current_penetrating_vertex_mask
        if mask is None:
            # No penetration this iteration (or w_penetration == 0) -- nothing to
            # down-weight, behave exactly like local_downweight=None would.
            return torch.ones(B, V, device=device)

        seed = mask.float().t()  # (V, B)
        khop = khop_expand_mask(seed, self.vertex_adjacency, k_primary).t().bool()  # (B, V)
        self._log_coverage(iteration, khop, k_primary)

        for k_alt in cfg.get("log_extra_k", []):
            khop_alt = khop_expand_mask(seed, self.vertex_adjacency, k_alt).t().bool()
            self._log_coverage(iteration, khop_alt, k_alt)

        return torch.where(khop, torch.full((B, V), factor, device=device), torch.ones(B, V, device=device))

    def _log_coverage(self, iteration, khop_mask, k):
        """Appends one row per specimen to self.local_downweight_log: how many
        vertices (and what fraction of the mesh) are currently down-weighted at
        hop radius k. n_touched is what the "global" control replays (see
        _compute_local_downweight_vertex_weight) -- logged as an exact integer
        count, not reconstructed from coverage_fraction, so the control's
        random subset is sized identically to what "local" actually did, no
        rounding drift. k == the primary k drove this iteration's actual
        down-weighting; other logged k values are for reference only (see
        local_downweight["log_extra_k"])."""
        n_touched = khop_mask.sum(dim=1)  # (B,)
        coverage = khop_mask.float().mean(dim=1)  # (B,)
        for spec_idx, (n_t, cov) in enumerate(zip(n_touched.tolist(), coverage.tolist())):
            specimen_name = self.mesh_names[spec_idx] if spec_idx < len(self.mesh_names) else f"specimen_{spec_idx}"
            self.local_downweight_log.append(
                {"iteration": iteration, "specimen": specimen_name, "k": k, "n_touched": n_t, "coverage_fraction": cov}
            )

    def save_local_downweight_log(self):
        """Writes the per-iteration, per-specimen, per-k coverage log collected
        during run() (see _log_coverage) to a CSV file. No-op if local_downweight
        was never enabled for this stage."""
        if not self.local_downweight_log:
            return
        out_path = os.path.join(self.out_dir, f"{self.name}_local_downweight_log.csv")
        fieldnames = list(self.local_downweight_log[0].keys())
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.local_downweight_log)

    def step(self, epoch):
        """Runs step of Stage, calculating loss, and running the optimiser"""

        new_src_verts = self.smal_3d_fitter()
        offsets = new_src_verts - self.src_verts
        new_src_mesh = self.src_mesh.offset_verts(offsets.view(-1, 3))

        loss, loss_components = self.forward(new_src_mesh, iteration=epoch)

        # Optimization step
        loss.backward()
        self.optimizer.step()

        return loss, loss_components

    def plot(self):

        new_src_verts = self.smal_3d_fitter()
        offsets = new_src_verts - self.src_verts
        new_src_mesh = self.src_mesh.offset_verts(offsets.view(-1, 3))

        figtitle = f"{self.name}, its = {self.n_it}"
        plot_meshes(
            self.target_meshes,
            new_src_mesh,
            self.mesh_names,
            title=self.name,
            figtitle=figtitle,
            out_dir=os.path.join(self.out_dir, "meshes"),
            plot_normals=self.plot_normals,
        )

    def run(self, plot=False):
        """Run the entire Stage"""

        with tqdm(np.arange(self.n_it)) as tqdm_iterator:
            for i in tqdm_iterator:
                self.optimizer.zero_grad()  # Initialise optimiser
                loss, loss_components = self.step(i)

                self.losses_to_plot.append(loss)
                if not hasattr(self, "loss_components_to_plot"):
                    self.loss_components_to_plot = {k: [] for k in loss_components.keys()}
                for k, v in loss_components.items():
                    self.loss_components_to_plot[k].append(v)

                # Print loss components at the end of each stage
                if i == self.n_it - 1:
                    print(f"\nFinal loss components for stage {self.name}:")
                    for k, v in loss_components.items():
                        print(f"{k}: {v.item():.6f}")

                tqdm_iterator.set_description(f"STAGE = {self.name}, TOT_LOSS = {loss:.6f}")

        if plot:
            self.plot()

    def save_npz(self, labels=None):
        """Given a directory, saves a .npz file of all params
        labels: optional list of size n_batch, to save as labels for all entries"""

        out = {}
        for param in ["global_rot", "joint_rot", "betas", "log_beta_scales", "trans", "deform_verts", "betas_trans"]:
            out[param] = getattr(self.smal_3d_fitter, param).cpu().detach().numpy()

        v = self.smal_3d_fitter()
        out["verts"] = v.cpu().detach().numpy()
        out["faces"] = self.faces.cpu().detach().numpy()
        out["labels"] = labels

        out_title = f"{self.name}.npz"
        np.savez(os.path.join(self.out_dir, out_title), **out)

    def compute_eval_metrics(self, tau_fractions=[0.01, 0.02, 0.05], n_samples=10000):
        """Computes post-hoc F-score metrics (precision/recall/F at multiple
        thresholds) between the current fitted mesh and the target meshes,
        and writes a per-specimen CSV breakdown.

        Penetration diagnostics are computed unconditionally (even when
        w_penetration=0, e.g. the baseline arm of an ablation) so
        baseline-vs-penetration-loss comparisons have the same columns
        available on both sides -- an unramped (n_ramp_iters=0), read-only
        measurement of the final geometry, not a training-time quantity.
        """

        new_src_verts = self.smal_3d_fitter()
        offsets = new_src_verts - self.src_verts
        new_src_mesh = self.src_mesh.offset_verts(offsets.view(-1, 3))

        metrics = f_score(new_src_mesh, self.target_meshes, tau_fractions=tau_fractions, n_samples=n_samples)

        penetration_mean_depth_per_specimen, penetration_diagnostics = penetration_loss_batched(
            verts_padded=new_src_mesh.verts_padded(),
            part_vertex_indices=self.part_vertex_indices,
            part_faces=self.part_faces,
            non_adjacent_pairs=self.non_adjacent_pairs,
            proximity_tau_fraction=self.loss_weights.get("penetration_tau_fraction", 0.03),
            max_depth_fraction=self.loss_weights.get("penetration_max_depth_fraction", 0.08),
            iteration=0,
            n_ramp_iters=0,  # unramped: report the true final-geometry depth, not a training-time ramp artifact
            return_diagnostics=True,
            return_per_pair=True,
        )

        n_meshes = metrics["f_score"].shape[0]
        rows = []
        for spec_idx in range(n_meshes):
            specimen_name = self.mesh_names[spec_idx] if spec_idx < len(self.mesh_names) else f"specimen_{spec_idx}"
            row = {"stage": self.name, "specimen": specimen_name, "bbox_diag": metrics["bbox_diag"][spec_idx].item()}
            for t_idx, tau_frac in enumerate(metrics["thresholds_used"]):
                row[f"precision@{tau_frac}"] = metrics["precision"][spec_idx, t_idx].item()
                row[f"recall@{tau_frac}"] = metrics["recall"][spec_idx, t_idx].item()
                row[f"f_score@{tau_frac}"] = metrics["f_score"][spec_idx, t_idx].item()
            # diluted: averaged over every checked vertex, penetrating or not -- washes
            # out severity when few vertices penetrate. Kept for backward compatibility.
            row["penetration_mean_depth"] = penetration_mean_depth_per_specimen[spec_idx].item()
            row["penetration_max_depth"] = penetration_diagnostics["max_depth"][spec_idx].item()
            row["penetration_num_penetrating"] = penetration_diagnostics["num_penetrating"][spec_idx].item()
            # Continuous surrogate for penetration_num_penetrating -- prefer this column when
            # comparing arms/seeds, since the hard count has substantial seed-to-seed variance
            # (near-boundary vertices flip discretely; see penetration_loss.py's
            # soft_num_penetrating comment). Same units (a count), not a normalized fraction.
            row["penetration_soft_num_penetrating"] = penetration_diagnostics["soft_num_penetrating"][
                spec_idx
            ].item()
            row["penetration_fraction_penetrating"] = penetration_diagnostics["fraction_penetrating"][spec_idx].item()
            # undiluted severity: mean depth over penetrating (vertex, pair-direction)
            # instances only. 0 (not NaN) for zero-collision specimens -- penetration_loss_batched
            # clamps the denominator to >=1. Use this, not penetration_mean_depth, to judge
            # whether collisions got shallower/deeper, independent of how many there are.
            row["penetration_mean_depth_among_penetrating"] = penetration_diagnostics["mean_depth_among_penetrating"][
                spec_idx
            ].item()
            rows.append(row)

        out_path = os.path.join(self.out_dir, f"{self.name}_eval_metrics.csv")
        fieldnames = list(rows[0].keys())
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        self._write_endpoint_per_pair_penetration(penetration_diagnostics, n_meshes)
        self.compute_per_part_precision(new_src_verts, tau_fractions=tau_fractions)

        return metrics

    def _write_endpoint_per_pair_penetration(self, penetration_diagnostics, n_meshes):
        """Writes the converged per-part-pair penetration breakdown (one row per
        specimen x pair x direction), measured on the final fitted geometry."""
        per_pair = penetration_diagnostics.get("per_pair")
        if not per_pair:
            return

        rows = []
        for spec_idx in range(n_meshes):
            specimen_name = self.mesh_names[spec_idx] if spec_idx < len(self.mesh_names) else f"specimen_{spec_idx}"
            for pair_key, directions in per_pair.items():
                part_a, part_b = pair_key.split("__")
                for direction, values in directions.items():
                    rows.append({
                        "stage": self.name,
                        "specimen": specimen_name,
                        "part_a": part_a,
                        "part_b": part_b,
                        "direction": direction,
                        "n_query_vertices": values["n_query"],
                        "num_penetrating": values["num_penetrating"][spec_idx].item(),
                        "soft_num_penetrating": values["soft_num_penetrating"][spec_idx].item(),
                        "max_depth": values["max_depth"][spec_idx].item(),
                        "mean_depth_among_penetrating": values["mean_depth_among_penetrating"][spec_idx].item(),
                    })

        out_path = os.path.join(self.out_dir, f"{self.name}_per_pair_penetration.csv")
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def compute_per_part_precision(self, fitted_verts, tau_fractions=[0.01, 0.02, 0.05]):
        """Computes per-anatomical-part precision (fitted part -> scan, see
        fitter_3d/eval_metrics_part_precision.py) and writes a per-specimen,
        per-part CSV breakdown. One-directional: the scan has no part labels,
        so only precision (not recall/F-score) is available per part.
        """
        results = per_part_precision(
            fitted_verts=fitted_verts,
            target_meshes=self.target_meshes,
            part_vertex_indices=self.part_vertex_indices,
            tau_fractions=tau_fractions,
        )

        n_meshes = fitted_verts.shape[0]
        rows = []
        for spec_idx in range(n_meshes):
            specimen_name = self.mesh_names[spec_idx] if spec_idx < len(self.mesh_names) else f"specimen_{spec_idx}"
            for part_name, part_result in results.items():
                if part_name in ("bbox_diag", "thresholds_used"):
                    continue
                row = {
                    "stage": self.name,
                    "specimen": specimen_name,
                    "part": part_name,
                    "n_vertices": part_result["n_vertices"],
                }
                for t_idx, tau_frac in enumerate(results["thresholds_used"]):
                    row[f"precision@{tau_frac}"] = part_result["precision"][spec_idx, t_idx].item()
                rows.append(row)

        if not rows:
            return results

        out_path = os.path.join(self.out_dir, f"{self.name}_per_part_precision.csv")
        fieldnames = list(rows[0].keys())
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        return results


class StageManager:
    """Container for multiple stages of optimisation"""

    def __init__(self, out_dir="static_fits_output", labels=None, plot_normals=False):
        """Labels: optional list of size n_batch with labels for each mesh"""
        self.stages = []
        self.out_dir = out_dir
        self.labels = labels
        self.plot_normals = plot_normals

    def run(self):
        for n, stage in enumerate(self.stages):
            stage.run(plot=config.PLOT_RESULTS)
            stage.save_npz(labels=self.labels)
            if stage.local_downweight is not None:
                stage.save_local_downweight_log()

        # Post-hoc eval panel (F-score, penetration count/depth, per-part
        # precision) on the final stage's converged geometry -- computed
        # unconditionally, so a w_penetration=0 baseline run still gets a
        # penetration reading to compare against a w_penetration>0 run.
        self.stages[-1].compute_eval_metrics()

        # plot loss components, total loss is plotted in the run method
        self.plot_loss_components()

    def plot_losses(self, out_src="losses"):
        """Plot combined losses for all stages."""

        fig, ax = plt.subplots()
        it_start = 0  # track number of its
        for stage in self.stages:
            out_losses = [i.cpu().detach().numpy() for i in stage.losses_to_plot]
            n_it = stage.n_it
            ax.semilogy(np.arange(it_start, it_start + n_it), out_losses, label=stage.name)
            it_start += n_it

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Total loss")
        ax.legend()
        out_src = os.path.join(self.out_dir, out_src + ".png")
        plt.tight_layout()
        fig.savefig(out_src)
        plt.close(fig)

    def plot_loss_components(self, out_src="loss_components"):
        """Plot individual loss components for all stages."""

        # Get all unique loss component names across all stages
        all_components = set()
        for stage in self.stages:
            if hasattr(stage, "loss_components_to_plot"):
                all_components.update(stage.loss_components_to_plot.keys())

        # Create a subplot for each loss component
        n_components = len(all_components)
        fig, axes = plt.subplots(n_components, 1, figsize=(10, 4 * n_components))
        if n_components == 1:
            axes = [axes]

        it_start = 0
        for stage in self.stages:
            if hasattr(stage, "loss_components_to_plot"):
                for i, component in enumerate(all_components):
                    if component in stage.loss_components_to_plot:
                        values = [v.cpu().detach().numpy() for v in stage.loss_components_to_plot[component]]
                        axes[i].semilogy(np.arange(it_start, it_start + len(values)), values, label=f"{stage.name}")
                        axes[i].set_title(f"{component} loss")
                        axes[i].set_xlabel("Epoch")
                        axes[i].set_ylabel("Loss value")
                        axes[i].legend()
            it_start += stage.n_it

        plt.tight_layout()
        out_src = os.path.join(self.out_dir, out_src + ".png")
        fig.savefig(out_src)
        plt.close(fig)

    def add_stage(self, stage):
        self.stages.append(stage)
