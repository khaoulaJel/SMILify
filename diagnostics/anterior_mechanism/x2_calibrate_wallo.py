"""X2 Task 1 -- calibrate w_allo (PREREGISTRATION_X2_allometric_prior.md Sec3 Step 1).

Method (see the preregistration for the full rationale):
  1. Run arm A's config (A_nocap.yaml, Stage_2_deform_coarse ONLY, w_scale=0, no w_allo term
     active) on `bench50_clean`, checkpointing every `--ckpt_every` iterations up to
     `--max_its`.
  2. At EACH checkpoint, compute the RAW (unweighted) gradient norm of scale_barrier w.r.t.
     log_beta_scales, and separately the RAW gradient norm of allometric_prior_loss w.r.t.
     betas / joint_rot / global_rot / trans / deform_verts (the tensors _allo_verts /
     _allo_joints trace back to via the forward pass -- NOT log_beta_scales/betas_trans,
     which the allo term's forward path does not read).

     EMPIRICAL FINDING (2026-09-08, this run): a 75-iteration pilot, as the preregistration's
     initial guess of "50-100 iterations" suggested, is STILL a degenerate calibration point --
     scale_barrier's free band is +/-ln(2)=0.693 in log-scale, wide enough that NOTHING in arm
     A's Stage_2 loss (w_scale=0, w_sym=0.5 only weakly ties L/R pairs that start believed
     identical) pushes any joint's log_beta_scales past it within 75 its; scale_barrier's value
     AND gradient were both exactly 0.0 at that checkpoint -- the same degenerate case the
     preregistration flagged for t=0, just not resolved by the originally-guessed pilot length.
     Fix: keep extending the SAME Stage_2-only pilot (still no Stage_0/Stage_1, still matching
     arm A exactly) via checkpointing until scale_barrier first engages, and use the EARLIEST
     such checkpoint (closest in spirit to "the start of the relevant stage") as the calibration
     point, rather than guessing a fixed iteration count in advance.
  3. w_allo = 0.052 * ||grad(scale_barrier)|| / ||grad(l_allo)||, combined L2 norm across all
     parameter tensors each term touches (concatenate-then-norm), evaluated at the first
     non-degenerate checkpoint found.
  4. Sanity-check: two more iterations from that checkpoint using the REAL weighted w_allo term,
     confirm no NaN/Inf and no order-of-magnitude loss blowup.

Writes diagnostics/anterior_mechanism/out_X2/wallo_calibration.json.
"""
import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import config  # noqa: E402
from fitter_3d.utils import load_meshes  # noqa: E402
from fitter_3d.trainer import SMAL3DFitter  # noqa: E402
from fitter_3d.trainer_moonshot import MoonshotStage  # noqa: E402
from fitter_3d.joint_limits import (  # noqa: E402
    scale_barrier, build_head_width_reporter_matrix, allometric_prior_loss,
)

W_SCALE_REF = 0.052  # arm B's shipped scale_cap weight, the calibration anchor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh_dir", default="diagnostics/moonshot/bench50_clean")
    ap.add_argument("--yaml_src", default="diagnostics/anterior_mechanism/A_nocap.yaml")
    ap.add_argument("--max_its", type=int, default=1000, help="cap = full Stage_2 length")
    ap.add_argument("--ckpt_every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="diagnostics/anterior_mechanism/out_X2")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[calib] device={device}", flush=True)

    out_dir = os.path.join(REPO, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    mesh_dir = os.path.join(REPO, args.mesh_dir)
    mesh_files = sorted(glob.glob(os.path.join(mesh_dir, "*.obj")))
    mesh_names = [os.path.basename(f) for f in mesh_files]
    print(f"[calib] {len(mesh_files)} meshes from {mesh_dir}", flush=True)

    _, target_meshes = load_meshes(mesh_files=mesh_files, device=device)

    with open(os.path.join(REPO, args.yaml_src)) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    stage_kw = dict(cfg["stages"]["Stage_2_deform_coarse"])
    lw = dict(stage_kw.pop("loss_weights"))
    assert lw.get("w_scale", 0.0) == 0.0, "expected arm A's w_scale=0.0 at Stage_2"
    assert "w_allo" not in lw, "arm A's yaml must not already carry w_allo"
    stage_kw["nits"] = args.max_its  # cap = full Stage_2 length; we checkpoint along the way

    smal = SMAL3DFitter(batch_size=len(target_meshes), device=device)

    stage = MoonshotStage(
        name="calib_pilot",
        smal_3d_fitter=smal,
        target_meshes=target_meshes,
        mesh_names=mesh_names,
        out_dir=out_dir,
        device=device,
        loss_weights=lw,
        **stage_kw,
    )

    with open(config.SMAL_FILE, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        dd = u.load()
    jnames = list(dd["J_names"])
    bt_idx = jnames.index("b_t")
    ba5_idx = jnames.index("b_a_5")
    allo_R = build_head_width_reporter_matrix(dd["v_template"], dtype=torch.float32, device=device)
    allo_param_names = ["betas", "joint_rot", "global_rot", "trans", "deform_verts"]

    def probe():
        """Raw (unweighted) grad norms of scale_barrier and l_allo at the CURRENT parameter
        state, via fresh, independent forward/backward passes that do not touch the live
        training graph or optimizer state."""
        lbs = smal.log_beta_scales
        l_sc = scale_barrier(lbs)
        (g_sc,) = torch.autograd.grad(l_sc, lbs, retain_graph=False, allow_unused=False)
        norm_sc = float(g_sc.norm().item())

        allo_params = [getattr(smal, n) for n in allo_param_names]
        allo_verts, allo_joints = smal(return_joints=True)
        bl = torch.linalg.norm(allo_joints[:, bt_idx] - allo_joints[:, ba5_idx], dim=-1)
        hw = torch.einsum("jv,bvc->bjc", allo_R.to(allo_verts.dtype), allo_verts)
        hw = torch.linalg.norm(hw[:, 0] - hw[:, 1], dim=-1)
        l_allo = allometric_prior_loss(hw, bl)
        grads = torch.autograd.grad(l_allo, allo_params, retain_graph=False, allow_unused=True)
        flat, per_tensor_norms = [], {}
        for name, g in zip(allo_param_names, grads):
            if g is None:
                per_tensor_norms[name] = 0.0
                continue
            per_tensor_norms[name] = float(g.norm().item())
            flat.append(g.reshape(-1))
        norm_allo = float(torch.cat(flat).norm().item()) if flat else 0.0
        return dict(
            it=None, norm_sc=norm_sc, norm_allo=norm_allo,
            l_sc=float(l_sc), l_allo=float(l_allo),
            body_length_mean=float(bl.mean()), head_width_mean=float(hw.mean()),
            per_tensor_norms=per_tensor_norms,
        )

    print(f"[calib] running Stage_2_deform_coarse for the FULL {args.max_its} its (its own "
          f"pre-registered length), checkpointing every {args.ckpt_every} its, probing "
          f"scale_barrier/l_allo grad norms at each (w_scale=0, no w_allo -- matches arm A "
          f"exactly). CALIBRATION POINT = the LAST checkpoint (end of Stage_2, where Stage_3 "
          f"picks up with the identical loss_weights) -- NOT the first tick where scale_barrier "
          f"crosses zero: an earlier attempt (75-it pilot) found the crossing point itself is a "
          f"second, subtler degenerate case -- the raw gradient right at the free-band boundary "
          f"is dominated by which single joint/specimen entry JUST crossed ln(2), giving a "
          f"noise-sized ||grad|| (1.0e-5 at it=350, vs ||grad_allo||=0.060 at the same point --"
          f" a ratio that would round w_allo to a degenerate 8.3e-6). The end-of-Stage_2 value is"
          f" a far more representative, less noise-dominated 'realistic mid-training state'.",
          flush=True)

    checkpoints = []
    it = 0
    while it < args.max_its:
        n_this = min(args.ckpt_every, args.max_its - it)
        for k in range(n_this):
            stage.optimizer.zero_grad()
            loss, comp = stage.step(it + k)
            stage.losses_to_plot.append(float(loss.detach()))
        it += n_this
        rec = probe()
        rec["it"] = it
        checkpoints.append(rec)
        print(f"[calib] ckpt it={it:4d}  loss={stage.losses_to_plot[-1]:.6f}  "
              f"scale_barrier={rec['l_sc']:.6f} ||grad_sc||={rec['norm_sc']:.6f}   "
              f"l_allo={rec['l_allo']:.6f} ||grad_allo||={rec['norm_allo']:.6f}", flush=True)

    chosen = checkpoints[-1]
    print(f"\n[calib] using LAST checkpoint (it={chosen['it']}, end of Stage_2) as the "
          f"calibration point.", flush=True)

    norm_sc, norm_allo = chosen["norm_sc"], chosen["norm_allo"]
    print(f"\n[calib] CALIBRATION CHECKPOINT: it={chosen['it']}", flush=True)
    print(f"[calib] scale_barrier(log_beta_scales) = {chosen['l_sc']:.6f}   ||grad|| = {norm_sc:.6f}",
          flush=True)
    print(f"[calib] allometric_prior_loss = {chosen['l_allo']:.6f}   "
          f"body_length mean={chosen['body_length_mean']:.5f}   "
          f"head_width mean={chosen['head_width_mean']:.5f}", flush=True)
    print(f"[calib] per-tensor grad norms: {chosen['per_tensor_norms']}", flush=True)
    print(f"[calib] ||grad(l_allo)|| (combined) = {norm_allo:.6f}", flush=True)

    if norm_allo <= 1e-12:
        raise SystemExit("l_allo gradient is degenerate (~0) at the chosen checkpoint -- cannot "
                          "calibrate w_allo as a ratio against it.")

    ratio = norm_sc / norm_allo
    w_allo_raw = W_SCALE_REF * ratio
    print(f"[calib] ratio ||grad_sc||/||grad_allo|| = {ratio:.6f}", flush=True)
    print(f"[calib] w_allo (raw) = {W_SCALE_REF} * {ratio:.6f} = {w_allo_raw:.6f}", flush=True)

    def round_sig(x, sig=2):
        if x == 0:
            return 0.0
        from math import log10, floor
        d = sig - int(floor(log10(abs(x)))) - 1
        return round(x, d)

    w_allo_final = round_sig(w_allo_raw, 2)
    print(f"[calib] w_allo (rounded, 2 s.f.) = {w_allo_final}", flush=True)

    # ---------------------------------------------------------------- sanity check
    print(f"[calib] sanity check: 2 more iterations from it={chosen['it']} with the REAL weighted "
          f"w_allo={w_allo_final} term", flush=True)
    stage.loss_weights["w_allo"] = w_allo_final
    stage.allo_R = allo_R
    stage.allo_bt_idx = bt_idx
    stage.allo_ba5_idx = ba5_idx
    sanity = []
    nan_or_blowup = False
    prev_loss = stage.losses_to_plot[-1]
    for i in range(2):
        stage.optimizer.zero_grad()
        loss, comp = stage.step(chosen["it"] + i)
        lval = float(loss.detach())
        entry = {k: float(v.detach()) for k, v in comp.items()}
        entry["total"] = lval
        sanity.append(entry)
        aval = float(comp["allo"].detach()) if "allo" in comp else float("nan")
        print(f"[calib] sanity it {i}: total={lval:.6f}  allo={aval:.6f}", flush=True)
        if not np.isfinite(lval) or lval > 100 * max(prev_loss, 1e-6):
            nan_or_blowup = True
        prev_loss = lval

    print(f"[calib] sanity check {'PASSED (finite, no blowup)' if not nan_or_blowup else 'FAILED'}",
          flush=True)

    payload = {
        "max_its": args.max_its,
        "ckpt_every": args.ckpt_every,
        "seed": args.seed,
        "checkpoints": checkpoints,
        "chosen_checkpoint_it": chosen["it"],
        "scale_barrier_value": chosen["l_sc"],
        "grad_norm_scale_barrier": norm_sc,
        "allo_loss_value": chosen["l_allo"],
        "body_length_mean": chosen["body_length_mean"],
        "head_width_mean": chosen["head_width_mean"],
        "grad_norm_allo_per_tensor": chosen["per_tensor_norms"],
        "grad_norm_allo_combined": norm_allo,
        "ratio": ratio,
        "w_scale_ref": W_SCALE_REF,
        "w_allo_raw": w_allo_raw,
        "w_allo_final": w_allo_final,
        "sanity_check": sanity,
        "sanity_check_passed": bool(not nan_or_blowup),
    }
    out_path = os.path.join(out_dir, "wallo_calibration.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[calib] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
