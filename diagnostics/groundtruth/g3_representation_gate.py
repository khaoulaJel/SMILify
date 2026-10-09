"""G3 -- CAN THE MODEL EXPRESS THE TRUTH?  Representation vs optimisation, cleanly separated.

THE QUESTION
Every result so far says the fit disagrees with reality. None says WHY. Two possibilities with
opposite consequences:

  CASE A  representation -- the parameterisation CANNOT express the observed anatomy.
          Then no objective, prior, initialisation or optimiser will fix it, and the model needs
          more or different degrees of freedom.
  CASE B  optimisation  -- it CAN express it, but the objective prefers another solution.
          Then the model is fine and the attack surface is priors/weighting/constraints.

This is the gate that decides which, and it has never been isolated.

THE DESIGN
Fit the model's joints DIRECTLY to the human-annotated joints. Correspondence is not merely fixed,
it is trivial: model joint j must reach annotated joint j. There is no chamfer term, no candidate
selection, no data-association step -- every mechanism previously suspected of causing the residual
is removed. What remains is purely "can these parameters put these joints there".

The result is an ORACLE upper bound on the current parameterisation. Compare against:
  * the unfitted template            (what you get with no fitting at all)
  * the production D1_PROD fit       (what the pipeline actually achieves)
  * this oracle                      (the best the parameterisation can do, given truth)

  oracle ~ 0                  -> CASE B. The gap is entirely the objective's fault.
  oracle ~ production fit     -> CASE A. The parameterisation is the binding constraint.
  in between                  -> quantifies the split, which is the useful outcome.

PARAMETER GROUPS are added cumulatively so the answer is not just yes/no but WHICH degrees of
freedom carry the burden -- which is what Phase 3B would need in order to add capability
specifically where ground truth shows it is missing, rather than "more flexibility everywhere".

ALIGNMENT. The annotation and model spaces have arbitrary relative scale and pose, so a similarity
transform is solved in closed form each step and applied before the residual (rotation and scale
detached from the gradient -- standard practice, and it keeps the optimiser working on shape rather
than on chasing a global frame). Residuals are reported in the same units as G1: percent of the
specimen's own mesosoma length, so the three rows are directly comparable.

EXCLUSIONS, consistent with G1: the 4 wing joints (annotator marked them biologically absent on
every specimen) and `b_h` (coincident with `b_t` by the model's neck-pivot convention, and the two
annotation batches used different conventions for it).
"""
import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")
os.environ.setdefault("SMILIFY_COUPLE_JOINT_BLENDSHAPES", "0")

import pickle  # noqa: E402

from smal_model.smal_torch import SMAL  # noqa: E402

WING = {"w_1_r", "w_2_r", "w_1_l", "w_2_l"}
N_IT = int(os.environ.get("G3_ITERS", "1500"))


def sim_align(src, tgt):
    """Closed-form similarity (R, s, t) taking src onto tgt. Detached: it is a frame, not shape."""
    with torch.no_grad():
        cs, ct = src.mean(0), tgt.mean(0)
        s0, t0 = src - cs, tgt - ct
        U, S, Vt = torch.linalg.svd(s0.T @ t0)
        d = torch.sign(torch.det(U @ Vt))
        D = torch.diag(torch.tensor([1.0, 1.0, d], device=src.device, dtype=src.dtype))
        R = U @ D @ Vt
        sc = (S * torch.tensor([1.0, 1.0, d], device=src.device, dtype=src.dtype)).sum() \
            / (s0 ** 2).sum()
    return sc * ((src - cs) @ R) + ct


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    with open(os.path.join(REPO, "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl"), "rb") as fh:
        u = pickle._Unpickler(fh)
        u.encoding = "latin1"
        dd = u.load()
    names = [str(x) for x in np.asarray(dd["J_names"]).ravel()]
    Jt = np.asarray(dd["J"], dtype=np.float64)
    Jr = dd["J_regressor"]
    Jr = np.asarray(Jr.todense() if hasattr(Jr, "todense") else Jr, dtype=np.float64)

    prod = {}
    for p in sorted(glob.glob(os.path.join(REPO, "diagnostics/moonshot/runs/Z8_W*/"
                                                 "Stage_3_deform_fine.npz"))):
        d = np.load(p, allow_pickle=True)
        V = np.asarray(d["verts"], dtype=np.float64)
        Jf = (np.einsum("jv,nvc->njc", Jr, V) if Jr.shape[1] == V.shape[1]
              else np.einsum("vj,nvc->njc", Jr, V))
        for i, lab in enumerate(d["labels"]):
            prod[str(lab).replace("_processed.obj", "").replace(".obj", "")] = Jf[i]

    specs = []
    for f in sorted(glob.glob(os.path.join(REPO, "annotation/gt_batch1/*_joints.json"))):
        gt = json.load(open(f))
        sid = gt["specimen_id"]
        if sid not in prod:
            continue
        P = {j["joint_name"]: np.array(j["position"], dtype=np.float64)
             for j in gt["joints"] if j.get("position") is not None}
        use = [n for n in names if n in P and n not in WING and n != "b_h"]
        specs.append({"sid": sid, "use": [names.index(n) for n in use],
                      "gt": np.array([P[n] for n in use]),
                      "bl": np.linalg.norm(P["b_a_3"] - P["b_t"])})
    print(f"{len(specs)} specimens with annotations and a production fit\n")

    GROUPS = [
        ("pose only (global_rot, joint_rot)", ["global_rot", "joint_rot"]),
        ("+ betas (25-D shape space)", ["global_rot", "joint_rot", "betas"]),
        ("+ per-joint scale & translation",
         ["global_rot", "joint_rot", "betas", "log_beta_scales", "betas_trans"]),
    ]

    def residual(Jmodel, sp):
        a = sim_align(Jmodel[sp["use"]], torch.as_tensor(sp["gt"], device=dev, dtype=torch.float32))
        return (a - torch.as_tensor(sp["gt"], device=dev, dtype=torch.float32)).norm(dim=-1)

    print("=" * 100)
    print("G3 -- can the parameterisation reach the annotated joints when handed them directly?")
    print("median residual, % of mesosoma length (same units as G1)")
    print("=" * 100)

    base = {}
    for sp in specs:
        gt_t = torch.as_tensor(sp["gt"], device=dev, dtype=torch.float32)
        tpl = torch.as_tensor(Jt[sp["use"]], device=dev, dtype=torch.float32)
        pr = torch.as_tensor(prod[sp["sid"]][sp["use"]], device=dev, dtype=torch.float32)
        base[sp["sid"]] = {
            "template": float(torch.median((sim_align(tpl, gt_t) - gt_t).norm(dim=-1))) / sp["bl"] * 100,
            "production": float(torch.median((sim_align(pr, gt_t) - gt_t).norm(dim=-1))) / sp["bl"] * 100}

    # BATCHED: one SMAL instance, all specimens at once, with a per-specimen joint mask.
    # Building the model inside the loop and running specimens serially was ~50x slower and is
    # what made the first attempt time out.
    NS = len(specs)
    NJ = len(names)
    mask = torch.zeros(NS, NJ, device=dev)
    gtpad = torch.zeros(NS, NJ, 3, device=dev)
    for i, sp in enumerate(specs):
        mask[i, sp["use"]] = 1.0
        gtpad[i, sp["use"]] = torch.as_tensor(sp["gt"], device=dev, dtype=torch.float32)
    bl = torch.as_tensor([sp["bl"] for sp in specs], device=dev, dtype=torch.float32)

    def batch_resid(J):
        """Per-specimen similarity alignment on that specimen's valid joints, then residual."""
        out = []
        for i in range(NS):
            k = specs[i]["use"]
            out.append((sim_align(J[i, k], gtpad[i, k]) - gtpad[i, k]).norm(dim=-1))
        return out

    smal = SMAL(dev)
    results = {g[0]: {} for g in GROUPS}
    for gname, free in GROUPS:
        params = {
            "global_rot": torch.zeros(NS, 3, device=dev, requires_grad=True),
            "joint_rot": torch.zeros(NS, 54, 3, device=dev, requires_grad=True),
            "betas": torch.zeros(NS, 25, device=dev, requires_grad=True),
            "log_beta_scales": torch.zeros(NS, 55, 3, device=dev, requires_grad=True),
            "betas_trans": torch.zeros(NS, 55, 3, device=dev, requires_grad=True),
        }
        opt = torch.optim.Adam([params[k] for k in free], lr=0.02)
        for it in range(N_IT):
            opt.zero_grad()
            theta = torch.cat([params["global_rot"][:, None, :], params["joint_rot"]], dim=1)
            J = smal(params["betas"], theta,
                     betas_logscale=params["log_beta_scales"],
                     betas_trans=params["betas_trans"])[1]
            loss = torch.stack([(r ** 2).mean() for r in batch_resid(J)]).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            theta = torch.cat([params["global_rot"][:, None, :], params["joint_rot"]], dim=1)
            J = smal(params["betas"], theta,
                     betas_logscale=params["log_beta_scales"],
                     betas_trans=params["betas_trans"])[1]
            for i, r in enumerate(batch_resid(J)):
                results[gname][specs[i]["sid"]] = float(torch.median(r)) / specs[i]["bl"] * 100
        v = np.array(list(results[gname].values()))
        print(f"  {gname:<40} median {np.median(v):>6.1f}%   range {v.min():.1f}-{v.max():.1f}%",
              flush=True)

    tpl = np.array([base[s["sid"]]["template"] for s in specs])
    pr = np.array([base[s["sid"]]["production"] for s in specs])
    print(f"\n  {'unfitted template (reference)':<40} median {np.median(tpl):>6.1f}%")
    print(f"  {'production D1_PROD fit (reference)':<40} median {np.median(pr):>6.1f}%")

    best = min(np.median(np.array(list(results[g[0]].values()))) for g in GROUPS)
    print("\n" + "=" * 100)
    print(f"ORACLE (best parameterisation, truth handed over) : {best:.1f}%")
    print(f"PRODUCTION FIT                                    : {np.median(pr):.1f}%")
    print(f"UNFITTED TEMPLATE                                 : {np.median(tpl):.1f}%")
    closed = (np.median(pr) - best) / max(np.median(pr) - 0.0, 1e-9) * 100
    print(f"\nshare of the production residual that the parameterisation COULD remove: {closed:.0f}%")
    if best < 0.35 * np.median(pr):
        print("=> CASE B (optimisation). The model can express the truth; the objective is not")
        print("   finding it. Attack priors, weighting, constraints, initialisation.")
    elif best > 0.7 * np.median(pr):
        print("=> CASE A (representation). The parameterisation cannot express the truth even when")
        print("   handed it. No optimiser change will fix this; the model needs new capability.")
    else:
        print("=> SPLIT. Both contribute; the numbers above give the fraction attributable to each.")
    print("=" * 100)
    json.dump({"oracle": {k: v for k, v in results.items()}, "baseline": base},
              open(os.path.join(HERE, "g3_results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
