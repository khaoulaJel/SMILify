"""G10 -- does the joint term improve the MORPHOMETRICS, or only the joints?

Bar fixed in PREREGISTRATION_G10_traits.md before this ran.

THE CIRCULARITY THIS AVOIDS
`measure.py`'s length traits are bone lengths = distances between adjacent joints. Supervising with
ground-truth joints and then scoring bone lengths against ground truth is true by construction.
So supervision is always NOISY (the realistic predictor case) and scoring is always against the
TRUE joints. sigma = 0 is retained only as an explicitly labelled circular upper bound.

THE COMPARATOR THAT MATTERS
If a predicted skeleton exists, the trivial alternative is to read the measurements straight off it
and not fit at all. Three arms at every noise level:
    P  production fit
    T  targets read directly -- measure the predicted skeleton, no fitting
    J  joint-constrained fit -- the proposal
**J must beat BOTH.** J ~ T means the fitting adds nothing over believing the predictor.

Traits, blocks and the Mosimann log-shape-ratio step are taken from `measure.py`/`analyse.py`
unchanged, so these R values sit on the same scale as G2's production baseline (median R -0.048).
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
sys.path.insert(0, os.path.join(REPO, "diagnostics", "morphometrics"))
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")
os.environ.setdefault("SMILIFY_COUPLE_JOINT_BLENDSHAPES", "0")

import pickle  # noqa: E402

import yaml  # noqa: E402

import measure as ms  # noqa: E402
from fitter_3d.trainer import SMAL3DFitter  # noqa: E402
from fitter_3d.trainer_moonshot import MoonshotStage  # noqa: E402
from fitter_3d.utils import load_meshes  # noqa: E402

WING = {"w_1_r", "w_2_r", "w_1_l", "w_2_l"}
MESHDIR = "/hpcwork/nao48500/worker_alt_data"
N_IT = int(os.environ.get("G10_ITERS", "700"))
SIGMAS = [0.0, 10.0, 15.0, 20.0]
LAM = float(os.environ.get("G10_LAM", "0.1"))
FREE = ["global_rot", "joint_rot", "betas", "trans", "deform_verts"]


def sim_fit(src, tgt):
    cs, ct = src.mean(0), tgt.mean(0)
    s0, t0 = src - cs, tgt - ct
    U, S, Vt = np.linalg.svd(s0.T @ t0)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1.0, 1.0, d]) @ Vt
    sc = (S * np.array([1, 1, d])).sum() / (s0 ** 2).sum()
    return sc * ((src - cs) @ R) + ct


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    with open(os.path.join(REPO, "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl"), "rb") as fh:
        u = pickle._Unpickler(fh); u.encoding = "latin1"; dd = u.load()
    names = [str(x) for x in np.asarray(dd["J_names"]).ravel()]
    sd_t = torch.as_tensor(np.sqrt(np.diag(np.asarray(dd["shape_cov"], dtype=np.float64))),
                           device=dev, dtype=torch.float32)
    Jr = dd["J_regressor"]
    Jr = np.asarray(Jr.todense() if hasattr(Jr, "todense") else Jr, dtype=np.float64)
    M = ms.load_model()
    bones = ms.bone_table(M)
    bnames = [b["name"] for b in bones]

    prod, pj = {}, {}
    for p in sorted(glob.glob(os.path.join(REPO, "diagnostics/moonshot/runs/Z8_W*/"
                                                 "Stage_3_deform_fine.npz"))):
        d = np.load(p, allow_pickle=True)
        V = np.asarray(d["verts"], dtype=np.float64)
        Jf = (np.einsum("jv,nvc->njc", Jr, V) if Jr.shape[1] == V.shape[1]
              else np.einsum("vj,nvc->njc", Jr, V))
        for i, lab in enumerate(d["labels"]):
            sid = str(lab).replace("_processed.obj", "").replace(".obj", "")
            prod[sid] = {k: np.asarray(d[k], dtype=np.float64)[i] for k in
                         ["betas", "global_rot", "joint_rot", "trans", "log_beta_scales",
                          "betas_trans", "deform_verts"]}
            pj[sid] = Jf[i]

    ann = []
    for f in sorted(glob.glob(os.path.join(REPO, "annotation/gt_batch1/*_joints.json"))):
        gt = json.load(open(f)); sid = gt["specimen_id"]
        obj = os.path.join(MESHDIR, f"{sid}_processed.obj")
        if sid not in prod or not os.path.exists(obj):
            continue
        P = {j["joint_name"]: np.array(j["position"], dtype=np.float64)
             for j in gt["joints"] if j.get("position") is not None}
        use = [n for n in names if n in P and n not in WING and n != "b_h"]
        idx = [names.index(n) for n in use]
        g = torch.as_tensor(sim_fit(np.array([P[n] for n in use]), pj[sid][idx]),
                            device=dev, dtype=torch.float32)
        full = torch.full((len(names), 3), float("nan"), device=dev)
        full[idx] = g
        ann.append({"sid": sid, "obj": obj, "idx": idx, "gt": g, "gt_full": full,
                    "bl": float((g[use.index("b_a_3")] - g[use.index("b_t")]).norm())})
    NS = len(ann)
    _, targets = load_meshes(mesh_files=[a["obj"] for a in ann], device=dev)
    cfg = yaml.safe_load(open(os.path.join(REPO, "diagnostics/moonshot/cfg/D1_PROD.yaml")))
    st = cfg["stages"]["Stage_3_deform_fine"]
    smal = SMAL3DFitter(batch_size=NS, device=dev)
    stage = MoonshotStage(nits=1, scheme="all", smal_3d_fitter=smal, target_meshes=targets,
                          mesh_names=[a["sid"] for a in ann], loss_weights=st["loss_weights"],
                          lr=st.get("lr", 5e-4), device=dev, n_sample=st.get("n_sample", 6000),
                          edge_mode=st.get("edge_mode", "shrink"), out_dir="/tmp/g10")
    P0 = {k: torch.as_tensor(np.stack([prod[a["sid"]][k] for a in ann]), device=dev,
                             dtype=torch.float32) for k in prod[ann[0]["sid"]]}

    def setp(d):
        with torch.no_grad():
            for k, v in d.items():
                if hasattr(smal, k):
                    getattr(smal, k).copy_(v)

    def joints_of(v):
        return (torch.einsum("jv,nvc->njc", smal.smal_model.J_regressor, v)
                if smal.smal_model.J_regressor.shape[1] == v.shape[1]
                else torch.einsum("vj,nvc->njc", smal.smal_model.J_regressor, v))

    def bone_lengths(Jfull):
        """(NS, n_bones), NaN where a required joint was not annotated."""
        out = np.full((NS, len(bones)), np.nan)
        for k, b in enumerate(bones):
            c, pa = b["child"], b["parent"]
            d = Jfull[:, c] - Jfull[:, pa]
            out[:, k] = np.linalg.norm(d, axis=-1)
        return out

    # ground-truth bone lengths, from the annotated joints
    GTfull = np.stack([a["gt_full"].cpu().numpy() for a in ann])
    Lgt = bone_lengths(GTfull)

    def block_R(Lfit):
        """per-block median R against GT, Mosimann log-shape-ratios on the common bone set."""
        ok = np.isfinite(Lgt).all(0) & (Lgt > 0).all(0) & np.isfinite(Lfit).all(0) & (Lfit > 0).all(0)
        if ok.sum() < 5:
            return {}, np.nan
        gz = np.log(Lgt[:, ok]) - np.log(Lgt[:, ok]).mean(1, keepdims=True)
        fz = np.log(Lfit[:, ok]) - np.log(Lfit[:, ok]).mean(1, keepdims=True)
        cols = [bnames[i] for i in np.where(ok)[0]]
        per, allr = {}, []
        for k, c in enumerate(cols):
            g, f = gz[:, k], fz[:, k]
            if g.std() < 1e-12 or f.std() < 1e-12:
                continue
            r = float(np.corrcoef(f, g)[0, 1])
            per.setdefault(ms.group_of(c.split("_", 1)[1]), []).append(r)
            allr.append(r)
        return {k: float(np.median(v)) for k, v in per.items()}, float(np.median(allr))

    setp(P0)
    Lp = bone_lengths(np.stack([joints_of(smal()).detach().cpu().numpy()]).squeeze(0))
    bP, mP = block_R(Lp)
    print(f"{NS} specimens, {len(bones)} bones. lambda={LAM}")
    print(f"\nPRODUCTION baseline: median R {mP:+.3f}   (G2 reported -0.048)\n")

    print("=" * 104)
    print("G10 -- trait reliability R vs ground truth.  J must beat BOTH P (production) and T (targets)")
    print("=" * 104)
    print(f"{'sigma':>7}  {'arm':<26}{'median R':>10}   per-block")
    rows = [{"sigma": None, "arm": "P production", "median_R": mP, "blocks": bP}]
    print(f"{'--':>7}  {'P production':<26}{mP:>+10.3f}   "
          + " ".join(f"{k}:{v:+.2f}" for k, v in sorted(bP.items())))

    for sig in SIGMAS:
        rng = np.random.default_rng(int(sig * 13) + 5)
        tgt, tfull = [], np.full((NS, len(names), 3), np.nan)
        for i, a in enumerate(ann):
            n = a["gt"].shape[0]
            e = torch.as_tensor(rng.normal(0.0, sig / 100.0 * a["bl"], size=(n, 3)),
                                device=dev, dtype=torch.float32)
            t = a["gt"] + e
            tgt.append(t)
            tfull[i, a["idx"]] = t.cpu().numpy()
        # ---- arm T: measure the predicted skeleton directly, no fitting
        bT, mT = block_R(bone_lengths(tfull))
        # ---- arm J: fit with the noisy targets
        setp(P0)
        for nm in FREE:
            getattr(smal, nm).requires_grad_(True)
        opt = torch.optim.Adam([getattr(smal, nm) for nm in FREE], lr=0.005)
        for it in range(N_IT):
            opt.zero_grad()
            v = smal()
            src = stage.src_mesh.offset_verts((v - stage.src_verts).view(-1, 3))
            tot, _ = stage.forward(src, it=0)
            J = joints_of(v)
            jt = torch.stack([((J[i, ann[i]["idx"]] - tgt[i]) ** 2).sum(-1).mean()
                              for i in range(NS)]).mean()
            (tot + LAM * jt).backward()
            opt.step()
            with torch.no_grad():
                smal.betas.clamp_(min=-sd_t, max=sd_t)
        bJ, mJ = block_R(bone_lengths(joints_of(smal()).detach().cpu().numpy()))
        tag = " (CIRCULAR)" if sig == 0 else ""
        for arm, m, b in [("T targets direct", mT, bT), ("J joint-constrained fit", mJ, bJ)]:
            rows.append({"sigma": sig, "arm": arm, "median_R": m, "blocks": b})
            print(f"{sig:>6.0f}%  {arm + tag:<26}{m:>+10.3f}   "
                  + " ".join(f"{k}:{v:+.2f}" for k, v in sorted(b.items())), flush=True)
        print()

    print("=" * 104)
    j15 = [r for r in rows if r["sigma"] == 15.0 and r["arm"].startswith("J")]
    t15 = [r for r in rows if r["sigma"] == 15.0 and r["arm"].startswith("T")]
    if j15 and t15:
        mj, mt = j15[0]["median_R"], t15[0]["median_R"]
        print(f"at sigma=15% (realistic predictor):  P {mP:+.3f}   T {mt:+.3f}   J {mj:+.3f}")
        print(f"  J - P = {mj - mP:+.3f}   (bar: >= +0.25)")
        print(f"  J - T = {mj - mt:+.3f}   (must be > 0, else the fitting adds nothing)")
        if mj - mP >= 0.25 and mj > mt:
            v = "PASS -- the joint term improves the MEASUREMENTS, and fitting beats the predictor"
        elif mj > mP and mj <= mt:
            v = "PARTIAL -- measurements improve, but no better than reading the skeleton directly"
        elif mj <= mP:
            v = "FAIL -- joints improve, measurements do not"
        else:
            v = "PARTIAL -- improvement below the pre-registered bar"
        print(f"\nVERDICT (pre-registered): {v}")
    print("=" * 104)
    json.dump({"production_median_R": mP, "lambda": LAM, "rows": rows},
              open(os.path.join(HERE, "g10_results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
