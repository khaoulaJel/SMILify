"""G8 -- does a joint-position term propagate anatomically, or memorise its targets?

Bar and decision rule fixed in PREREGISTRATION_G8_heldout.md BEFORE this ran.

G7 showed `production objective + lambda*joint_target` cuts joint error 38.9% -> 27.5% at 1.19x
chamfer. But it measured the residual on the SAME joints it supplied as targets, using a complete
human skeleton that does not exist at inference. This asks whether constraining a SUBSET improves
the joints that were NOT constrained -- i.e. whether the term pulls the model into a globally
correct configuration, or memorises whatever it is handed.

Scheme D (scrambled targets) is VOIDING: if permuted, anatomically meaningless targets improve
held-out joints as much as true ones, the effect is extra regularisation rather than anatomy and
the primary result does not stand however large it looks.
"""
import glob
import itertools
import json
import os
import sys

import numpy as np
import torch
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")
os.environ.setdefault("SMILIFY_COUPLE_JOINT_BLENDSHAPES", "0")

import pickle  # noqa: E402

import yaml  # noqa: E402

from fitter_3d.trainer import SMAL3DFitter  # noqa: E402
from fitter_3d.trainer_moonshot import MoonshotStage  # noqa: E402
from fitter_3d.utils import load_meshes  # noqa: E402

WING = {"w_1_r", "w_2_r", "w_1_l", "w_2_l"}
MESHDIR = "/hpcwork/nao48500/worker_alt_data"
N_IT = int(os.environ.get("G8_ITERS", "700"))
LAMBDAS = [0.03, 0.1]
FREE = ["global_rot", "joint_rot", "betas", "trans", "deform_verts"]
PROXIMAL = ("_co_", "_tr_", "_fe_")


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
        ann.append({"sid": sid, "obj": obj, "use": use, "idx": idx,
                    "bt": use.index("b_t"), "ba3": use.index("b_a_3"),
                    "gt": torch.as_tensor(sim_fit(np.array([P[n] for n in use]), pj[sid][idx]),
                                          device=dev, dtype=torch.float32)})
    NS = len(ann)
    _, targets = load_meshes(mesh_files=[a["obj"] for a in ann], device=dev)
    cfg = yaml.safe_load(open(os.path.join(REPO, "diagnostics/moonshot/cfg/D1_PROD.yaml")))
    st = cfg["stages"]["Stage_3_deform_fine"]
    smal = SMAL3DFitter(batch_size=NS, device=dev)
    stage = MoonshotStage(nits=1, scheme="all", smal_3d_fitter=smal, target_meshes=targets,
                          mesh_names=[a["sid"] for a in ann], loss_weights=st["loss_weights"],
                          lr=st.get("lr", 5e-4), device=dev, n_sample=st.get("n_sample", 6000),
                          edge_mode=st.get("edge_mode", "shrink"), out_dir="/tmp/g8")
    P0 = {k: torch.as_tensor(np.stack([prod[a["sid"]][k] for a in ann]), device=dev,
                             dtype=torch.float32) for k in prod[ann[0]["sid"]]}

    def setp(d):
        with torch.no_grad():
            for k, v in d.items():
                if hasattr(smal, k):
                    getattr(smal, k).copy_(v)

    def joints(v):
        return (torch.einsum("jv,nvc->njc", smal.smal_model.J_regressor, v)
                if smal.smal_model.J_regressor.shape[1] == v.shape[1]
                else torch.einsum("vj,nvc->njc", smal.smal_model.J_regressor, v))

    def resid_on(sel_per_spec):
        """median-over-specimens of the per-specimen median residual on the given joint subset."""
        with torch.no_grad():
            J = joints(smal())
        out = []
        for i, a in enumerate(ann):
            s = sel_per_spec[i]
            if not len(s):
                continue
            r = (J[i, [a["idx"][k] for k in s]] - a["gt"][s]).norm(dim=-1)
            bl = float((a["gt"][a["ba3"]] - a["gt"][a["bt"]]).norm())
            out.append(float(torch.median(r)) / bl * 100)
        return np.array(out)

    def run(sup, lam, scramble=False, seed=0):
        """sup[i] = indices (into a['use']) supervised for specimen i."""
        setp(P0)
        for nm in FREE:
            getattr(smal, nm).requires_grad_(True)
        opt = torch.optim.Adam([getattr(smal, nm) for nm in FREE], lr=0.005)
        rng = np.random.default_rng(seed)
        tgt = []
        for i, a in enumerate(ann):
            g = a["gt"][sup[i]]
            if scramble:
                g = g[torch.as_tensor(rng.permutation(len(g)), device=dev)]
            tgt.append(g)
        for it in range(N_IT):
            opt.zero_grad()
            v = smal()
            src = stage.src_mesh.offset_verts((v - stage.src_verts).view(-1, 3))
            tot, _ = stage.forward(src, it=0)
            J = joints(v)
            jt = torch.stack([((J[i, [ann[i]["idx"][k] for k in sup[i]]] - tgt[i]) ** 2)
                              .sum(-1).mean() for i in range(NS)]).mean()
            (tot + lam * jt).backward()
            opt.step()
            with torch.no_grad():
                smal.betas.clamp_(min=-sd_t, max=sd_t)
        with torch.no_grad():
            ch = float(stage.forward(stage.src_mesh.offset_verts(
                (smal() - stage.src_verts).view(-1, 3)), it=0)[1].get("chamfer", np.nan))
            bz = float((smal.betas.detach() - P0["betas"]).norm())
        return ch, bz

    # ---- baselines
    setp(P0)
    ALL = [list(range(len(a["use"]))) for a in ann]
    base_all = resid_on(ALL)
    print(f"{NS} specimens; production baseline (all joints): {np.median(base_all):.2f}%\n")

    def split_anatomical():
        sup, hel = [], []
        for a in ann:
            s = [k for k, n in enumerate(a["use"])
                 if n.startswith("b_") or any(t in n for t in PROXIMAL)]
            h = [k for k in range(len(a["use"])) if k not in s]
            sup.append(s); hel.append(h)
        return sup, hel

    def split_random(frac, seed):
        rng = np.random.default_rng(seed)
        sup, hel = [], []
        for a in ann:
            n = len(a["use"]); k = max(2, int(round(frac * n)))
            p = rng.permutation(n)
            sup.append(sorted(p[:k].tolist())); hel.append(sorted(p[k:].tolist()))
        return sup, hel

    rows = []
    print("=" * 108)
    print("G8 -- held-out joint residual. PRIMARY = held-out; supervised shown to confirm the term acts")
    print("=" * 108)
    print(f"{'scheme':<34}{'lam':>6}{'sup n':>7}{'SUPERVISED':>12}{'HELD-OUT':>11}"
          f"{'base(held)':>12}{'rel impr':>10}{'chamfer':>10}")

    def report(tag, sup, hel, lam, scramble=False, seed=0):
        ch, bz = run(sup, lam, scramble, seed)
        rs, rh = resid_on(sup), resid_on(hel)
        setp(P0); bh = resid_on(hel)
        rel = 100 * (1 - np.median(rh) / np.median(bh))
        nb = int((rh < bh).sum())
        p = float(stats.binomtest(nb, len(rh), 0.5).pvalue)
        rows.append({"scheme": tag, "lam": lam, "scramble": scramble,
                     "sup": float(np.median(rs)), "held": float(np.median(rh)),
                     "base_held": float(np.median(bh)), "rel": rel, "n_better": nb,
                     "n": len(rh), "p": p, "chamfer": ch, "beta_move": bz,
                     "sup_n": int(np.median([len(s) for s in sup]))})
        print(f"{tag:<34}{lam:>6.3g}{np.median([len(s) for s in sup]):>7.0f}"
              f"{np.median(rs):>11.2f}%{np.median(rh):>10.2f}%{np.median(bh):>11.2f}%"
              f"{rel:>9.1f}%{ch:>10.5f}", flush=True)

    for lam in LAMBDAS:
        s, h = split_anatomical()
        report("A anatomical (prox+axis -> distal)", s, h, lam)
    for lam in LAMBDAS:
        for f in range(3):
            s, h = split_random(0.5, 100 + f)
            report(f"B random 50% fold{f}", s, h, lam)
    for frac in [0.1, 0.25, 0.75]:
        for f in range(2):
            s, h = split_random(frac, 200 + f)
            report(f"C sparsity {int(frac*100)}% fold{f}", s, h, 0.1)
    for f in range(3):
        s, h = split_random(0.5, 100 + f)
        report(f"D SCRAMBLED 50% fold{f}", s, h, 0.1, scramble=True, seed=300 + f)

    A = [r for r in rows if r["scheme"].startswith("A")]
    D = [r for r in rows if r["scheme"].startswith("D")]
    B = [r for r in rows if r["scheme"].startswith("B")]
    bestA = max(A, key=lambda r: r["rel"])
    relD = float(np.median([r["rel"] for r in D]))
    print("\n" + "=" * 108)
    print(f"[VOIDING] scheme D (scrambled targets) held-out improvement: {relD:+.1f}%")
    void = relD >= 0.5 * bestA["rel"] and bestA["rel"] > 0
    print(f"  -> {'FAIL -- VOID: scrambled targets help as much as real ones' if void else 'PASS -- real targets do something scrambled ones do not'}")
    print(f"\nscheme A best: held-out {bestA['held']:.2f}% vs base {bestA['base_held']:.2f}%  "
          f"= {bestA['rel']:+.1f}% relative, {bestA['n_better']}/{bestA['n']} specimens better, "
          f"p={bestA['p']:.4f}")
    print(f"scheme B (random 50%) median relative improvement: "
          f"{float(np.median([r['rel'] for r in B])):+.1f}%")
    verdict = ("VOID (scheme D)" if void else
               "PASS -- the term PROPAGATES" if (bestA["rel"] >= 20 and bestA["n_better"] >= 10
                                                 and bestA["p"] < 0.05) else
               "FAIL -- the term MEMORISES" if bestA["rel"] < 10 else "PARTIAL")
    print(f"\nVERDICT (pre-registered): {verdict}")
    print("=" * 108)
    json.dump(rows, open(os.path.join(HERE, "g8_results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
