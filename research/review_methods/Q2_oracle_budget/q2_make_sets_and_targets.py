"""Q2 step 1-2: fresh regime sets (DEVIATIONS D4: REALPOSE and REALSCALE) and their correspondence targets.

Sets: `common/make_shift_sets.real_param_set` with NEW seeds (779000 + j), so Q2 is not scored on
the specimens S2 used to select the regime. Written as .obj (template faces) + ground_truth.npz
(verts, FK joints, all parameters, names) under /hpcwork/.../Q2/<regime>/.

Targets, all in the fitter's normalised frame (vertex mean, max abs coordinate = load_meshes):
  deployed  C3 network as JAB I_cse: 8 x 2048 points, cycle filter keep 0.5, all segments, mean
            scan position per retrieved vertex, >= 1 point per vertex. Re-implemented (same rules as
            generate_cse_correspondence_20260827.py) so that `deployed` and `opart` share ONE code
            path and differ only in the part restriction. Parity with the original script is
            checked statistically (q2_parity_probe in the sbatch: same meshes through the original
            generator, leg-correctness compared), not bitwise (different samplers).
  opart     same, but each point's retrieval is restricted to template vertices whose dominant
            joint equals the point's TRUE dominant joint (true part known; within-part choice from
            the network)
  ocorr     every template vertex's true position (dense ground truth, mask all)
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(HERE, "..", "common"))
sys.path.insert(0, os.path.join(HERE, "..", "Q3a_cse_transfer_case_study"))
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")
os.environ.setdefault("SMILIFY_COUPLE_JOINT_BLENDSHAPES", "0")

import anatomy_proxy as ap  # noqa: E402
import make_shift_sets as mss  # noqa: E402
from a4_cycle_signal import load_net  # noqa: E402

OUT = "/hpcwork/nao48500/review_methods/Q2"
REGIMES = {"REALPOSE": 779000, "REALSCALE": 779001}
N_POINTS, N_REP, KEEP = 2048, 8, 0.5


def write_obj(path, V, F):
    with open(path, "w") as f:
        f.writelines(f"v {x:.7f} {y:.7f} {z:.7f}\n" for x, y, z in V)
        f.writelines(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in F)


def sample(V, F, n, rng):
    a = np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    fi = rng.choice(len(F), n, p=a / a.sum())
    b = rng.dirichlet([1, 1, 1], n)
    return (b[:, :, None] * V[F[fi]]).sum(1), F[fi][np.arange(n), b.argmax(1)]


@torch.no_grad()
def targets(net, keys, V, F, dom, rng, device, restrict):
    """Deployed (restrict=False) or oracle-part (restrict=True) per-vertex targets for one specimen."""
    Vn = len(keys)
    ret_l, pts_l, conf_l = [], [], []
    for _ in range(N_REP):
        P, corner = sample(V, F, N_POINTS, rng)
        pts = torch.as_tensor(P, dtype=torch.float32, device=device)
        sim = net(pts.unsqueeze(0))[0] @ keys.t()
        if restrict:
            allowed = torch.as_tensor(dom[corner][:, None] == dom[None, :], device=device)
            sim = sim.masked_fill(~allowed, -1e9)
        r = sim.argmax(1)
        back = sim.argmax(0)[r]
        conf_l.append((-torch.norm(pts - pts[back], dim=1)).cpu())
        ret_l.append(r.cpu()); pts_l.append(pts.cpu())
    ret, P, conf = torch.cat(ret_l).numpy(), torch.cat(pts_l).numpy(), torch.cat(conf_l)
    k = max(1, int(len(conf) * KEEP))
    sel = (conf >= torch.topk(conf, k).values.min()).numpy()
    acc = np.zeros((Vn, 3)); cnt = np.zeros(Vn, np.int64)
    np.add.at(acc, ret[sel], P[sel]); np.add.at(cnt, ret[sel], 1)
    m = cnt >= 1
    T = np.zeros((Vn, 3), np.float32); T[m] = acc[m] / cnt[m, None]
    return T, m


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    dd = ap.load_dd()
    F = np.asarray(dd["f"]).astype(np.int64)
    dom = np.asarray(dd["weights"]).argmax(1)
    net, _ = load_net(device)
    keys = net.vertex_embeddings()
    for regime, seed in REGIMES.items():
        d = mss.real_param_set(regime, seed, device)
        odir = os.path.join(OUT, regime, "meshes")
        os.makedirs(odir, exist_ok=True)
        n = len(d["verts"])
        names = [f"q2_{regime.lower()}_{i:03d}" for i in range(n)]
        for i in range(n):
            write_obj(os.path.join(odir, f"{names[i]}.obj"), d["verts"][i], F)
        np.savez(os.path.join(odir, "ground_truth.npz"), names=np.array(names), seed=seed,
                 **{k: v for k, v in d.items() if k != "source_labels"}, source_labels=d["source_labels"])
        rng = np.random.default_rng(seed)
        T = {k: np.zeros((n, len(dom), 3), np.float32) for k in ("deployed", "opart", "ocorr")}
        M = {k: np.zeros((n, len(dom)), bool) for k in T}
        for i in range(n):
            V = d["verts"][i].astype(np.float64)
            c = V.mean(0); sc = np.abs(V - c).max(); Vn = (V - c) / sc
            T["deployed"][i], M["deployed"][i] = targets(net, keys, Vn, F, dom, rng, device, False)
            T["opart"][i], M["opart"][i] = targets(net, keys, Vn, F, dom, rng, device, True)
            T["ocorr"][i], M["ocorr"][i] = Vn.astype(np.float32), True
        for k in T:
            np.savez_compressed(os.path.join(OUT, regime, f"targets_{k}.npz"), names=np.array(names),
                                verts=T[k], mask=M[k])
        # probe: how leg-correct are the deployed / oracle-part targets? (true label of the vertex
        # whose TRUE position is nearest the target, vs the template vertex's own label)
        vlab = ap.template_vertex_labels(dd)
        legv = np.array([str(x).startswith("l_") for x in vlab])
        from scipy.spatial import cKDTree
        for k in ("deployed", "opart"):
            acc = []
            for i in range(n):
                V = d["verts"][i].astype(np.float64); c = V.mean(0); sc = np.abs(V - c).max()
                m = M[k][i] & legv
                _, nn = cKDTree((V - c) / sc).query(T[k][i][m])
                acc.append((ap.coarse_vec(vlab[nn], "leg") == ap.coarse_vec(vlab[m], "leg")).mean())
            print(f"[q2] {regime} targets_{k}: leg-correct {np.mean(acc):.3f}  coverage {M[k].mean():.3f}", flush=True)
        print(f"[q2] {regime}: {n} specimens -> {odir}", flush=True)


if __name__ == "__main__":
    main()
