"""Q3a / A4: does the GT-free cycle-consistency signal see the real-scan failure?

Runs the C3 CSE head (the checkpoint JAB arm I_cse used) exactly as
`generate_cse_correspondence_20260827.py --normalise --filter cycle` does: 8 repeats x 2048 points,
retrieval = argmax cosine over the vertex key table, cycle distance = |p - p_back| where p_back is
the point the retrieved vertex retrieves back.

Per point it records cycle distance and whether the retrieved vertex is on the right leg:
  real scans -> expert-skeleton proxy label at the point (tr/fe/ti/ta, coxa excluded, pt->ta)
  synthetic P48 (reference) -> true label of the sampled face
Reports per specimen: median cycle distance, leg accuracy of all points and of the cycle-kept 50%,
and the within-specimen AUC of (-cycle distance) for leg correctness. Comparing the real and
synthetic cycle distributions says whether the signal also works as a specimen-level alarm.

Checkpoint loaded strictly and one named tensor compared with the raw file (CLAUDE.md).
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "fitter_3d", "pointcloud2smil"))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import anatomy_proxy as ap  # noqa: E402
from smil_cse_net import SMILCSENet  # noqa: E402
from a2_real_correspondence import read_obj, seg_of, EXCL, MESHDIR, JAB  # noqa: E402
from a2_gate_synthetic import gt_joints_p48  # noqa: E402

CKPT = os.path.join(REPO, "diagnostics/anatomical_pose_init/out_C3_cse_head_20260826/best_model.pt")
SEGS = ("tr", "fe", "ti", "ta")
N_POINTS, N_REP = 2048, 8


def load_net(device):
    ck = torch.load(CKPT, map_location=device)
    net = SMILCSENet(n_vertices=ck["n_vertices"], embed_dim=ck["embed_dim"]).to(device)
    res = net.load_state_dict(ck["model_state_dict"], strict=True)
    assert not res.missing_keys and not res.unexpected_keys
    k = "vertex_embed.weight"
    assert torch.equal(net.state_dict()[k].cpu(), ck["model_state_dict"][k].cpu()), "tensor mismatch after load"
    net.eval()
    return net, ck


def sample(V, F, n, rng):
    a = np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    fi = rng.choice(len(F), n, p=a / a.sum())
    b = rng.dirichlet([1, 1, 1], n)
    return (b[:, :, None] * V[F[fi]]).sum(1), F[fi][np.arange(n), b.argmax(1)]


@torch.no_grad()
def retrieve(net, keys, P, device):
    q = net(torch.as_tensor(P, dtype=torch.float32, device=device).unsqueeze(0))[0]
    sim = q @ keys.t()
    r = sim.argmax(1)
    back = sim.argmax(0)[r]
    pts = torch.as_tensor(P, dtype=torch.float32, device=device)
    return r.cpu().numpy(), torch.norm(pts - pts[back], dim=1).cpu().numpy()


def auc(score, y):
    y = np.asarray(y, bool)
    if y.all() or (~y).all():
        return None
    from scipy.stats import rankdata
    rk = rankdata(score)
    return float((rk[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def summarise(cyc, correct):
    keep = cyc <= np.median(cyc)
    return dict(n=int(len(cyc)), cycle_median=float(np.median(cyc)),
                leg_acc_all=float(correct.mean()), leg_acc_cyclekept50=float(correct[keep].mean()),
                auc_neg_cycle=auc(-cyc, correct))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    net, ck = load_net(device)
    keys = net.vertex_embeddings()
    dd = ap.load_dd()
    names, parent = ap.joint_tree(dd)
    vlab = ap.template_vertex_labels(dd)
    F_t = np.asarray(dd["f"]).astype(np.int64)
    rng = np.random.default_rng(0)
    out = {"ckpt_epoch": int(ck["epoch"]), "real": {}, "synthetic": {}}

    gt = json.load(open(os.path.join(JAB, "gt_joints_fitframe.json")))
    for sid in sorted(gt):
        V, F = read_obj(os.path.join(MESHDIR, f"{sid}_processed.obj"))
        c = V.mean(0)
        V = (V - c) / np.abs(V - c).max()
        joints = {k: np.asarray(v["fit"], float) for k, v in gt[sid]["joints"].items()}
        cyc_l, cor_l = [], []
        for _ in range(N_REP):
            P, _ = sample(V, F, N_POINTS, rng)
            r, cyc = retrieve(net, keys, P, device)
            prox, _ = ap.label_points(P, joints, parent, exclude=EXCL)
            tl = vlab[r]
            m = np.isin([seg_of(n) for n in tl], SEGS)
            cyc_l.append(cyc[m])
            cor_l.append(ap.coarse_vec(tl[m], "leg") == ap.coarse_vec(prox[m], "leg"))
        s = summarise(np.concatenate(cyc_l), np.concatenate(cor_l))
        out["real"][sid] = s
        print(f"REAL {sid[:14]:<14} cyc_med {s['cycle_median']:.4f} leg_acc {s['leg_acc_all']:.2f} "
              f"-> kept50 {s['leg_acc_cyclekept50']:.2f}  AUC {s['auc_neg_cycle']}", flush=True)

    z, _, _ = gt_joints_p48()
    for i in range(0, len(z["verts"]), 4):          # every 4th P48 specimen: 12 references
        V = z["verts"][i].astype(float)
        c = V.mean(0)
        V = (V - c) / np.abs(V - c).max()           # same normalisation as the real scans
        cyc_l, cor_l = [], []
        for _ in range(N_REP):
            P, corner = sample(V, F_t, N_POINTS, rng)
            r, cyc = retrieve(net, keys, P, device)
            tl, true = vlab[r], vlab[corner]
            m = np.isin([seg_of(n) for n in tl], SEGS)
            cyc_l.append(cyc[m])
            cor_l.append(ap.coarse_vec(tl[m], "leg") == ap.coarse_vec(true[m], "leg"))
        s = summarise(np.concatenate(cyc_l), np.concatenate(cor_l))
        out["synthetic"][str(z["names"][i])] = s
        print(f"SYN  {str(z['names'][i]):<14} cyc_med {s['cycle_median']:.4f} leg_acc {s['leg_acc_all']:.2f} "
              f"-> kept50 {s['leg_acc_cyclekept50']:.2f}  AUC {s['auc_neg_cycle']}", flush=True)
    json.dump(out, open(os.path.join(HERE, "out", "A4_cycle_signal.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
