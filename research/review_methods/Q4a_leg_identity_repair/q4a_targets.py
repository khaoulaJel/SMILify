"""Q4a target generators (PREREGISTRATION.md). Input: deployed CSE targets + the B1 fit at H2 for the
same regime and seed (skeleton placed with the deployed targets). Output: the arm's targets for the
surface stages, in the fitter's normalised frame, fitter correspondence format.

  LCF     keep target v only if the nearest H2-fit vertex to the target position has the same leg as v
          (leg vertices only; other vertices unchanged)
  REEST   re-run the network with a spatial prior from the H2 fit:
          score(p, v) = logit_scale * cos(q_p, k_v) - ||p - x_v(H2)||^2 / (2 sigma^2), sigma = 0.05;
          same 8 x 2048 sampling, cycle filter keep 0.5 and aggregation as the deployed targets
  ORACLE  drop exactly the wrong-leg targets using ground truth (upper bound for a filter)
Also writes a probe: wrong-leg fraction and coverage before/after (the pre-registered mechanism check).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(HERE, "..", "common"))
sys.path.insert(0, os.path.join(HERE, "..", "Q2_oracle_budget"))
sys.path.insert(0, os.path.join(HERE, "..", "Q3a_cse_transfer_case_study"))
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import anatomy_proxy as ap  # noqa: E402
from a4_cycle_signal import load_net  # noqa: E402
from q2_make_sets_and_targets import sample, N_POINTS, N_REP, KEEP  # noqa: E402

Q2 = "/hpcwork/nao48500/review_methods/Q2"
OUT = "/hpcwork/nao48500/review_methods/Q4a"
SIGMA = 0.05


def wrong_leg_fraction(T, M, Vn, legc, legv):
    m = M & legv
    _, nn = cKDTree(Vn).query(T[m])
    return float((legc[nn] != legc[m]).mean()) if m.any() else float("nan")


@torch.no_grad()
def reest_targets(net, keys, Vn, F, Xfit, rng, device):
    Vt = len(keys)
    xf = torch.as_tensor(Xfit, dtype=torch.float32, device=device)
    ret_l, pts_l, conf_l = [], [], []
    for _ in range(N_REP):
        P, _ = sample(Vn, F, N_POINTS, rng)
        pts = torch.as_tensor(P, dtype=torch.float32, device=device)
        s = net.logit_scale() * (net(pts.unsqueeze(0))[0] @ keys.t()) \
            - torch.cdist(pts, xf).pow(2) / (2 * SIGMA ** 2)
        r = s.argmax(1)
        back = s.argmax(0)[r]
        conf_l.append((-torch.norm(pts - pts[back], dim=1)).cpu()); ret_l.append(r.cpu()); pts_l.append(pts.cpu())
    ret, P, conf = torch.cat(ret_l).numpy(), torch.cat(pts_l).numpy(), torch.cat(conf_l)
    k = max(1, int(len(conf) * KEEP))
    sel = (conf >= torch.topk(conf, k).values.min()).numpy()
    acc = np.zeros((Vt, 3)); cnt = np.zeros(Vt, np.int64)
    np.add.at(acc, ret[sel], P[sel]); np.add.at(cnt, ret[sel], 1)
    m = cnt >= 1
    T = np.zeros((Vt, 3), np.float32); T[m] = acc[m] / cnt[m, None]
    return T, m


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--regime", required=True)
    a.add_argument("--seed", type=int, required=True)
    args = a.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    dd = ap.load_dd()
    F = np.asarray(dd["f"]).astype(np.int64)
    vlab = ap.template_vertex_labels(dd)
    legc = ap.coarse_vec(vlab, "leg")
    legv = legc != "other"
    gt = np.load(os.path.join(Q2, args.regime, "meshes", "ground_truth.npz"), allow_pickle=True)
    dep = np.load(os.path.join(Q2, args.regime, "targets_deployed.npz"))
    h2 = np.load(os.path.join(Q2, args.regime, "runs", f"B1_s{args.seed}_hier", "H2_joint.npz"), allow_pickle=True)
    labs = [str(x).replace(".obj", "") for x in h2["labels"]]
    net, _ = load_net(device)
    keys = net.vertex_embeddings()
    rng = np.random.default_rng(args.seed)
    names = list(dep["names"])
    out = {k: (np.zeros_like(dep["verts"]), np.zeros_like(dep["mask"])) for k in ("LCF", "REEST", "ORACLE")}
    probe = {k: [] for k in ("deployed", "LCF", "REEST", "ORACLE")}
    for i, sid in enumerate(names):
        g = list(gt["names"]).index(sid)
        V = gt["verts"][g].astype(np.float64); c = V.mean(0); Vn = (V - c) / np.abs(V - c).max()
        Xfit = h2["verts"][labs.index(sid)]
        T0, M0 = dep["verts"][i], dep["mask"][i]
        # LCF: leg of the nearest H2-fit vertex to each target position
        _, nf = cKDTree(Xfit).query(T0)
        keep = ~(M0 & legv) | (legc[nf] == legc)
        out["LCF"][0][i], out["LCF"][1][i] = T0, M0 & keep
        # ORACLE: true leg of the target position
        _, nt = cKDTree(Vn).query(T0)
        out["ORACLE"][0][i], out["ORACLE"][1][i] = T0, M0 & (~legv | (legc[nt] == legc))
        # REEST
        out["REEST"][0][i], out["REEST"][1][i] = reest_targets(net, keys, Vn, F, Xfit, rng, device)
        probe["deployed"].append((wrong_leg_fraction(T0, M0, Vn, legc, legv), float(M0.mean())))
        for k in ("LCF", "REEST", "ORACLE"):
            probe[k].append((wrong_leg_fraction(out[k][0][i], out[k][1][i], Vn, legc, legv), float(out[k][1][i].mean())))
    os.makedirs(os.path.join(OUT, args.regime), exist_ok=True)
    for k, (T, M) in out.items():
        np.savez_compressed(os.path.join(OUT, args.regime, f"targets_{k}_s{args.seed}.npz"),
                            names=np.array(names), verts=T, mask=M)
    summ = {k: dict(wrong_leg=float(np.nanmean([p[0] for p in v])), coverage=float(np.mean([p[1] for p in v])),
                    wrong_leg_per_specimen=[p[0] for p in v]) for k, v in probe.items()}
    json.dump(summ, open(os.path.join(HERE, "out", f"probe_{args.regime}_s{args.seed}.json"), "w"), indent=1)
    for k, v in summ.items():
        print(f"[q4a] {args.regime} s{args.seed} {k:<8} wrong-leg {v['wrong_leg']:.3f}  coverage {v['coverage']:.3f}", flush=True)


if __name__ == "__main__":
    main()
