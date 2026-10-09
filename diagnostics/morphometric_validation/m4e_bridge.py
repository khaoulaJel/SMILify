"""M4-E -- does the interpretable phenotype agree with the dense one?

Per PREREGISTRATION_M4_corpus_analysis.md §6. Scale-free on both sides, so it does not depend on
the unresolved absolute-scale question.

Question: do HW/WL and HL/WL correspond to biologically structured DIRECTIONS in the dense 3D
phenotype -- i.e. do the validated scalars recover a measurable component of the genus structure
present in the dense fitted shape?

Trait roles are those frozen in M3/M4-A: HW primary, HL secondary. Not modified by M4-C.
"""
import json, os, glob, collections, warnings
import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
sys.path[:0] = [REPO, os.path.join(REPO, "diagnostics/groundtruth"),
                os.path.join(REPO, "diagnostics/morphometrics"),
                os.path.join(REPO, "diagnostics/morphometric_validation")]
from measure import load_model, kabsch                  # noqa: E402
from mv_framework import forward_pair                   # noqa: E402
OUT = os.path.join(REPO, "diagnostics/morphometric_validation/out")
SEED = 20260910
NPC = 50


def dense_shape():
    """Canonical-pose vertices, Procrustes-normalised (centre, unit scale, rotate to template).
    Pose and size are BOTH removed, so this is a scale-free dense SHAPE representation -- the same
    footing as the scale-free scalar ratios."""
    M = load_model()
    T = M["v_template"] - M["v_template"].mean(0)
    T /= np.linalg.norm(T)
    X, labels = [], []
    for f in sorted(glob.glob(os.path.join(REPO, "diagnostics/moonshot/runs/M4_W*/Stage_3_deform_fine.npz"))):
        _o, can, lb = forward_pair(f)          # canonical = joint rotation zeroed
        for v in can:
            v = v - v.mean(0)
            v = v / np.linalg.norm(v)          # unit size -> scale-free
            X.append((v @ kabsch(v, T)).ravel())
        labels += lb
    return np.asarray(X), labels, M


def rank_eta_sq(vals, groups):
    r = stats.rankdata(vals)
    keys = sorted(set(groups))
    gm = r.mean()
    idx = {k: [i for i, g in enumerate(groups) if g == k] for k in keys}
    ss_b = sum(len(idx[k]) * (r[idx[k]].mean() - gm) ** 2 for k in keys)
    return float(ss_b / ((r - gm) ** 2).sum())


def main():
    d = json.load(open(os.path.join(OUT, "m4a_admissible_set.json")))
    meta = {r["specimen"]: r for r in d["per_specimen"]}

    X, labels, M = dense_shape()
    print(f"[m4e] dense shape matrix {X.shape} (canonical pose, Procrustes, unit scale)")
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = U[:, :NPC] * S[:NPC]                    # specimen scores on top PCs
    ev = (S ** 2 / (S ** 2).sum())[:NPC]
    print(f"[m4e] top {NPC} PCs explain {100*ev.sum():.1f}% of dense shape variance")

    res = {"preregistration": "PREREGISTRATION_M4_corpus_analysis.md §6",
           "dense": dict(n=int(X.shape[0]), n_pcs=NPC, var_explained=float(ev.sum())),
           "traits": {}}
    rng = np.random.default_rng(SEED)

    for name, col, gate, tier in (("HW/WL", "HW_WL", "HW_admissible", "primary"),
                                  ("HL/WL", "HL_WL", "HL_admissible", "secondary")):
        keep = [i for i, l in enumerate(labels) if meta[l][gate]]
        Pk = P[keep]
        y = np.array([meta[labels[i]][col] for i in keep])
        gen = [meta[labels[i]]["genus"] for i in keep]
        cnt = collections.Counter(gen)
        big = np.array([cnt[g] >= 5 for g in gen])

        # 1. how well does dense shape predict the scalar? (upper bound on shared information)
        b, *_ = np.linalg.lstsq(np.c_[Pk, np.ones(len(Pk))], y, rcond=None)
        pred = np.c_[Pk, np.ones(len(Pk))] @ b
        r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()

        # 2. the trait-aligned DIRECTION in dense shape space, and its genus structure
        w = b[:NPC] / np.linalg.norm(b[:NPC])
        s = Pk @ w
        eta_dir = rank_eta_sq(s[big], [g for g, k in zip(gen, big) if k])
        eta_trait = rank_eta_sq(y[big], [g for g, k in zip(gen, big) if k])

        # 3. null: random directions in the same PC space
        null = np.array([rank_eta_sq((Pk @ (lambda v: v / np.linalg.norm(v))(rng.normal(size=NPC)))[big],
                                     [g for g, k in zip(gen, big) if k]) for _ in range(2000)])
        p_dir = float((null >= eta_dir).mean())

        # 4. total genus structure available in the dense space (all PCs, first PC-wise then summed)
        eta_pcs = np.array([rank_eta_sq(Pk[big, j], [g for g, k in zip(gen, big) if k])
                            for j in range(NPC)])

        res["traits"][name] = dict(
            tier=tier, n=len(keep), n_tested=int(big.sum()),
            dense_predicts_trait_R2=float(r2),
            eta_sq_trait=eta_trait, eta_sq_trait_aligned_direction=eta_dir,
            null_random_direction=dict(median=float(np.median(null)), q95=float(np.quantile(null, .95))),
            p_vs_random_direction=p_dir,
            best_single_pc=dict(pc=int(eta_pcs.argmax()) + 1, eta_sq=float(eta_pcs.max())),
            mean_eta_top10_pcs=float(eta_pcs[:10].mean()))
        print(f"\n[m4e] {name} ({tier})  n={len(keep)}, {int(big.sum())} in genera n>=5")
        print(f"        dense shape predicts the scalar : R^2 = {r2:.3f}")
        print(f"        genus eta^2 of the scalar itself: {eta_trait:.3f}")
        print(f"        genus eta^2 of its dense-aligned direction: {eta_dir:.3f}")
        print(f"        random-direction null: median {np.median(null):.3f}, 95th {np.quantile(null,.95):.3f}"
              f"   -> p = {p_dir:.4f}")
        print(f"        best single dense PC: PC{eta_pcs.argmax()+1} eta^2={eta_pcs.max():.3f};"
              f"  mean of top-10 PCs {eta_pcs[:10].mean():.3f}")

    p = os.path.join(OUT, "m4e_bridge.json")
    json.dump(res, open(p, "w"), indent=1)
    print(f"\n[m4e] wrote {p}")


if __name__ == "__main__":
    main()
