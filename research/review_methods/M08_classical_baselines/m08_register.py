"""M08 registration wrappers producing fitter correspondence targets (PREREGISTRATION.md).

Every method returns, for one specimen, (targets (V,3), mask (V,)) in the fitter's normalised frame
(vertex mean, max abs coordinate of the scan), V = template vertices: the position on the scan that the
method assigns to each template vertex. Source = template at rest normalised the same way.

BCPD / GBCPD call the vendored official binary (third_party/bcpd/bcpd) through text files in a temp dir.
Verified output format (manual run 2026-10-07): `e` = one row per TARGET point [n, matched source m,
probability], `c` = per target point inlier flag. Per-template-vertex target = mean of the inlier scan
points matched to that vertex, mask = vertices with >= 1 match: the same aggregation as the CSE targets.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
BCPD = os.path.join(HERE, "third_party", "bcpd", "bcpd")


def normalise(V):
    c = V.mean(0)
    s = np.abs(V - c).max()
    return (V - c) / s


def template(dd):
    return normalise(np.asarray(dd["v_template"], np.float64)), np.asarray(dd["f"]).astype(np.int64)


def sample_surface(V, F, n, rng):
    a = np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    fi = rng.choice(len(F), n, p=a / a.sum())
    b = rng.dirichlet([1, 1, 1], n)
    return (b[:, :, None] * V[F[fi]]).sum(1)


def run_bcpd(Y, X, F_src=None, beta=1.0, lam=10.0, omega=0.1, tau=None, n_loop=500, extra=()):
    """Y: source (template, M x 3); X: target points (N x 3). Returns matched points (M,3), mask (M,)."""
    with tempfile.TemporaryDirectory() as d:
        np.savetxt(os.path.join(d, "X.txt"), X, fmt="%.8f", delimiter=",")
        np.savetxt(os.path.join(d, "Y.txt"), Y, fmt="%.8f", delimiter=",")
        cmd = [BCPD, "-x", os.path.join(d, "X.txt"), "-y", os.path.join(d, "Y.txt"),
               f"-w{omega}", f"-b{beta}", f"-l{lam}", "-g0.1", "-ux", f"-n{n_loop}", "-q",
               "-o", os.path.join(d, "out_"), "-sec"]
        if tau is not None:
            np.savetxt(os.path.join(d, "tri.txt"), F_src + 1, fmt="%d", delimiter="\t")   # README: tab-separated
            # rank/Nystrom/KD-tree: the author's published full-body GBCPD settings (demo-python/bcpd/
            # gbcpd_plusplus.py, male/female cases), not tuned here (DEVIATIONS D8)
            cmd += ["-G", f"geodesic,{tau},{os.path.join(d, 'tri.txt')}", "-K300", "-J300", "-p"]
        cmd += list(extra)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"bcpd failed ({r.returncode}): {r.stderr[-500:]}")
        e = np.loadtxt(os.path.join(d, "out_e.txt"), skiprows=1)      # [n, m, prob] per TARGET point (1-based)
        c = np.loadtxt(os.path.join(d, "out_c.txt"), skiprows=1)      # [n, inlier]  per TARGET point
    assert len(e) == len(X) and len(c) == len(X), "bcpd outputs are not one row per target point"
    n = e[:, 0].astype(int) - 1
    msrc = e[:, 1].astype(int) - 1
    keep = c[n, 1].astype(bool)                                        # inliers only
    acc = np.zeros_like(Y); cnt = np.zeros(len(Y), np.int64)
    np.add.at(acc, msrc[keep], X[n[keep]]); np.add.at(cnt, msrc[keep], 1)
    mask = cnt >= 1                                                    # same aggregation as the CSE targets
    T = np.zeros_like(Y); T[mask] = acc[mask] / cnt[mask, None]
    return T, mask


def run_nicp(Yv, F, Xv, XF, scale=1.0):
    """Amberg NICP (trimesh reference implementation) after rigid ICP with scale."""
    import trimesh
    from trimesh.registration import icp, nricp_amberg
    tgt = trimesh.Trimesh(Xv, XF, process=False)
    src = trimesh.Trimesh(Yv, F, process=False)
    pts = src.sample(4000)
    M, _, _ = icp(pts, tgt, scale=True, max_iterations=50)
    src.apply_transform(M)
    steps = [[0.01 * scale, 10, 0.5, 10], [0.02 * scale, 5, 0.5, 10],
             [0.03 * scale, 2.5, 0.5, 10], [0.01 * scale, 0, 0.0, 10]]   # trimesh's documented default, scaled
    out = nricp_amberg(src, tgt, steps=steps)
    closest, _, _ = trimesh.proximity.closest_point(tgt, out)
    return closest, np.ones(len(out), bool)


def smoke_test():
    import anatomy_proxy as ap
    from scipy.spatial import cKDTree
    dd = ap.load_dd()
    Y, F = template(dd)
    z = np.load(os.path.join(REPO, "diagnostics/moonshot/synth_power48/ground_truth.npz"))
    Xv = normalise(z["verts"][0].astype(np.float64))
    X = sample_surface(Xv, F, 8000, np.random.default_rng(0))
    vlab = ap.template_vertex_labels(dd)
    legc = ap.coarse_vec(vlab, "leg")
    tree = cKDTree(Xv)
    for name, fn in (("BCPD", lambda: run_bcpd(Y, X, beta=1.0, lam=10.0)),
                     ("GBCPD", lambda: run_bcpd(Y, X, F_src=F, beta=1.0, lam=10.0, tau=0.2)),
                     ("NICP", lambda: run_nicp(Y, F, Xv, F))):
        import time
        t = time.time()
        T, m = fn()
        d_on, _ = cKDTree(X).query(T[m])
        _, nn = tree.query(T)                            # true identity of the scan point each target hits
        leg = legc != "other"
        acc = (legc[nn][leg & m] == legc[leg & m]).mean()
        print(f"[smoke] {name}: {time.time() - t:.0f}s  mask {m.mean():.2f}  target-to-scan median "
              f"{np.median(d_on):.4f}  leg-correct {acc:.3f}", flush=True)


if __name__ == "__main__":
    smoke_test()
