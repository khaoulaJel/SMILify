"""Q3b S1: dense ground-truth correspondence targets for P48, with coherent corruption of the real type.

Output format is what `--cse_correspondence_from` consumes: names, verts (N,V,3) in the fitter's
normalised target frame, mask (N,V). Frame = fitter_3d/utils.py:load_meshes (centre by vertex mean,
divide by max abs coordinate), computed from the P48 .obj, which equals ground_truth.npz exactly
(checked: max |d| = 0.0).

Corruption (DEVIATIONS D2): a fraction rho of the 24 (leg, segment) units -- segments tr, fe, ti,
ta+pt on six legs -- is redirected as a whole to the adjacent same-side leg (1->2, 3->2, 2->1 or 3 at
random). A vertex v of segment g on leg k maps to the vertex of segment g on leg k' nearest to
v's rest position translated by (J_k'g - J_kg) (proximal-joint alignment in the rest template); its
target becomes that vertex's ground-truth posed position.

Probes written alongside: the fraction of leg vertices actually corrupted per rho, and the mean
displacement of corrupted targets (as % of body-axis length), so "rho" is a measured quantity.
"""
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import anatomy_proxy as ap  # noqa: E402

P48 = os.path.join(REPO, "diagnostics/moonshot/synth_power48")
OUT = "/hpcwork/nao48500/review_methods/Q3b_S1"
UNITS_SEG = ("tr", "fe", "ti", "ta")          # pt folded into ta


def seg_unit(name):
    n = str(name)
    if not n.startswith("l_"):
        return None
    b = n.split("_")
    s = "ta" if b[2] == "pt" else b[2]
    return (int(b[1]), b[-1], s) if s in UNITS_SEG else None


def main():
    os.makedirs(OUT, exist_ok=True)
    dd = ap.load_dd()
    names, _ = ap.joint_tree(dd)
    vlab = ap.template_vertex_labels(dd)
    vt = np.asarray(dd["v_template"])
    Jr = np.asarray(dd["J"])
    unit_of = np.array([seg_unit(n) for n in vlab], dtype=object)
    units = sorted({u for u in unit_of if u is not None})
    assert len(units) == 24, len(units)
    members = {u: np.where(np.array([x == u for x in unit_of]))[0] for u in units}

    # vertex maps between adjacent legs, per segment and side (rest template, joint-aligned)
    vmap = {}
    for (k, s, g) in units:
        for k2 in (k - 1, k + 1):
            if k2 < 1 or k2 > 3:
                continue
            seg_name = "ta" if g == "ta" else g
            j1 = names.index(f"l_{k}_{seg_name}_{s}")
            j2 = names.index(f"l_{k2}_{seg_name}_{s}")
            src, dst = members[(k, s, g)], members[(k2, s, g)]
            moved = vt[src] - Jr[j1] + Jr[j2]
            _, nn = cKDTree(vt[dst]).query(moved)
            vmap[((k, s, g), k2)] = dst[nn]

    z = np.load(os.path.join(P48, "ground_truth.npz"))
    gtv = z["verts"].astype(np.float64)
    stems = [str(x) for x in z["names"]]
    probes = {}
    for rho in (0.0, 0.2, 0.4):
        rng = np.random.default_rng(int(rho * 100) + 7)
        T = np.empty_like(gtv)
        frac, disp = [], []
        for i in range(len(gtv)):
            V = gtv[i]
            c = V.mean(0)
            sc = np.abs(V - c).max()
            tgt = V.copy()
            n_bad = int(round(rho * len(units)))
            bad = [units[j] for j in rng.choice(len(units), n_bad, replace=False)] if n_bad else []
            nv = 0
            for (k, s, g) in bad:
                k2 = {1: 2, 3: 2}.get(k) or int(rng.choice([1, 3]))
                src = members[(k, s, g)]
                tgt[src] = V[vmap[((k, s, g), k2)]]
                nv += len(src)
                disp.append(np.linalg.norm(tgt[src] - V[src], axis=1).mean() / sc)
            frac.append(nv / sum(len(m) for m in members.values()))
            T[i] = (tgt - c) / sc
        fn = os.path.join(OUT, f"targets_rho{rho:.1f}.npz")
        np.savez_compressed(fn, names=np.array(stems), verts=T.astype(np.float32),
                            mask=np.ones(T.shape[:2], bool), rho=rho)
        probes[rho] = dict(frac_leg_vertices_corrupted=float(np.mean(frac)),
                           mean_target_displacement_normalised=float(np.mean(disp)) if disp else 0.0)
        print(f"rho {rho}: corrupted {100*np.mean(frac):.1f}% of leg-unit vertices, "
              f"mean displacement {probes[rho]['mean_target_displacement_normalised']:.3f} (normalised units) -> {fn}")
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    json.dump(probes, open(os.path.join(HERE, "out", "S1_targets_probe.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
