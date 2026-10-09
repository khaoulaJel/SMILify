"""D2a -- is the distal leg literally starved of chamfer supervision by area-weighted sampling?

MOTIVATION, found while reading `fitter_3d/trainer_hierarchical.py` for this task (not assumed):
its own docstring (anatomical_groups, lines ~103-117) already computes, on the TEMPLATE, that
tarsus+pretarsus hold 2.4% of a leg chain's surface area, so at the fitter's `n_sample=8000`
(`HierarchicalStage.run`, trainer_hierarchical.py:567/578) a leg's ~500 target-point budget is
expected to put ~12.6 points on the tarsus and ~0.3 on the pretarsus -- and target points are
resampled only every `reassign_every=50` iterations, not every step, so a bad draw persists for
50 iterations at a time. `_partitioned_chamfer` (trainer_hierarchical.py:417-476) skips a GROUP's
term entirely below 10 target points (line 437/443), but that check is at whole-LEG granularity
in every run so far (`--split_distal` was never passed) -- a whole leg clears 10 easily, so the
skip never fires; what actually starves is the distal SEGMENT's share of the samples inside that
leg's un-split pool, which no code path checks or reports.

METHOD: sample area-weighted points from each specimen's OWN target mesh (same algorithm as
`pytorch3d.ops.sample_points_from_meshes`: face chosen with probability proportional to area,
uniform barycentric coordinates within the chosen face -- reimplemented in plain numpy here so
this needs no GPU and no fit output, only the corpus .obj files themselves), at the exact
`n_sample=8000` the fitter uses, repeated many times per specimen to get the empirical
distribution of "how many samples land on the tarsus+pretarsus region" and "how many on the
pretarsus alone" -- exactly what `_partitioned_chamfer` sees on a given 50-iteration window.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MOON = os.path.join(REPO, "diagnostics", "moonshot")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "diagnostics", "absolute_scale"))
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import measure as ms  # noqa: E402
from pytorch3d.io import load_obj  # noqa: E402

N_SAMPLE = 8000  # HierarchicalStage default, trainer_hierarchical.py:311
N_DRAWS = 200  # independent resamples per specimen, to get a distribution not one number


def face_group(M):
    """Group id (by dominant-weight majority vote of its 3 vertices) for every template face."""
    faces = np.asarray(M["dd"]["f"])
    dom = M["dominant"]
    jn = M["jnames"]

    def leg_group(j):
        n = jn[j]
        if not n.startswith("l_"):
            return None
        bits = n.split("_")
        k, seg, side = bits[1], bits[2], bits[-1]
        half = "d" if seg in ("ti", "ta", "pt") else "p"
        return f"l{k}{half}_{side}"

    vgroup = np.array([leg_group(j) or "other" for j in dom])
    fgroup = vgroup[faces]  # (F, 3)
    # majority vote per face; ties broken by first vertex (faces are tiny, ties rare and harmless)
    out = np.empty(faces.shape[0], dtype=object)
    for i in range(faces.shape[0]):
        vals, counts = np.unique(fgroup[i], return_counts=True)
        out[i] = vals[np.argmax(counts)]
    return out


def face_group_by_segment(M):
    """Like `face_group`, but keyed by the actual leg SEGMENT (co/tr/fe/ti/ta/pt), not the
    proximal/distal half -- needed to see whether starvation is uniform across the 'distal'
    half or concentrated on specific segments within it (e.g. pretarsus vs tibia)."""
    faces = np.asarray(M["dd"]["f"])
    dom = M["dominant"]
    jn = M["jnames"]

    def seg_group(j):
        n = jn[j]
        if not n.startswith("l_"):
            return None
        bits = n.split("_")
        k, seg, side = bits[1], bits[2], bits[-1]
        return f"l{k}_{seg}_{side}"

    vgroup = np.array([seg_group(j) or "other" for j in dom])
    fgroup = vgroup[faces]
    out = np.empty(faces.shape[0], dtype=object)
    for i in range(faces.shape[0]):
        vals, counts = np.unique(fgroup[i], return_counts=True)
        out[i] = vals[np.argmax(counts)]
    return out


def face_areas(verts, faces):
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)


def sample_counts(verts, faces, fgroup, groups_of_interest, n_sample, n_draws, rng):
    """Returns {group: (n_draws,) array of sample counts landing in that group}."""
    areas = face_areas(verts, faces)
    p = areas / areas.sum()
    out = {g: np.zeros(n_draws, dtype=int) for g in groups_of_interest}
    for d in range(n_draws):
        face_idx = rng.choice(len(p), size=n_sample, p=p)
        chosen = fgroup[face_idx]
        for g in groups_of_interest:
            out[g][d] = int((chosen == g).sum())
    return out


def main():
    M = ms.load_model()
    fgroup = face_group(M)
    faces = np.asarray(M["dd"]["f"])
    rng = np.random.default_rng(0)

    legs = [(k, s) for k in (1, 2, 3) for s in ("r", "l")]
    distal_groups = [f"l{k}d_{s}" for k, s in legs]
    proximal_groups = [f"l{k}p_{s}" for k, s in legs]

    corpora = [
        ("clean_pose0", "synth_clean_pose0"),
        ("clean_pose05", "synth_clean_pose05"),
        ("clean_pose10", "synth_clean_pose10"),
        ("clean_pose15", "synth_clean_pose15"),
        ("clean_pose20", "synth_clean_pose20"),
        ("clean_pose25", "synth_clean"),
        ("clean_pose35", "synth_clean_pose35"),
    ]

    print(f"n_sample={N_SAMPLE}, {N_DRAWS} independent draws per specimen\n")
    print(f"{'condition':<14}{'specimen':<12}{'distal(median)':>16}{'distal(p10)':>13}"
          f"{'P(distal==0)':>14}{'proximal(median)':>18}")
    results = {}
    for label, corpus in corpora:
        cdir = os.path.join(MOON, corpus)
        if not os.path.isdir(cdir):
            print(f"[skip] {corpus} not found")
            continue
        obj_files = sorted(f for f in os.listdir(cdir) if f.endswith(".obj"))
        cond_rows = []
        for fn in obj_files:
            ov, of, _ = load_obj(os.path.join(cdir, fn), load_textures=False)
            verts = ov.numpy()
            f_idx = of.verts_idx.numpy()
            # each specimen's own topology is the template's (no remesh in synth_clean corpora
            # -- confirm face COUNT matches template so `fgroup` (template-indexed) applies)
            assert f_idx.shape == faces.shape, f"topology mismatch in {fn}: {f_idx.shape} vs {faces.shape}"
            per_leg = sample_counts(verts, f_idx, fgroup, distal_groups + proximal_groups, N_SAMPLE, N_DRAWS, rng)
            all_distal = np.stack([per_leg[g] for g in distal_groups])  # (6legs, n_draws)
            all_prox = np.stack([per_leg[g] for g in proximal_groups])
            row = dict(
                specimen=fn,
                distal_median=float(np.median(all_distal)),
                distal_p10=float(np.percentile(all_distal, 10)),
                p_distal_zero=float((all_distal == 0).mean()),
                proximal_median=float(np.median(all_prox)),
                per_leg_distal_median={g: float(np.median(per_leg[g])) for g in distal_groups},
            )
            cond_rows.append(row)
            print(f"{label:<14}{fn:<12}{row['distal_median']:>16.1f}{row['distal_p10']:>13.1f}"
                  f"{row['p_distal_zero']:>14.3f}{row['proximal_median']:>18.1f}")
        results[label] = cond_rows

    # ---- per-SEGMENT breakdown (co/tr/fe/ti/ta/pt), pooled over the clean_pose25 corpus ----
    # `distal_groups` above (ti+ta+pt pooled) turned out NOT starved -- see whether that pooled
    # number is hiding a starved pretarsus/tarsus behind a well-sampled tibia.
    print("\n=== per-SEGMENT sample counts, pooled over all 12 clean_pose25 specimens x 6 legs ===")
    fgroup_seg = face_group_by_segment(M)
    seg_names = ("co", "tr", "fe", "ti", "ta", "pt")
    seg_groups = [f"l{k}_{seg}_{s}" for k, s in legs for seg in seg_names]
    cdir = os.path.join(MOON, "synth_clean")
    pooled = {seg: [] for seg in seg_names}
    for fn in sorted(f for f in os.listdir(cdir) if f.endswith(".obj")):
        ov, of, _ = load_obj(os.path.join(cdir, fn), load_textures=False)
        verts, f_idx = ov.numpy(), of.verts_idx.numpy()
        counts = sample_counts(verts, f_idx, fgroup_seg, seg_groups, N_SAMPLE, N_DRAWS, rng)
        for seg in seg_names:
            for k, s in legs:
                pooled[seg].append(counts[f"l{k}_{seg}_{s}"])
    print(f"{'segment':<10}{'median':>10}{'p10':>10}{'P(==0)':>10}{'P(<10)':>10}")
    seg_summary = {}
    for seg in seg_names:
        arr = np.concatenate(pooled[seg])
        seg_summary[seg] = dict(
            median=float(np.median(arr)), p10=float(np.percentile(arr, 10)),
            p_zero=float((arr == 0).mean()), p_lt10=float((arr < 10).mean()),
        )
        print(f"{seg:<10}{seg_summary[seg]['median']:>10.1f}{seg_summary[seg]['p10']:>10.1f}"
              f"{seg_summary[seg]['p_zero']:>10.3f}{seg_summary[seg]['p_lt10']:>10.3f}")

    import json
    OUT = os.path.join(HERE, "out")
    os.makedirs(OUT, exist_ok=True)
    json.dump(
        dict(by_condition=results, by_segment_pooled_clean_pose25=seg_summary),
        open(os.path.join(OUT, "d2a_sample_starvation.json"), "w"),
        indent=1,
    )
    print(f"\nwrote {OUT}/d2a_sample_starvation.json")


if __name__ == "__main__":
    main()
