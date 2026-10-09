"""Retroactive check of CYCLE2_REPORT.md's B6/B7 damage-corpus claim ("gnc_legonly_topofree is
now the unqualified recommendation for both clean and damage data"), using the same protocol as
bootstrap_noise_floor.py: raw fits still on disk, an unambiguous named metric (not audit_run's
raw fields relabeled "R"), paired bootstrap, all available seeds.

DAMAGE-CORPUS WRINKLE: drop30/drop60 genuinely remove vertices and renumber faces, so the
face-index-based ground truth confusion.py/labels.py rely on elsewhere does NOT directly apply
(target face indices no longer match the template's). Fixed here by reconstructing the exact
vertex-drop/remap the corpus generator used (recovered from git history, commit c462a95c,
`diagnostics/appendage_evidence/{make_damage_corpus,candidates}.py` -- both fully deterministic,
no RNG, driven only by the frozen template geometry) and remapping labels through it before
reusing every existing function unchanged. The FITTED mesh side needs no such remap: SMAL always
outputs full template topology regardless of target damage.
"""

import os
import sys

import numpy as np
import torch
from pytorch3d.io import load_obj

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MOON = os.path.join(REPO, "diagnostics", "moonshot")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "diagnostics", "morphometrics"))
sys.path.insert(0, HERE)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

DAMAGE_RECOVERY = "/tmp/nao48500/login23-1_237870/claude-54434/-rwthfs-rz-cluster-home-nao48500-SMILify/d3035fe2-22d3-4316-bc65-8103b53c37ee/scratchpad/damage_recovery"
sys.path.insert(0, DAMAGE_RECOVERY)

import measure as ms  # noqa: E402
import labels as lb  # noqa: E402
import confusion as cf  # noqa: E402
import geodesic as geo  # noqa: E402
import bootstrap_noise_floor as bn  # noqa: E402
import candidates as cand  # noqa: E402

N_SAMPLE = 8000
DEVICE = "cpu"


def keep_idx_for_frac(M, frac):
    """Global template vertex indices SURVIVING at this drop fraction, in the same order
    make_damage_corpus.py's `build_damaged_mesh` keeps them (ascending original index) -- so
    keep_idx[local_i] is local_i's ORIGINAL template vertex index."""
    cache = cand.build_geodesic_cache(M)
    drop_by_leg = cand.dropout_vertices(M, cache, frac)
    all_drop = np.concatenate(list(drop_by_leg.values()))
    V = len(M["dominant"])
    keep_mask = np.ones(V, dtype=bool)
    keep_mask[all_drop] = False
    return np.where(keep_mask)[0]


def per_specimen_stats_damaged(run, corpus, npz_path, keep_idx, vlabels, geo_lookup, rng):
    remapped_vlabels = {k: v[keep_idx] for k, v in vlabels.items()}

    d = np.load(npz_path)
    spec_labels = [str(x) for x in d["labels"]]
    fitted = d["verts"].astype(np.float64)
    n_verts_template = len(vlabels["region"])
    assert fitted.shape[1] == n_verts_template, "fitted mesh must be full template topology"

    out = []
    for i, spec_lab in enumerate(spec_labels):
        stem = spec_lab[:-4] if spec_lab.endswith(".obj") else spec_lab
        obj_path = os.path.join(MOON, corpus, f"{stem}.obj")
        ov, of, _ = load_obj(obj_path, load_textures=False)
        tgt_verts, tgt_faces = ov.numpy(), of.verts_idx.numpy()
        assert tgt_verts.shape[0] == len(keep_idx), (
            f"{run}/{stem}: damaged target has {tgt_verts.shape[0]} verts, expected {len(keep_idx)} "
            "-- keep_idx reconstruction does not match this specimen's actual topology"
        )
        damaged_face_lab = lb.face_labels(tgt_faces, remapped_vlabels)

        pts, point_lab = cf.sample_true_labeled_points(tgt_verts, tgt_faces, damaged_face_lab, N_SAMPLE, rng)
        pts_t = torch.tensor(pts, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        true_is_leg = point_lab["leg_id"] != None  # noqa: E711
        # true_vertex_idx from sample_true_labeled_points is a LOCAL (damaged-mesh) index --
        # convert to the GLOBAL template index every downstream function (geodesic lookup,
        # consistency check) expects.
        point_lab["true_vertex_idx"] = keep_idx[point_lab["true_vertex_idx"]]

        fitted_i = torch.tensor(fitted[i], dtype=torch.float32, device=DEVICE).unsqueeze(0)

        leg_acc_i, leg_pred, pts_leg, true_leg_sub = cf.leg_confusion(
            point_lab["leg_id"], true_is_leg, pts_t, fitted_i, vlabels
        )
        seg_acc_i = cf.within_leg_segment_confusion(
            point_lab["leg_seg"], true_is_leg, pts_leg, true_leg_sub, leg_pred, fitted_i, vlabels
        )
        true_vidx_here, matched_vidx, legs_here = cf.within_leg_segment_geodesic(
            point_lab["true_vertex_idx"], true_is_leg, pts_leg, true_leg_sub, leg_pred, fitted_i, vlabels
        )

        pt_ta, ti_ta = [], []
        if len(true_vidx_here):
            consistent = vlabels["leg_id"][true_vidx_here] == legs_here
            true_vidx_here, matched_vidx, legs_here = (
                true_vidx_here[consistent], matched_vidx[consistent], legs_here[consistent]
            )
        if len(true_vidx_here):
            dists = geo_lookup.dist(legs_here, true_vidx_here, matched_vidx)
            true_seg_of_point = vlabels["leg_seg"][true_vidx_here]
            matched_seg_of_point = vlabels["leg_seg"][matched_vidx]
            for ts, ms_, dd_ in zip(true_seg_of_point, matched_seg_of_point, dists):
                if np.isnan(dd_):
                    continue
                if ts == "pt" and ms_ == "ta":
                    pt_ta.append(dd_)
                elif ts == "ti" and ms_ == "ta":
                    ti_ta.append(dd_)

        out.append(dict(
            stem=stem, leg_M=leg_acc_i.M.copy(), seg_M=seg_acc_i.M.copy(),
            pt_ta=np.array(pt_ta, dtype=np.float64), ti_ta=np.array(ti_ta, dtype=np.float64),
        ))
    return out


CONDITIONS = [
    ("drop30", 0.30, [
        ("SYN_clean_drop30_gncmedium_w5", "gnc_medium", 0),
        ("SYN_clean_drop30_gncmedium_w5_seed1", "gnc_medium", 1),
        ("SYN_clean_drop30_gncmedium_w5_seed2", "gnc_medium", 2),
        ("SYN_clean_drop30_gnclegonlytopofree_w5", "topofree", 0),
        ("SYN_clean_drop30_gnclegonlytopofree_w5_seed1", "topofree", 1),
        ("SYN_clean_drop30_gnclegonlytopofree_w5_seed2", "topofree", 2),
    ]),
    ("drop60", 0.60, [
        ("SYN_clean_drop60_gncmedium_w5", "gnc_medium", 0),
        ("SYN_clean_drop60_gncmedium_w5_seed1", "gnc_medium", 1),
        ("SYN_clean_drop60_gncmedium_w5_seed2", "gnc_medium", 2),
        ("SYN_clean_drop60_gnclegonlytopofree_w5", "topofree", 0),
        ("SYN_clean_drop60_gnclegonlytopofree_w5_seed1", "topofree", 1),
        ("SYN_clean_drop60_gnclegonlytopofree_w5_seed2", "topofree", 2),
    ]),
]


def main():
    M = ms.load_model()
    vlabels = lb.vertex_labels(M["jnames"], M["dominant"])
    template_faces = np.asarray(M["dd"]["f"])
    v_template = np.asarray(M["v_template"], dtype=np.float64)
    geo_tables = geo.load_or_build_leg_tables(v_template, template_faces, vlabels)
    geo_lookup = geo.LegGeodesicLookup(geo_tables)

    for corpus_tag, frac, runs in CONDITIONS:
        corpus = f"synth_clean_{corpus_tag}"
        keep_idx = keep_idx_for_frac(M, frac)
        print(f"\n########## {corpus_tag} (frac={frac}, {len(keep_idx)} surviving verts) ##########")

        by_seed = {}
        for run, arm, seed in runs:
            npz_path = os.path.join(MOON, "runs", run, "Stage_3_deform_fine.npz")
            if not os.path.isfile(npz_path):
                print(f"[skip] {run} not found")
                continue
            rng = np.random.default_rng(1)
            specs = per_specimen_stats_damaged(run, corpus, npz_path, keep_idx, vlabels, geo_lookup, rng)
            by_seed.setdefault(seed, {})[arm] = specs
            pe = bn.pooled_stat(specs)
            print(f"  {run:<42}leg_acc={pe['leg_acc']:.3f}  seg_acc={pe['seg_acc']:.3f}  "
                  f"pt_ta={pe['pt_ta_mean']:.4f}  ti_ta={pe['ti_ta_mean']:.4f}")

        print(f"\n  -- paired bootstrap, topofree - gnc_medium, per seed --")
        for seed, arms in sorted(by_seed.items()):
            if "gnc_medium" not in arms or "topofree" not in arms:
                continue
            boot = bn.bootstrap(arms, n_boot=1000, seed=0)
            pe = {a: bn.pooled_stat(s) for a, s in arms.items()}
            print(f"  seed {seed}:")
            for k in ("leg_acc", "seg_acc", "pt_ta_mean", "ti_ta_mean"):
                diff = boot["topofree"][k] - boot["gnc_medium"][k]
                s = bn.summarize(diff)
                pd = pe["topofree"][k] - pe["gnc_medium"][k]
                sig = "***" if (s["ci95"][0] > 0 or s["ci95"][1] < 0) else ""
                print(f"    {k:<14} point_diff={pd:+.4f}  95% CI=({s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}) {sig}")


if __name__ == "__main__":
    main()
