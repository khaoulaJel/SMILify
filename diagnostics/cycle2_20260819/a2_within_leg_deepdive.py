"""Cycle2 A2: within-leg mismatch deep-dive (CPU-only, no new fitting jobs).

Extends diagnostics/registration_failure/probe_d2b_correspondence_audit.py's `audit_run` (reused
via import, not modified) to record the FULL confusion matrix (true segment -> matched segment),
not just an aggregate mismatch fraction per segment, across the same pose-sweep + damage
conditions that script already audits.

Answers, per the task list's A2 spec:
  - breakdown by specific segment (coxa/trochanter/femur/tibia/tarsus/pretarsus)
  - systematic vs random error pattern: is mismatch concentrated on the ADJACENT segment along
    the kinematic chain (ti<->ta<->pt), or spread uniformly across all other segments/legs?
  - correlation with target sampling density per segment (reuses
    diagnostics/registration_failure/out/d2a_sample_starvation.json, already computed)
"""
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
REG_FAIL = os.path.join(REPO, "diagnostics", "registration_failure")
MOON = os.path.join(REPO, "diagnostics", "moonshot")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "diagnostics", "absolute_scale"))
sys.path.insert(0, REG_FAIL)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import measure as ms  # noqa: E402
from pytorch3d.io import load_obj  # noqa: E402
from pytorch3d.ops import knn_points  # noqa: E402
from fitter_3d.trainer_hierarchical import TargetPartition, vertex_groups  # noqa: E402
from probe_d2a_sample_starvation import face_areas, face_group_by_segment  # noqa: E402
from probe_d2b_correspondence_audit import true_face_labels, sample_with_true_labels  # noqa: E402

N_SAMPLE = 8000
DEVICE = "cpu"
SEGS = ("co", "tr", "fe", "ti", "ta", "pt")
CHAIN_ADJACENT = {"co": {"tr"}, "tr": {"co", "fe"}, "fe": {"tr", "ti"}, "ti": {"fe", "ta"},
                  "ta": {"ti", "pt"}, "pt": {"ta"}}

OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)


def confusion_for_run(run, corpus, M, vg, group_names, leg_of_face, seg_of_face, rng, n_specimens=None):
    d = np.load(os.path.join(MOON, "runs", run, "Stage_3_deform_fine.npz"))
    labels = [str(x) for x in d["labels"]]
    fitted = d["verts"].astype(np.float64)
    dom = M["dominant"]
    jn = M["jnames"]

    def vertex_seg(j):
        n = jn[j]
        return n.split("_")[2] if n.startswith("l_") else None

    vertex_true_seg = np.array([vertex_seg(j) for j in dom], dtype=object)
    vg_t = torch.as_tensor(vg, device=DEVICE)
    name2i = {n: i for i, n in enumerate(group_names)}

    # confusion[true_seg][matched_seg] = count, pooled over all specimens in this run
    confusion = {s: {s2: 0 for s2 in SEGS} for s in SEGS}
    n_specimens_used = 0

    for i, lab in enumerate(labels[: n_specimens or len(labels)]):
        stem = lab[:-4] if lab.endswith(".obj") else lab
        objp = os.path.join(MOON, corpus, f"{stem}.obj")
        if not os.path.isfile(objp):
            continue
        ov, of, _ = load_obj(objp, load_textures=False)
        verts_t, faces_t = ov.numpy(), of.verts_idx.numpy()
        pts, true_leg, true_seg = sample_with_true_labels(verts_t, faces_t, leg_of_face, seg_of_face, N_SAMPLE, rng)
        is_leg_pt = true_leg != None  # noqa: E711
        pts_t = torch.tensor(pts[is_leg_pt], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        true_leg_pt, true_seg_pt = true_leg[is_leg_pt], true_seg[is_leg_pt]

        fitted_i = torch.tensor(fitted[i], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        part = TargetPartition(vg_t, len(group_names), DEVICE)
        part.update(fitted_i, pts_t)
        assigned_name = np.array(group_names)[part.assign[0].cpu().numpy()]
        correct_leg_mask = assigned_name == true_leg_pt  # only within-leg-correct points count here

        for leg_name_u in np.unique(true_leg_pt):
            gid = name2i[leg_name_u]
            leg_vert_idx = np.where(vg == gid)[0]
            src = torch.tensor(fitted[i, leg_vert_idx], dtype=torch.float32).unsqueeze(0)
            mm = (true_leg_pt == leg_name_u) & correct_leg_mask
            if not mm.any():
                continue
            tgt = pts_t[0][mm].unsqueeze(0)
            nn_idx = knn_points(tgt, src, K=1).idx[0, :, 0].cpu().numpy()
            matched_seg = vertex_true_seg[leg_vert_idx[nn_idx]]
            true_seg_here = true_seg_pt[mm]
            for ts, ms_ in zip(true_seg_here, matched_seg):
                if ts in SEGS and ms_ in SEGS:
                    confusion[ts][ms_] += 1
        n_specimens_used += 1

    return confusion, n_specimens_used


def main():
    M = ms.load_model()
    weights, jnames, faces = M["weights"], M["jnames"], np.asarray(M["dd"]["f"])
    vg, group_names = vertex_groups(weights, jnames, split_distal=False)
    leg_of_face, seg_of_face = true_face_labels(M)
    rng = np.random.default_rng(2)

    conditions = [
        ("SYN_clean_pose0_w5", "synth_clean_pose0"),
        ("SYN_clean_pose05_w5", "synth_clean_pose05"),
        ("SYN_clean_pose10_w5", "synth_clean_pose10"),
        ("SYN_clean_pose15_w5", "synth_clean_pose15"),
        ("SYN_clean_pose20_w5", "synth_clean_pose20"),
        ("SYN_clean_w5", "synth_clean"),
        ("SYN_clean_pose35_w5", "synth_clean_pose35"),
        ("SYN_clean_drop30_w5", "synth_clean_drop30"),
        ("SYN_clean_drop60_w5", "synth_clean_drop60"),
    ]

    report = {}
    pooled_confusion = {s: {s2: 0 for s2 in SEGS} for s in SEGS}
    for run, corpus in conditions:
        conf, n = confusion_for_run(run, corpus, M, vg, group_names, leg_of_face, seg_of_face, rng)
        report[run] = dict(corpus=corpus, n_specimens=n, confusion=conf)
        for ts in SEGS:
            for ms_ in SEGS:
                pooled_confusion[ts][ms_] += conf[ts][ms_]
        print(f"{run:<24} n={n}")

    # ---------------- systematic vs random: adjacent-chain fraction of all MISmatches ----------
    print(f"\n{'true_seg':<10}{'n_correct':>12}{'n_mismatch':>12}{'frac_adjacent':>16}{'frac_other':>14}")
    adjacency_summary = {}
    for ts in SEGS:
        row = pooled_confusion[ts]
        total = sum(row.values())
        correct = row[ts]
        mismatch = total - correct
        adj = sum(row[ms_] for ms_ in CHAIN_ADJACENT[ts])
        other = mismatch - adj
        frac_adj = adj / mismatch if mismatch > 0 else float("nan")
        frac_other = other / mismatch if mismatch > 0 else float("nan")
        adjacency_summary[ts] = dict(n_total=total, n_correct=correct, n_mismatch=mismatch,
                                      n_adjacent=adj, n_other=other,
                                      frac_adjacent_of_mismatch=frac_adj, frac_other_of_mismatch=frac_other)
        print(f"{ts:<10}{correct:>12}{mismatch:>12}{frac_adj:>16.3f}{frac_other:>14.3f}")

    # ---------------- correlation with sampling density (reuse existing d2a output) ------------
    d2a = json.load(open(os.path.join(REG_FAIL, "out", "d2a_sample_starvation.json")))
    density = d2a["by_segment_pooled_clean_pose25"]
    mismatch_rate_pose25 = {}
    conf25 = report["SYN_clean_w5"]["confusion"]
    for ts in SEGS:
        row = conf25[ts]
        total = sum(row.values())
        mismatch_rate_pose25[ts] = (total - row[ts]) / total if total > 0 else float("nan")

    segs_with_data = [s for s in SEGS if np.isfinite(mismatch_rate_pose25.get(s, np.nan))]
    x = [density[s]["median"] for s in segs_with_data]
    y = [mismatch_rate_pose25[s] for s in segs_with_data]
    rho, p = (spearmanr(x, y) if len(x) >= 3 else (None, None))
    print(f"\nSpearman(median target samples per segment, within-leg mismatch rate) at pose25: "
          f"rho={rho} p={p} (n={len(x)} segments)")
    print(f"{'segment':<10}{'median_samples':>16}{'mismatch_rate':>16}")
    for s in segs_with_data:
        print(f"{s:<10}{density[s]['median']:>16.1f}{mismatch_rate_pose25[s]:>16.3f}")

    out = dict(
        per_condition=report,
        pooled_confusion_matrix=pooled_confusion,
        adjacency_summary=adjacency_summary,
        sampling_density_correlation=dict(
            rho=rho, p=p, segments=segs_with_data,
            median_samples=x, mismatch_rate=y,
        ),
        interpretation=(
            "Systematic (chain-adjacent) confusion supported for a segment if frac_adjacent > 0.5 "
            "(mismatches concentrate on the neighbor along the kinematic chain, not spread "
            "uniformly across all other segments/legs -- consistent with a geodesic/rank-order "
            "correspondence ambiguity that B5's geodesic-rank channel is designed to fix). "
            "Otherwise mismatch looks closer to noise/debris, which B5 would not fix."
        ),
    )
    with open(os.path.join(OUT, "a2_within_leg_deepdive.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\nwrote {os.path.join(OUT, 'a2_within_leg_deepdive.json')}")


if __name__ == "__main__":
    main()
