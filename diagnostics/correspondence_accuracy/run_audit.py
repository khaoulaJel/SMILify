"""Correspondence-accuracy audit: true anatomical segment x matched anatomical segment,
confusion matrices at leg-level, within-leg-segment-level, and antenna-level, run against every
converged arm that shares the `synth_clean` ground-truth corpus (topology-identical to the
template by construction -- see `diagnostics/moonshot/SYNTHETIC_ROUNDTRIP.md`).

This is the metric the task asked to build BEFORE touching the correspondence mechanism itself:
independent of Chamfer/edge/etc., so "did the anatomical constraint actually fix
correspondence" can be checked directly instead of inferred from a smoother loss curve.

Run: `python diagnostics/correspondence_accuracy/run_audit.py` (needs the pytorch3d conda env).
"""

import json
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

import measure as ms  # noqa: E402
import labels as lb  # noqa: E402
import confusion as cf  # noqa: E402
import geodesic as geo  # noqa: E402

N_SAMPLE = 8000  # matches HierarchicalStage / optimise_moonshot's own chamfer sample size
DEVICE = "cpu"

# (run_name, corpus_dir, npz_path) -- every arm evaluated here shares the synth_clean corpus,
# so results are directly comparable to each other and are not confounded by different ground
# truth. npz_path=None defaults to the moonshot handoff layout (diagnostics/moonshot/runs/
# <run>/Stage_3_deform_fine.npz); pass an explicit path for arms audited straight off
# optimise_hierarchical.py's own output (which has no moonshot handoff at all).
RUNS = [
    ("SYN_clean_w5", "synth_clean", None),  # stock baseline, no GNC / anatomical constraint
    ("SYN_clean_pose25_gnclegonly_w5", "synth_clean", None),  # GNC, leg-only partitioning
    ("SYN_clean_pose25_splitdistal_w5", "synth_clean", None),  # --split_distal (Task 6 candidate)
    ("SYN_clean_pose25_gtinit_w5", "synth_clean", None),  # ground-truth pose init (Task 6 ceiling)
    ("SYN_clean_pose25_stratq05_w5", "synth_clean", None),  # stratified target sampling, q=0.05
    ("SYN_clean_pose25_stratq10_w5", "synth_clean", None),  # stratified target sampling, q=0.10
    ("SYN_clean_pose25_stratq20_w5", "synth_clean", None),  # stratified target sampling, q=0.20
    ("SYN_clean_pose25_c5protected_w5", "synth_clean", None),  # protected small-group robust kernel
    # Hierarchical-stage-only arms (no moonshot handoff -- see submit_split_segments_probe.sbatch).
    # MoonshotStage (fitter_3d/trainer_moonshot.py) has no TargetPartition/partitioned-chamfer
    # mechanism at all, so every *_w5 row above was already past the point where any partition
    # (including split_distal) could still act by the time Stage_3_deform_fine was captured --
    # these two are audited straight off HierarchicalStage's own final H3_deform output instead,
    # so the partitioned mechanism being changed is actually what's being measured.
    (
        "HIER_baseline",
        "synth_clean",
        os.path.join(HERE, "hier_runs", "baseline", "H3_deform.npz"),
    ),
    (
        "HIER_split_segments",
        "synth_clean",
        os.path.join(HERE, "hier_runs", "split_segments", "H3_deform.npz"),
    ),
    # Anatomical Initialization Ceiling Test (2026-08-20, diagnostics/anatomical_pose_init/).
    # Arm A/C reuse the SYN_clean_w5 / SYN_clean_pose25_gtinit_w5 rows above (same corpus, same
    # D1 recipe) rather than re-fitting -- ACI_zero and ACI_gt are deliberately NOT separate
    # rows, to avoid double-counting the same underlying data under two names.
    ("ACI_cheap", "synth_clean", None),  # cheap_anatomical init (simple_leg_heuristic.py)
    ("ACI_learned", "synth_clean", None),  # learned init (train_leg_pose_regressor.py, 25pc corpus)
    # Basin-structure follow-up (2026-08-20): magnitude-matched (~23deg) structured perturbations
    # of GT pose, see generate_structured_perturbation_init.py. Isolates error STRUCTURE from
    # error MAGNITUDE, since ACI_learned's advantage over zero-init came at near-identical
    # magnitude (22.79 vs 23.09deg).
    ("ACI_D_random", "synth_clean", None),
    ("ACI_E_proximal", "synth_clean", None),
    ("ACI_F_distal", "synth_clean", None),
    # Pose-error sweep (2026-08-20 follow-up, pose_noise_sweep.py): GT pose + fixed-magnitude
    # random-axis noise on the 36 leg joints only, everything else at exact GT. Level 0 is the
    # existing SYN_clean_pose25_gtinit_w5 row above, reused, not duplicated here.
    ("ACI_noise5", "synth_clean", None),
    ("ACI_noise10", "synth_clean", None),
    ("ACI_noise15", "synth_clean", None),
    ("ACI_noise20", "synth_clean", None),
    ("ACI_noise25", "synth_clean", None),
    ("ACI_noise30", "synth_clean", None),
    # Capacity-ceiling arm: GT-pose init + scale_cap=0.052 (D1_PROD.yaml), compared against the
    # existing SYN_clean_pose25_gtinit_w5 (GT-pose + free/uncapped segment scale, D1_low.yaml).
    ("ACI_gtcapacitycap", "synth_clean", None),
]


def audit_run(run, corpus, npz_path, template_faces, face_lab, vlabels, n_verts_template, geo_lookup, rng, n_specimens=None):
    d = np.load(npz_path)
    spec_labels = [str(x) for x in d["labels"]]
    fitted = d["verts"].astype(np.float64)
    assert fitted.shape[1] == n_verts_template, (
        f"{run}: fitted vertex count {fitted.shape[1]} != template {n_verts_template} -- "
        "fitted mesh does not share template topology, per-vertex labels invalid"
    )

    leg_acc = cf.ConfusionAccumulator(list(lb.LEGS))
    seg_acc = cf.ConfusionAccumulator(list(lb.LEG_SEGMENTS))
    ant_side_acc = cf.ConfusionAccumulator(list(lb.SIDES))
    ant_seg_acc = cf.ConfusionAccumulator(list(lb.ANTENNA_SEGMENTS))
    # geodesic error (mesh-surface distance, template units), pooled by TRUE segment and
    # separately by (true_seg, matched_seg) pair -- lets pt->ta / ti->ta be read off directly
    # instead of re-deriving them from the confusion matrix's row/col ordering.
    geo_by_true_seg = {s: [] for s in lb.LEG_SEGMENTS}
    geo_by_pair = {}  # (true_seg, matched_seg) -> list of distances
    per_specimen_leg_acc = []  # [(stem, leg_acc)] -- for trajectory plots (protocol section 4)

    n = n_specimens or len(spec_labels)
    for i, spec_lab in enumerate(spec_labels[:n]):
        stem = spec_lab[:-4] if spec_lab.endswith(".obj") else spec_lab
        obj_path = os.path.join(MOON, corpus, f"{stem}.obj")
        ov, of, _ = load_obj(obj_path, load_textures=False)
        tgt_verts, tgt_faces = ov.numpy(), of.verts_idx.numpy()
        assert tgt_faces.shape == template_faces.shape, (
            f"{run}/{stem}: target face topology {tgt_faces.shape} != template "
            f"{template_faces.shape} -- ground-truth face labels are invalid for this specimen"
        )

        pts, point_lab = cf.sample_true_labeled_points(tgt_verts, tgt_faces, face_lab, N_SAMPLE, rng)
        pts_t = torch.tensor(pts, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        # Gate on the leg_id/ant_seg field itself, not the independently-voted "region" field:
        # a face straddling two DIFFERENT legs (rare, only near adjacent coxae) can have a
        # 3-way tie among its vertices' leg_id ('l1_r' / 'l2_r' / None-from-a-body-vertex),
        # which majority-vote resolves to None even though "region" majority-votes to 'leg'
        # (2 of 3 vertices are leg vertices, just not the SAME leg). Gating on leg_id directly
        # drops these genuinely-ambiguous boundary points instead of assigning them a
        # meaningless true label.
        true_is_leg = point_lab["leg_id"] != None  # noqa: E711
        true_is_ant = point_lab["ant_seg"] != None  # noqa: E711

        fitted_i = torch.tensor(fitted[i], dtype=torch.float32, device=DEVICE).unsqueeze(0)

        leg_acc_i, leg_pred, pts_leg, true_leg_sub = cf.leg_confusion(
            point_lab["leg_id"], true_is_leg, pts_t, fitted_i, vlabels
        )
        seg_acc_i = cf.within_leg_segment_confusion(
            point_lab["leg_seg"], true_is_leg, pts_leg, true_leg_sub, leg_pred, fitted_i, vlabels
        )
        side_acc_i, antseg_acc_i = cf.antenna_confusion(
            point_lab["ant_side"], point_lab["ant_seg"], true_is_ant, pts_t, fitted_i, vlabels
        )
        true_vidx_here, matched_vidx, legs_here = cf.within_leg_segment_geodesic(
            point_lab["true_vertex_idx"], true_is_leg, pts_leg, true_leg_sub, leg_pred, fitted_i, vlabels
        )

        per_specimen_leg_acc.append((stem, leg_acc_i.as_dict()["accuracy"]))

        leg_acc.M += leg_acc_i.M
        leg_acc.n_unmapped_true += leg_acc_i.n_unmapped_true
        seg_acc.M += seg_acc_i.M
        ant_side_acc.M += side_acc_i.M
        ant_seg_acc.M += antseg_acc_i.M

        if len(true_vidx_here):
            # true_vertex_idx is the SAMPLED POINT's own dominant-barycentric-weight corner,
            # computed independently of the face-level leg_id majority vote true_is_leg/
            # legs_here rely on -- on a boundary face they can disagree (majority vote says
            # 'leg', the specific point's nearest corner happens to be the one non-leg vertex),
            # which would look up a vertex absent from that leg's geodesic table. Same
            # genuinely-ambiguous-boundary-point situation as the leg_id/region gating earlier
            # in this function; drop the inconsistent few rather than crash or mislabel.
            consistent = vlabels["leg_id"][true_vidx_here] == legs_here
            true_vidx_here, matched_vidx, legs_here = (
                true_vidx_here[consistent], matched_vidx[consistent], legs_here[consistent]
            )
        if len(true_vidx_here):
            dists = geo_lookup.dist(legs_here, true_vidx_here, matched_vidx)
            true_seg_of_point = vlabels["leg_seg"][true_vidx_here]
            matched_seg_of_point = vlabels["leg_seg"][matched_vidx]
            for ts, ms_, dd_ in zip(true_seg_of_point, matched_seg_of_point, dists):
                if ts is None or np.isnan(dd_):
                    continue
                geo_by_true_seg[ts].append(dd_)
                geo_by_pair.setdefault((ts, ms_), []).append(dd_)

    def _stats(xs):
        xs = np.asarray(xs, dtype=np.float64)
        if len(xs) == 0:
            return dict(n=0, mean=float("nan"), median=float("nan"))
        return dict(n=int(len(xs)), mean=float(xs.mean()), median=float(np.median(xs)))

    geodesic_summary = dict(
        by_true_segment={s: _stats(v) for s, v in geo_by_true_seg.items()},
        by_pair={f"{t}->{p}": _stats(v) for (t, p), v in geo_by_pair.items()},
    )

    return dict(
        run=run,
        corpus=corpus,
        n_specimens=n,
        leg_confusion=leg_acc.as_dict(),
        within_leg_segment_confusion=seg_acc.as_dict(),
        antenna_side_confusion=ant_side_acc.as_dict(),
        antenna_segment_confusion=ant_seg_acc.as_dict(),
        geodesic=geodesic_summary,
        per_specimen_leg_acc=per_specimen_leg_acc,
    )


def main():
    M = ms.load_model()
    template_faces = np.asarray(M["dd"]["f"])
    vlabels = lb.vertex_labels(M["jnames"], M["dominant"])
    face_lab = lb.face_labels(template_faces, vlabels)
    n_verts_template = len(M["dominant"])
    rng = np.random.default_rng(1)

    v_template = np.asarray(M["v_template"], dtype=np.float64)
    geo_tables = geo.load_or_build_leg_tables(v_template, template_faces, vlabels)
    geo_lookup = geo.LegGeodesicLookup(geo_tables)

    results = []
    for run, corpus, npz_path in RUNS:
        if npz_path is None:
            npz_path = os.path.join(MOON, "runs", run, "Stage_3_deform_fine.npz")
        if not os.path.isfile(npz_path):
            print(f"[skip] {run}: {npz_path} not found")
            continue
        print(f"[run] {run} (corpus={corpus}) ...")
        r = audit_run(run, corpus, npz_path, template_faces, face_lab, vlabels, n_verts_template, geo_lookup, rng)
        results.append(r)

    print(f"\n{'run':<34}{'leg_acc':>9}{'xleg_err':>10}{'seg_acc':>9}{'ant_side':>10}{'ant_seg':>9}")
    for r in results:
        leg_acc = r["leg_confusion"]["accuracy"]
        seg_acc = r["within_leg_segment_confusion"]["accuracy"]
        side_acc = r["antenna_side_confusion"]["accuracy"]
        antseg_acc = r["antenna_segment_confusion"]["accuracy"]
        print(f"{r['run']:<34}{leg_acc:>9.3f}{1 - leg_acc:>10.3f}{seg_acc:>9.3f}{side_acc:>10.3f}{antseg_acc:>9.3f}")

    print(f"\n{'run':<34}{'pt->ta n':>10}{'pt->ta mean':>13}{'ti->ta n':>10}{'ti->ta mean':>13}{'pt mean(any)':>14}")
    for r in results:
        bp = r["geodesic"]["by_pair"]
        bt = r["geodesic"]["by_true_segment"]
        pt_ta = bp.get("pt->ta", dict(n=0, mean=float("nan")))
        ti_ta = bp.get("ti->ta", dict(n=0, mean=float("nan")))
        pt_any = bt.get("pt", dict(n=0, mean=float("nan")))
        print(
            f"{r['run']:<34}{pt_ta['n']:>10}{pt_ta['mean']:>13.4f}"
            f"{ti_ta['n']:>10}{ti_ta['mean']:>13.4f}{pt_any['mean']:>14.4f}"
        )

    OUT = os.path.join(HERE, "out")
    os.makedirs(OUT, exist_ok=True)
    out_path = os.path.join(OUT, "correspondence_confusion.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
