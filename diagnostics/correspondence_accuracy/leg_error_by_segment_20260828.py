"""Where does the RESIDUAL leg-level error live?

Context (2026-08-28). Two premises this session had been reasoning from were both wrong:

  1. "leg_acc never moves, not even under the dense GT oracle." FALSE -- it was an n=12 power
     artifact. At n=48 (P48_zero -> P48_cse_all) leg_acc moves +0.0578, 34/48, sign p=0.0055,
     wilcoxon p=5.4e-05 -- a LARGER effect than seg_acc (+0.0310).
  2. "the residual is a wrong-basin problem, so a correspondence-derived rigid per-leg
     initialization should fix it." Not supported: correspondence already rescues the
     catastrophic specimens (below-0.8 count 10 -> 1; below-0.7 3 -> 0), and the leg confusion
     matrix puts essentially ALL residual mass on SPATIALLY ADJACENT legs (l1<->l2, l2<->l3,
     l1_r<->l1_l), with ~0.000 on non-adjacent pairs. A misplaced limb would scatter mass
     non-adjacently; it does not.

So the residual is a boundary problem between neighbouring legs. Adjacent legs approach each
other in two places: the coxal bases along the thorax, and the distal tips near the substrate.
This probe asks WHICH, by stratifying the leg-level error rate by the point's TRUE segment.

Reads exactly the same fitted .npz / sampling / knn path run_audit.py uses (imports its helpers)
so the numbers are commensurable with the audit's own leg_confusion accuracy.

Run: python diagnostics/correspondence_accuracy/leg_error_by_segment_20260828.py
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

N_SAMPLE = 8000
DEVICE = "cpu"

# "CEILING" is not a fit: it substitutes the ground-truth mesh itself for the fitted mesh.
# The synthetic corpus is topology-identical to the template by construction, so this is the
# GT_as_fitted_ceiling arm computed ON THE SAME 48 SPECIMENS as the two fit arms -- a paired
# ceiling, measured rather than assumed. Any leg error it shows is IRREDUCIBLE: it is what
# nearest-fitted-vertex costs when the mesh is already perfect, i.e. genuine geometric overlap
# between physically adjacent leg roots, not a correspondence failure.
ARMS = [("P48_zero", "synth_power48"), ("P48_cse_all", "synth_power48"),
        ("CEILING", "synth_power48")]


def run_arm(run, corpus, template_faces, face_lab, vlabels, rng):
    if run == "CEILING":
        # take the specimen list from a real arm on the same corpus so the pairing is exact
        d = np.load(os.path.join(MOON, "runs", "P48_zero", "Stage_3_deform_fine.npz"))
        spec_labels = [str(x) for x in d["labels"]]
        fitted = None  # filled per-specimen from the target mesh itself
    else:
        d = np.load(os.path.join(MOON, "runs", run, "Stage_3_deform_fine.npz"))
        spec_labels = [str(x) for x in d["labels"]]
        fitted = d["verts"].astype(np.float64)

    # counts[seg] = [n_points, n_leg_errors]; adjacency split of those errors
    counts = {s: np.zeros(2) for s in lb.LEG_SEGMENTS}
    for i, spec_lab in enumerate(spec_labels):
        stem = spec_lab[:-4] if spec_lab.endswith(".obj") else spec_lab
        ov, of, _ = load_obj(os.path.join(MOON, corpus, f"{stem}.obj"), load_textures=False)
        tgt_verts = ov.numpy()
        pts, point_lab = cf.sample_true_labeled_points(
            tgt_verts, of.verts_idx.numpy(), face_lab, N_SAMPLE, rng
        )
        pts_t = torch.tensor(pts, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        true_is_leg = point_lab["leg_id"] != None  # noqa: E711
        fv = tgt_verts.astype(np.float64) if fitted is None else fitted[i]
        fitted_i = torch.tensor(fv, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        _, leg_pred, _, true_leg_sub = cf.leg_confusion(
            point_lab["leg_id"], true_is_leg, pts_t, fitted_i, vlabels
        )
        true_seg_sub = point_lab["leg_seg"][true_is_leg]
        wrong = leg_pred != true_leg_sub
        for s in lb.LEG_SEGMENTS:
            m = true_seg_sub == s
            counts[s][0] += int(m.sum())
            counts[s][1] += int((m & wrong).sum())
    return counts


def main():
    M = ms.load_model()
    template_faces = np.asarray(M["dd"]["f"])
    vlabels = lb.vertex_labels(M["jnames"], M["dominant"])
    face_lab = lb.face_labels(template_faces, vlabels)

    out = {}
    for run, corpus in ARMS:
        print(f"[run] {run} ...", flush=True)
        out[run] = run_arm(run, corpus, template_faces, face_lab, vlabels, np.random.default_rng(1))

    segs = list(lb.LEG_SEGMENTS)
    names = [a[0] for a in ARMS]
    hdr = "".join(f"{n:>13}" for n in names)
    print(f"\n{'segment':<8}{'n_pts':>9}{hdr}{'reducible':>11}{'share_res':>11}")
    # residual share is computed over the CSE arm's error that EXCEEDS the ceiling, i.e. the
    # part that is actually addressable.
    red = {}
    for s_ in segs:
        nc, ec = out["P48_cse_all"][s_]
        nk, ek = out["CEILING"][s_]
        red[s_] = max(ec / max(nc, 1) - ek / max(nk, 1), 0.0) * nc
    tot_red = sum(red.values())
    rows = []
    for s_ in segs:
        rates = {}
        for n in names:
            nn, ee = out[n][s_]
            rates[n] = ee / max(nn, 1)
        r_red = rates["P48_cse_all"] - rates["CEILING"]
        share = red[s_] / max(tot_red, 1e-9)
        rows.append(dict(segment=s_, n_points=int(out["P48_cse_all"][s_][0]),
                         **{f"err_{n}": rates[n] for n in names},
                         reducible=r_red, share_of_reducible_residual=share))
        cells = "".join(f"{rates[n]:>13.4f}" for n in names)
        print(f"{s_:<8}{int(out['P48_cse_all'][s_][0]):>9}{cells}{r_red:>+11.4f}{share:>11.1%}")
    for n in names:
        e = sum(out[n][s_][1] for s_ in segs); t = sum(out[n][s_][0] for s_ in segs)
        print(f"  ALL {n:<20} leg_err={e / t:.4f}  leg_acc={1 - e / t:.4f}")

    op = os.path.join(HERE, "out", "leg_error_by_segment_20260828.json")
    with open(op, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {op}")


if __name__ == "__main__":
    main()
