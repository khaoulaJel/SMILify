#!/usr/bin/env python3
"""
All-pairs leg-joint error over the synth_clean corpus.
Answers: is retrieval's 32.0 deg just the "random pair" baseline?
"""
import argparse, os, pickle, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

from fitter_3d.geom_leg_init import leg_chains  # noqa: E402


def leg_rows_from_names(jnames):
    chains = leg_chains(jnames)
    return sorted({idx - 1 for chain in chains.values() for idx in chain})


def leg_joint_error_deg(jr_pred, jr_gt, leg_rows):
    import scipy.spatial.transform as st
    Ra = st.Rotation.from_rotvec(jr_pred[leg_rows])
    Rb = st.Rotation.from_rotvec(jr_gt[leg_rows])
    return float((Ra.inv() * Rb).magnitude().mean() * 180.0 / np.pi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="diagnostics/moonshot/synth_clean")
    ap.add_argument("--model", default="3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")
    args = ap.parse_args()

    with open(args.model, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        dd = u.load()
    jnames = list(dd["J_names"])
    leg_rows = leg_rows_from_names(jnames)

    gt = np.load(os.path.join(args.corpus, "ground_truth.npz"), allow_pickle=True)
    jr_gt_all = gt["joint_rot"]
    names = [str(n) for n in gt["names"]]
    n = len(names)

    errs = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            errs.append(leg_joint_error_deg(jr_gt_all[j], jr_gt_all[i], leg_rows))

    errs = np.array(errs)
    print(f"all-pairs (n={len(errs)} ordered pairs, {n} specimens):")
    print(f"  mean   = {errs.mean():.3f} deg")
    print(f"  median = {np.median(errs):.3f} deg")
    print(f"  std    = {errs.std():.3f} deg")
    print(f"  min    = {errs.min():.3f} deg")
    print(f"  max    = {errs.max():.3f} deg")
    print()
    print(f"retrieval mean was 32.040 deg -> "
          f"{'INDISTINGUISHABLE from random pairing (descriptor carries ~0 signal)' if abs(errs.mean()-32.040) < errs.std()*0.5 else 'compare manually vs the distribution above'}")


if __name__ == "__main__":
    main()