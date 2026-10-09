"""qc_annotations.py -- catch annotation defects that a visibility flag cannot express.

Flipping `not_present` -> `clear` exposes whatever position the record already held, which may be
an unset placeholder rather than a placed point. And laterality is easy to mix up in a 3D view.
Neither is malformed JSON, so neither shows up as an error anywhere else.

CHECKS
  UNPLACED  a coordinate exactly 0 to within 1e-6. A hand-placed point essentially never lands on
            an exact zero; a default-initialised record does.
  SIDE      a `_r` joint on the left, or `_l` on the right, judged against the specimen's OWN
            midline: the lateral axis is the mean of every bilateral difference present and the
            origin is the mean of every bilateral midpoint, so a single mislabelled pair cannot
            define the frame. Only disagreements beyond `TOL` of mesosoma length are reported;
            joints legitimately near the midline are not flagged.
  DUP       two joints at an identical position, excluding b_t/b_h, whose coincidence is the
            neck-pivot convention (G1) rather than a defect.
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOL = 0.03          # fraction of mesosoma length; below this a joint is "on the midline"
OK_DUP = {("b_t", "b_h"), ("b_h", "b_t")}


def load(p):
    d = json.load(open(p))
    sid = d["specimen_id"].replace("_edited", "")
    h = len(sid) // 2
    if len(sid) % 2 == 0 and sid[:h] == sid[h:]:
        sid = sid[:h]
    P = {j["joint_name"]: np.array(j["position"], float)
         for j in d["joints"] if j.get("position") is not None}
    return sid, P


def main():
    total = 0
    for p in sorted(glob.glob(os.path.join(HERE, "gt_expert", "*_joints.json"))):
        sid, P = load(p)
        pairs = [(n, n[:-2] + "_l") for n in P if n.endswith("_r") and n[:-2] + "_l" in P]
        if not pairs or "b_t" not in P or "b_a_3" not in P:
            print(f"{sid}: insufficient structure -- skipped")
            continue
        scale = np.linalg.norm(P["b_a_3"] - P["b_t"])
        lat = np.mean([P[r] - P[l] for r, l in pairs], axis=0)
        lat /= np.linalg.norm(lat)
        mid = np.mean([(P[r] + P[l]) / 2 for r, l in pairs], axis=0)

        msgs = []
        for n, v in P.items():
            if np.any(np.abs(v) < 1e-6):
                msgs.append(f"UNPLACED {n} has a coordinate at exact zero {np.round(v, 4)}")
        sd = {n: float((v - mid) @ lat) / scale for n, v in P.items()}
        # ORDER is the robust invariant: whatever the midline, a `_r` joint must lie to the RIGHT
        # of its `_l` partner. This catches a swapped pair even when BOTH members sit on the same
        # side of the midline (a turned head, a leaning specimen), which a per-joint side test
        # misses -- it missed exactly this on Eciton's mandibles.
        for r, l in pairs:
            if sd[r] < sd[l] - TOL:
                # ma_r/ma_l are ARTICULATIONS and cannot cross -- but if they were placed at the
                # visible TIPS instead, crossing is expected on a species that holds its mandibles
                # twisted (Aenictus, Eciton). So say which reading to check rather than asserting.
                hint = (" -- if these were placed at the mandible TIPS, crossing is normal; "
                        "the joint should be the ARTICULATION" if r == "ma_r" else
                        " -- pair looks swapped")
                msgs.append(f"ORDER    {r} is LEFT of {l}  (r={100*sd[r]:+.0f}%, "
                            f"l={100*sd[l]:+.0f}% mesosoma){hint}")
        for n, s in sd.items():
            if n.endswith("_r") and s < -TOL:
                msgs.append(f"SIDE     {n} is on the LEFT ({100*s:+.0f}% mesosoma)")
            if n.endswith("_l") and s > TOL:
                msgs.append(f"SIDE     {n} is on the RIGHT ({100*s:+.0f}% mesosoma)")
        ks = list(P)
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                if np.allclose(P[ks[i]], P[ks[j]]) and (ks[i], ks[j]) not in OK_DUP:
                    msgs.append(f"DUP      {ks[i]} and {ks[j]} identical")
        print(f"{sid:46s} {len(P):>2d} joints   {'OK' if not msgs else str(len(msgs))+' ISSUE(S)'}")
        for m in msgs:
            print(f"    {m}")
        total += len(msgs)
    print(f"\n{'=' * 72}\n{total} issue(s) total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
