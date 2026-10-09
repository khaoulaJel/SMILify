"""
07_batch_qa_check.py

Runs the same classes of check we caught by hand on the first two specimens,
across ALL exported *_joints.json files in a folder at once. Doesn't fix
anything — just flags exactly what to look at before sending.

Usage:
    python 07_batch_qa_check.py /path/to/annotation/folder
"""

import json
import sys
import math
from pathlib import Path

folder = Path(sys.argv[1])
files = sorted(folder.glob("*_joints.json"))

if not files:
    print(f"No *_joints.json files found in {folder}")
    sys.exit(1)

LEG_CHAINS = [
    ["l_1_co_r", "l_1_tr_r", "l_1_fe_r", "l_1_ti_r", "l_1_ta_r", "l_1_pt_r"],
    ["l_2_co_r", "l_2_tr_r", "l_2_fe_r", "l_2_ti_r", "l_2_ta_r", "l_2_pt_r"],
    ["l_3_co_r", "l_3_tr_r", "l_3_fe_r", "l_3_ti_r", "l_3_ta_r", "l_3_pt_r"],
    ["l_1_co_l", "l_1_tr_l", "l_1_fe_l", "l_1_ti_l", "l_1_ta_l", "l_1_pt_l"],
    ["l_2_co_l", "l_2_tr_l", "l_2_fe_l", "l_2_ti_l", "l_2_ta_l", "l_2_pt_l"],
    ["l_3_co_l", "l_3_tr_l", "l_3_fe_l", "l_3_ti_l", "l_3_ta_l", "l_3_pt_l"],
]

print(f"Checking {len(files)} specimens in {folder}\n")

any_issues_overall = False

for fpath in files:
    with open(fpath) as f:
        data = json.load(f)

    specimen = data.get("specimen_id", fpath.stem)
    joints = {j["joint_name"]: j for j in data["joints"]}
    issues = []

    # 1. Placeholder-zero check — the exact bug found on Paraponera pass 2
    for name, j in joints.items():
        pos = j.get("position")
        if pos is not None and all(abs(v) < 1e-3 for v in pos):
            issues.append(f"  [ZERO-POS] {name} sits at (0,0,0) — looks like an unmoved placeholder, not a real placement")

    # 2. Known mesh defects blank, but not_present joints exist (should have a note)
    n_not_present = sum(1 for j in joints.values() if j["visibility"] == "not_present")
    if n_not_present > 0 and not data.get("known_mesh_defects", "").strip():
        issues.append(f"  [NO DEFECT NOTE] {n_not_present} joint(s) marked not_present, but known_mesh_defects is blank")

    # 3. Leg-chain monotonicity — real joints should progress smoothly in
    #    at least one dominant axis; a chain that jumps around suggests a
    #    misplaced or swapped joint
    for chain in LEG_CHAINS:
        pts = []
        for name in chain:
            j = joints.get(name)
            if j and j.get("position"):
                pts.append((name, j["position"]))
        if len(pts) < 3:
            continue
        # check consecutive-segment distances for one wildly out of line
        dists = []
        for i in range(len(pts) - 1):
            d = math.dist(pts[i][1], pts[i+1][1])
            dists.append(d)
        if dists:
            mean_d = sum(dists) / len(dists)
            for i, d in enumerate(dists):
                if mean_d > 1e-6 and (d > mean_d * 4 or d < mean_d * 0.15):
                    issues.append(f"  [CHAIN JUMP] {chain[0][:5]}... segment {pts[i][0]}->{pts[i+1][0]} "
                                   f"distance {d:.3f} is very unlike the chain's average ({mean_d:.3f}) — worth a visual check")

    # 4. Completeness sanity — every joint should have a visibility, not "unannotated"
    n_unannotated = sum(1 for j in joints.values() if j["visibility"] == "unannotated")
    if n_unannotated > 0:
        issues.append(f"  [INCOMPLETE] {n_unannotated} joint(s) never annotated at all")

    status = "OK" if not issues else f"{len(issues)} ISSUE(S)"
    print(f"{specimen:<45} {status}")
    for line in issues:
        print(line)
        any_issues_overall = True
    if issues:
        print()

print("\n" + ("Some specimens flagged issues above — worth a visual check before sending."
              if any_issues_overall else
              "No structural issues found across any specimen. Still worth Fabian's expert eye,"
              " but nothing here looks broken."))