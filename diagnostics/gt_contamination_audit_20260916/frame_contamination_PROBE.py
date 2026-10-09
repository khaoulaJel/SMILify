"""Quantify how far the raw `original` landmark coords (used by R3/R9/A1/V6-V13 and the
2026-09-16 slide8 rebuild) sit from the frame-corrected coords that gt_registration.py uses.

obj = (x, z, -y) of stored  --  gt_registration.py:143
Reported as % of Weber's length, the same unit JAB reports joint error in.
"""
import glob
import json
import os

import numpy as np

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
LM = os.path.join(REPO, "annotation", "landmarks")

# the landmarks R3/R9/A1/slide8 pinned as "anatomically correct" supervision
# (r9_objective_audit.py:35-39 CORRECT map; wl_* are used only to define Weber's length)
SUPERVISED = ["clypeal_ant_mid", "cephalic_post_mid", "head_width_r", "head_width_l",
              "mandibular_apex_r", "mandibular_apex_l", "antennal_insertion_r",
              "antennal_insertion_l", "scape_apex_r", "scape_apex_l"]

fix = lambda v: np.array([v[0], v[2], -v[1]], float)

rows = []
for p in sorted(glob.glob(os.path.join(LM, "*_traits.json"))):
    sid = os.path.basename(p).replace("_traits.json", "")
    L = json.load(open(p))["landmarks"]
    if "wl_anterior_r" not in L or "wl_posterior_r" not in L:
        continue
    wl = float(np.linalg.norm(fix(L["wl_anterior_r"]["original"]) - fix(L["wl_posterior_r"]["original"])))

    disp, sup_disp = {}, {}
    for k, v in L.items():
        raw = np.array(v["original"], float)
        d = float(np.linalg.norm(raw - fix(v["original"]))) / wl * 100.0
        disp[k] = d
        if k in SUPERVISED:
            sup_disp[k] = d

    vals = np.array(list(disp.values()))
    rows.append(dict(
        sid=sid, n=len(vals), wl=wl,
        median=float(np.median(vals)), mx=float(vals.max()),
        sup_median=float(np.median(list(sup_disp.values()))) if sup_disp else float("nan"),
        sup=sup_disp,
    ))

print(f"{'specimen':<46} {'n':>3} {'med %WL':>8} {'max %WL':>8} {'supervised med %WL':>19}")
print("-" * 90)
for r in rows:
    print(f"{r['sid']:<46} {r['n']:>3} {r['median']:>8.1f} {r['mx']:>8.1f} {r['sup_median']:>19.1f}")

allmed = np.array([r["median"] for r in rows])
allsup = np.array([r["sup_median"] for r in rows])
print("-" * 90)
print(f"corpus median of per-specimen medians : {np.median(allmed):.1f} % WL")
print(f"range                                 : {allmed.min():.1f} - {allmed.max():.1f} % WL")
print(f"supervised-landmark median            : {np.median(allsup):.1f} % WL")
print(f"supervised range                      : {allsup.min():.1f} - {allsup.max():.1f} % WL")
print()
print("JAB reports D1_PROD joint error at 25.1 % WL. Displacements above that magnitude mean the")
print("supervision target was further from the true anatomy than the unsupervised fit already was.")

json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "frame_contamination.json"), "w"), indent=1)
