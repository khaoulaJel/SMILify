"""Task 1, P1/P3 per specimen: does a version's weld distance exceed the scan's smallest self-approach gap?

From the run logs only (no re-computation): V2 logs `min_self_approach_gap` (closest approach of two
topologically distant parts of the pre-weld surface) and every version logs its weld threshold
(V0: "Calculated weld_merge_threshold", V1/V2: "Using weld_merge_threshold"). V1/V2 also log the weld
face loss. Caveat: the gap is measured on V2's pre-weld mesh; V0 keeps only the largest component and
V1 equals V2 up to the weld, so the gap applies to V1 exactly and to V0 approximately.
Writes out/weld_gap.csv and prints the summary.
"""
import csv
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
O = "/hpcwork/nao48500/holotype_task1"


def grab(path, pattern, last=False):
    if not os.path.exists(path):
        return None
    hits = re.findall(pattern, open(path, errors="ignore").read())
    return float(hits[-1] if last else hits[0]) if hits else None


rows = []
for s in open(os.path.join(HERE, "specimens.txt")).read().split():
    log = {v: os.path.join(O, v, s, "run.log") for v in ("V0", "V1", "V2")}
    r = dict(specimen=s,
             gap=grab(log["V2"], r"min_self_approach_gap=([0-9.eE+-]+)"),
             thr_V0=grab(log["V0"], r"Calculated weld_merge_threshold:\s*([0-9.eE+-]+)"),
             thr_V1=grab(log["V1"], r"Using weld_merge_threshold:\s*([0-9.eE+-]+)"),
             loss_V1=grab(log["V1"], r"Weld: \d+ -> \d+ faces \(([0-9.]+)% loss\)"),
             loss_V2=grab(log["V2"], r"Weld: \d+ -> \d+ faces \(([0-9.]+)% loss\)"),
             gap_verdict_V2=(re.findall(r"'verdict': '(\w+)'", open(log["V2"], errors="ignore").read()) or [None])[0]
             if os.path.exists(log["V2"]) else None)
    rows.append(r)
with open(os.path.join(HERE, "out", "weld_gap.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

ok = [r for r in rows if r["gap"] and r["thr_V0"] and r["thr_V1"]]
g = np.array([r["gap"] for r in ok]); t0 = np.array([r["thr_V0"] for r in ok]); t1 = np.array([r["thr_V1"] for r in ok])
print(f"specimens with gap + both thresholds: {len(ok)}/{len(rows)}")
print(f"V0 threshold > gap (weld can fuse across the gap): {(t0 > g).sum()}/{len(ok)}; "
      f"median V0/gap {np.median(t0 / g):.2f}")
print(f"V1 threshold > gap (fix insufficient):          {(t1 > g).sum()}/{len(ok)}; "
      f"median V1/gap {np.median(t1 / g):.2f}, max {np.max(t1 / g):.2f}")
print(f"V1 margin below gap (1 - V1/gap): p10 {np.percentile(1 - t1 / g, 10):.2f}, median {np.median(1 - t1 / g):.2f}")
loss = [r["loss_V1"] for r in rows if r["loss_V1"] is not None]
print(f"V1 weld face loss: median {np.median(loss):.2f}%, max {np.max(loss):.2f}%, >= 20% (abort) {sum(x >= 20 for x in loss)}")
verd = [r["gap_verdict_V2"] for r in rows]
print("V2 post-weld gap verdicts:", {v: verd.count(v) for v in set(verd)})
