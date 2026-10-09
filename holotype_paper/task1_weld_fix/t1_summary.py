"""Task 1 verdict table: PROTOCOL.md pass criteria + the user's non-inferiority bar, per specimen.

Inputs: out/measure_<s>.json (t1_measure), out/fidelity_<s>.json (t1_fidelity_maps), out/weld_gap.csv
(t1_weld_gap). Every specimen where V1 is worse than V0 on any check is listed by name, so each gets
an individual look (renders/maps) instead of disappearing into a median.
Checks, V1 vs V0 (per specimen):
  lost      scan surface lost by V1 only > 1 pp                        (anatomy V0 keeps but V1 drops)
  fidelity  V1 deviation p99 above V0's by > 0.1% of diagonal          (geometry altered)
  nonmanif  main-body non-manifold edges of V1 > V0                    (fused-surface signature)
  open      main-body open edges of V1 > V0                            (unwelded seams; topology)
  debris    faces in pieces < 1% of total, V1 > V0 + 1 pp
  abort     V1 run failed / aborted
"""
import csv
import glob
import json
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
specs = open(os.path.join(HERE, "specimens.txt")).read().split()
gap = {r["specimen"]: r for r in csv.DictReader(open(os.path.join(HERE, "out", "weld_gap.csv")))}
timing = open("/hpcwork/nao48500/holotype_task1/timing.txt").read()
rows, flags = [], {}
for s in specs:
    mp, fp = os.path.join(HERE, "out", f"measure_{s}.json"), os.path.join(HERE, "out", f"fidelity_{s}.json")
    if not (os.path.exists(mp) and os.path.exists(fp)):
        flags.setdefault("missing_outputs", []).append(s)
        continue
    m, f = json.load(open(mp)), json.load(open(fp))
    v0, v1 = m["V0"], m["V1"]
    r = dict(specimen=s,
             lost_V0only=f["lost_outer_pct"]["V0_only"], lost_V1only=f["lost_outer_pct"]["V1_only"],
             uncovered_V0=v0["uncovered_outer_pct"], uncovered_V1=v1["uncovered_outer_pct"],
             dev_p99_V0=f["V0"]["dev_p99_pct"], dev_p99_V1=f["V1"]["dev_p99_pct"], floor_p99=f["floor_p99_pct"],
             nonmanif_V0=v0["nonmanifold_edges"], nonmanif_V1=v1["nonmanifold_edges"],
             open_V0=v0["boundary_edges"], open_V1=v1["boundary_edges"],
             debris_V0=100 * v0["debris_face_share"], debris_V1=100 * v1["debris_face_share"],
             gap=gap.get(s, {}).get("gap"), thr_V0=gap.get(s, {}).get("thr_V0"), thr_V1=gap.get(s, {}).get("thr_V1"),
             verdict_V2=gap.get(s, {}).get("gap_verdict_V2"))
    checks = dict(lost=r["lost_V1only"] > 1.0, fidelity=r["dev_p99_V1"] > r["dev_p99_V0"] + 0.1,
                  nonmanif=r["nonmanif_V1"] > r["nonmanif_V0"], open=r["open_V1"] > r["open_V0"],
                  debris=r["debris_V1"] > r["debris_V0"] + 1.0,
                  abort=f"version=V1 specimen={s} exit=0" not in timing)
    for k, bad in checks.items():
        r[f"worse_{k}"] = bool(bad)
        if bad:
            flags.setdefault(k, []).append(s)
    rows.append(r)
with open(os.path.join(HERE, "out", "verdict_table.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

n = len(rows)
d = np.array([r["uncovered_V0"] - r["uncovered_V1"] for r in rows])
print(f"specimens scored: {n}/{len(specs)}")
print(f"P2 lost anatomy: V1 loses less on {(d > 0).sum()}/{n}, more on {(d < 0).sum()}; median V0 {np.median([r['uncovered_V0'] for r in rows]):.2f}% "
      f"vs V1 {np.median([r['uncovered_V1'] for r in rows]):.2f}%; sign p {stats.binomtest(int((d > 0).sum()), int((d != 0).sum())).pvalue:.2g}")
print(f"      surface lost by V0 only: median {np.median([r['lost_V0only'] for r in rows]):.2f}% max {max(r['lost_V0only'] for r in rows):.2f}% | "
      f"by V1 only: median {np.median([r['lost_V1only'] for r in rows]):.2f}% max {max(r['lost_V1only'] for r in rows):.2f}%")
print(f"fidelity: V1 dev p99 above floor by median {np.median([r['dev_p99_V1'] - r['floor_p99'] for r in rows]):+.3f}% (V0 {np.median([r['dev_p99_V0'] - r['floor_p99'] for r in rows]):+.3f}%)")
for k in ("lost", "fidelity", "nonmanif", "open", "debris", "abort"):
    names = flags.get(k, [])
    print(f"V1 worse than V0 on {k:<9}: {len(names):>2}/{n}  {', '.join(x.split('_CASENT')[0].split('_OKENT')[0] for x in names[:12])}")
if flags.get("missing_outputs"):
    print("missing outputs:", flags["missing_outputs"])
