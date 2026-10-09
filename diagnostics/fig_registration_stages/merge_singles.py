"""DEVIATIONS.md D2: stack the 20 single-specimen CPU fits into the 20-specimen layout score_stages.py
reads (one npz per stage, specimens in sorted mesh order; one metrics.csv). Pure concatenation: every
array is copied, nothing is recomputed, and every input must exist or the merge refuses.

    python diagnostics/fig_registration_stages/merge_singles.py \
        --src /hpcwork/nao48500/fig_registration_stages/D1_s0_cpu1 \
        --hier_out .../D1_s0_cpu1_merged_hier --moon_out .../D1_s0_cpu1_merged
"""

import argparse
import csv
import os

import numpy as np

HIER = ["H0_body", "H1_legs", "H2_joint"]
MOON = ["Stage_2_deform_coarse", "Stage_3_deform_fine"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--hier_out", required=True)
    ap.add_argument("--moon_out", required=True)
    args = ap.parse_args()
    specs = [f"{i:02d}" for i in range(1, 21)]
    for out, sub, stages in ((args.hier_out, "hier", HIER), (args.moon_out, "moon", MOON)):
        os.makedirs(out, exist_ok=True)
        for st in stages:
            zs = [np.load(os.path.join(args.src, s, sub, f"{st}.npz"), allow_pickle=True) for s in specs]
            for s, z in zip(specs, zs):
                assert [os.path.splitext(str(x))[0] for x in z["labels"]] == [s], (st, s, z["labels"])
            keys = zs[0].files
            assert all(z.files == keys for z in zs), st
            np.savez(os.path.join(out, f"{st}.npz"), **{k: np.concatenate([z[k] for z in zs]) for k in keys})
            print(f"merged {st}: {len(zs)} specimens", flush=True)
    rows = []
    for s in specs:
        with open(os.path.join(args.src, s, "moon", "metrics.csv")) as f:
            r = list(csv.DictReader(f))
        assert len(r) == 1 and os.path.splitext(r[0]["mesh"])[0] == s, (s, r)
        rows += r
    with open(os.path.join(args.moon_out, "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("merged metrics.csv", flush=True)


if __name__ == "__main__":
    main()
