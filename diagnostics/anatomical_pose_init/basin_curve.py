"""Basin-curve + capacity-ceiling analysis (2026-08-20 pose-error-sweep follow-up).

Consumes, after `run_audit.py` and `calib_features.py` have been (re-)run to include the sweep
arms (see this repo's `diagnostics/correspondence_accuracy/run_audit.py` RUNS list and
`python diagnostics/morphometrics/calib_features.py --runs SYN_clean_w5 SYN_clean_pose25_gtinit_w5
ACI_noise5 ACI_noise10 ACI_noise15 ACI_noise20 ACI_noise25 ACI_noise30 ACI_gtcapacitycap
--primary SYN_clean_w5`):
  - diagnostics/correspondence_accuracy/out/correspondence_confusion.json
  - diagnostics/morphometrics/out/feature_reliability_all_runs.json

Produces:
  - fig_basin_curve.png: leg_acc / distal-leg R / antenna R / all-feature R vs noise level
    (0/5/10/15/20/25/30 deg), zero_init plotted as a dashed reference line (not on the x-axis
    scale -- it is not a GT+noise point).
  - fig_capacity_ceiling.png: bar comparison, GT-pose free-scale (existing gt_init run) vs
    GT-pose capped-scale (ACI_gtcapacitycap), on the same 4 metrics.
  - basin_curve_table.csv
  - RESULTS_v2.md skeleton with the operating-region call filled in from the data.
"""

import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "out_ceiling_20260820")

LEVELS = [0, 5, 10, 15, 20, 25, 30]
LEVEL_RUN = {
    0: "SYN_clean_pose25_gtinit_w5",
    5: "ACI_noise5",
    10: "ACI_noise10",
    15: "ACI_noise15",
    20: "ACI_noise20",
    25: "ACI_noise25",
    30: "ACI_noise30",
}
ZERO_RUN = "SYN_clean_w5"
CAP_FREE_RUN = "SYN_clean_pose25_gtinit_w5"
CAP_CAPPED_RUN = "ACI_gtcapacitycap"


def load_confusion():
    p = os.path.join(REPO, "diagnostics", "correspondence_accuracy", "out", "correspondence_confusion.json")
    return {r["run"]: r for r in json.load(open(p))}


def load_calib():
    p = os.path.join(REPO, "diagnostics", "morphometrics", "out", "feature_reliability_all_runs.json")
    return json.load(open(p))


def block_of(c):
    if c.startswith("shapedev_"):
        c = c[len("shapedev_") :]
    if c.startswith("leaflen_ma") or c.startswith("mandible") or c.startswith("ma") or c == "attach_ma":
        return "mandible"
    if c.startswith("head"):
        return "head"
    if c.startswith("mesosoma"):
        return "mesosoma"
    if c.startswith("gaster") or "b_a" in c:
        return "gaster"
    if "an_" in c:
        return "antenna"
    if any(s in c for s in ("_ti", "_ta", "_pt")):
        return "leg_distal"
    if any(s in c for s in ("_co", "_tr", "_fe")):
        return "leg_prox"
    return "other"


def block_median_R(calib_run, block):
    cols = [c for c in calib_run["cols"] if block_of(c) == block]
    vals = [calib_run["R"][c] for c in cols if np.isfinite(calib_run["R"][c])]
    return float(np.median(vals)) if vals else float("nan")


def all_feature_median_R(calib_run):
    vals = [v for v in calib_run["R"].values() if np.isfinite(v)]
    return float(np.median(vals)) if vals else float("nan")


def metrics_for(run, confusion, calib):
    c, cal = confusion.get(run), calib.get(run)
    if not c or not cal:
        return None
    return dict(
        leg_acc=c["leg_confusion"]["accuracy"],
        leg_distal_R=block_median_R(cal, "leg_distal"),
        antenna_R=block_median_R(cal, "antenna"),
        all_feature_R=all_feature_median_R(cal),
    )


def main():
    confusion = load_confusion()
    calib = load_calib()

    rows = []
    missing = []
    for level in LEVELS:
        run = LEVEL_RUN[level]
        m = metrics_for(run, confusion, calib)
        if m is None:
            missing.append(run)
            continue
        rows.append(dict(level=level, run=run, **m))
    if missing:
        print(f"[basin_curve] WARNING: missing data for {missing} -- re-run run_audit.py / "
              f"calib_features.py with these included in --runs, then re-run this script.")

    zero_m = metrics_for(ZERO_RUN, confusion, calib)

    csv_path = os.path.join(OUT, "basin_curve_table.csv")
    with open(csv_path, "w") as f:
        f.write("level,run,leg_acc,leg_distal_R,antenna_R,all_feature_R\n")
        for r in rows:
            f.write(f"{r['level']},{r['run']},{r['leg_acc']},{r['leg_distal_R']},{r['antenna_R']},{r['all_feature_R']}\n")
        if zero_m:
            f.write(f"zero_init_ref,{ZERO_RUN},{zero_m['leg_acc']},{zero_m['leg_distal_R']},{zero_m['antenna_R']},{zero_m['all_feature_R']}\n")
    print(f"wrote {csv_path}")

    if rows:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        panels = [("leg_acc", "leg-level correspondence accuracy"),
                  ("leg_distal_R", "distal-leg R"),
                  ("antenna_R", "antenna R"),
                  ("all_feature_R", "all-feature R")]
        xs = [r["level"] for r in rows]
        for ax, (key, title) in zip(axes, panels):
            ys = [r[key] for r in rows]
            ax.plot(xs, ys, "o-", color="#2b6cb0", label="GT + leg-joint noise")
            if zero_m:
                ax.axhline(zero_m[key], ls="--", color="#a0aec0", label="zero_init (ref, not on this x-scale)")
            ax.set_xlabel("injected leg-joint rotation error (deg)")
            ax.set_ylabel(key)
            ax.set_title(title, fontsize=10)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        fig.suptitle("Basin curve: D1 pipeline tolerance to leg-pose init error")
        plt.tight_layout()
        p = os.path.join(OUT, "fig_basin_curve.png")
        fig.savefig(p, dpi=130)
        print(f"wrote {p}")

    cap_free = metrics_for(CAP_FREE_RUN, confusion, calib)
    cap_capped = metrics_for(CAP_CAPPED_RUN, confusion, calib)
    if cap_free and cap_capped:
        fig, ax = plt.subplots(figsize=(7, 5))
        keys = ["leg_acc", "leg_distal_R", "antenna_R", "all_feature_R"]
        x = np.arange(len(keys))
        ax.bar(x - 0.2, [cap_capped[k] for k in keys], width=0.4, label="GT-pose, capped scale (scale_cap=0.052)", color="#dd6b20")
        ax.bar(x + 0.2, [cap_free[k] for k in keys], width=0.4, label="GT-pose, free scale (existing gt_init)", color="#2b6cb0")
        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=20)
        ax.legend(fontsize=8)
        ax.set_title("Capacity ceiling: does extra (free) segment-scale capacity\nrecover more signal once pose is perfect?")
        plt.tight_layout()
        p = os.path.join(OUT, "fig_capacity_ceiling.png")
        fig.savefig(p, dpi=130)
        print(f"wrote {p}")
    else:
        print(f"[basin_curve] WARNING: missing capacity-ceiling data "
              f"(free={cap_free is not None}, capped={cap_capped is not None})")

    print(json.dumps(dict(rows=rows, zero_init_ref=zero_m, capacity_free=cap_free, capacity_capped=cap_capped), indent=1))


if __name__ == "__main__":
    main()
