"""A1 deliverable (ii): basin-map leg_acc curves, proximal/distal/coherent/random x 15/23/30deg,
old (i.i.d.) sampler vs corrected (rho=0.6 AR(1)) sampler, overlaid line plot on the same axes as
`out_basin_map_20260825/RESULTS_basin_map_20260825.md`'s table.

PRECONDITION: `run_audit.py` must have been (re)run after the corrected-sampler basin map
(`run_basin_map_corr_rho06_20260825.sh`) finished fitting, so
`diagnostics/correspondence_accuracy/out/correspondence_confusion.json` contains both the
`BASIN_*` (old sampler) and `BASINCORR_*` (corrected sampler) rows.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
CONF_PATH = os.path.join(REPO, "diagnostics", "correspondence_accuracy", "out", "correspondence_confusion.json")

CONDITIONS = ["random", "proximal", "distal", "coherent"]
MAGS = [15, 23, 30]


def load_leg_acc_by_run():
    with open(CONF_PATH) as f:
        rows = json.load(f)
    return {r["run"]: r["leg_confusion"]["accuracy"] for r in rows}


def main():
    acc = load_leg_acc_by_run()

    missing = [f"BASINCORR_{c}_{m}deg" for c in CONDITIONS for m in MAGS if f"BASINCORR_{c}_{m}deg" not in acc]
    if missing:
        raise SystemExit(
            "Missing BASINCORR_* rows in correspondence_confusion.json (run "
            "run_basin_map_corr_rho06_20260825.sh then run_audit.py first): " + ", ".join(missing)
        )

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"random": "#4C72B0", "proximal": "#DD8452", "distal": "#55A868", "coherent": "#C44E52"}
    for cond in CONDITIONS:
        old_y = [acc[f"BASIN_{cond}_{m}deg"] for m in MAGS]
        new_y = [acc[f"BASINCORR_{cond}_{m}deg"] for m in MAGS]
        ax.plot(MAGS, old_y, "--o", color=colors[cond], alpha=0.55, label=f"{cond} (old, i.i.d.)")
        ax.plot(MAGS, new_y, "-s", color=colors[cond], label=f"{cond} (corrected, rho=0.6)")

    if "SYN_clean_zero_wsl" in acc:
        ax.axhline(acc["SYN_clean_zero_wsl"], color="gray", linestyle=":", linewidth=1,
                    label=f"zero-init reference ({acc['SYN_clean_zero_wsl']:.3f})")

    ax.set_xlabel("target perturbation magnitude (deg)")
    ax.set_ylabel("leg_acc")
    ax.set_xticks(MAGS)
    ax.set_title("A1: basin-map leg_acc, old vs corrected (rho=0.6) sampler\n(dashed=old i.i.d. GT poses, solid=corrected AR(1)-coupled GT poses)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()

    out_dir = os.path.join(HERE, "out_A1_corrected_sampler_20260825")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fig_A1_basin_map_leg_acc_old_vs_corrected.png")
    fig.savefig(out_path, dpi=150)
    print(f"[A1] wrote {out_path}")

    print("\nleg_acc, old vs corrected:")
    print(f"{'condition':10s} {'mag':>5s} {'old':>8s} {'corrected':>10s} {'delta':>8s}")
    for cond in CONDITIONS:
        for m in MAGS:
            o = acc[f"BASIN_{cond}_{m}deg"]
            n = acc[f"BASINCORR_{cond}_{m}deg"]
            print(f"{cond:10s} {m:>4d}deg {o:8.3f} {n:10.3f} {n - o:+8.3f}")


if __name__ == "__main__":
    main()
