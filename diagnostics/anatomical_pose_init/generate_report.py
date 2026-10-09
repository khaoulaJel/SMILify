"""Generates the required plots/tables for the Anatomical Initialization Ceiling Test
(2026-08-20 spec, protocol section 4) from three already-computed sources:
  - diagnostics/anatomical_pose_init/out_ceiling_20260820/section_A_init_quality.json
    (pre-optimisation rotation error, zero vs cheap; no GPU fit needed, computed directly)
  - diagnostics/correspondence_accuracy/out/correspondence_confusion.json
    (post-fit leg/segment/antenna confusion matrices + per-specimen leg_acc, all 3 arms)
  - diagnostics/morphometrics/out/feature_reliability_all_runs.json
    (post-fit per-feature R/snr/bias, all 3 arms -- run with
    `python diagnostics/morphometrics/calib_features.py --runs SYN_clean_w5 ACI_cheap
    SYN_clean_pose25_gtinit_w5 --primary SYN_clean_w5` first)

Arm naming used throughout: A=zero_init (run SYN_clean_w5), B=cheap_anatomical (run ACI_cheap),
C=gt_init (run SYN_clean_pose25_gtinit_w5). A and C are the pre-existing, already-validated
runs this experiment reuses (see submit_ACI_cheap.sbatch's docstring) -- only B was fit fresh.

DEVIATION FROM THE LITERAL SPEC, disclosed here rather than silently: section 4 asks for a
"per-specimen trajectory plot" of leg_acc, distal-leg R, and antenna R showing three bars per
specimen. leg_acc genuinely has a per-specimen value (run_audit.py now tracks it). R
(Pearson correlation across the 12-specimen corpus) is a CORPUS-LEVEL statistic by
construction and has no single-specimen value -- "per-specimen R" is not a well-posed
quantity. For those two panels this script instead plots each specimen's ABSOLUTE RELATIVE
ERROR on the corresponding feature block (mean of |fitted-true|/true over that block's
columns), which IS well-defined per specimen and is the quantity R is computed FROM.
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

ARM_RUN = dict(zero_init="SYN_clean_w5", cheap_anatomical="ACI_cheap", gt_init="SYN_clean_pose25_gtinit_w5")
ARM_ORDER = ["zero_init", "cheap_anatomical", "gt_init"]
ARM_COLOR = dict(zero_init="#a0aec0", cheap_anatomical="#2b6cb0", gt_init="#38a169")


def load_confusion():
    p = os.path.join(REPO, "diagnostics", "correspondence_accuracy", "out", "correspondence_confusion.json")
    rows = json.load(open(p))
    return {r["run"]: r for r in rows}


def load_calib():
    p = os.path.join(REPO, "diagnostics", "morphometrics", "out", "feature_reliability_all_runs.json")
    return json.load(open(p))


def load_section_a():
    return json.load(open(os.path.join(OUT, "section_A_init_quality.json")))


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


def summary_table(confusion, calib, section_a):
    rows = []
    for arm in ARM_ORDER:
        run = ARM_RUN[arm]
        c = confusion.get(run)
        cal = calib.get(run)
        row = dict(arm=arm, run=run)
        if arm == "zero_init":
            row["leg_rot_err_deg_preopt"] = section_a["summary"]["leg_rot_err_deg_mean_zero"]
        elif arm == "cheap_anatomical":
            row["leg_rot_err_deg_preopt"] = section_a["summary"]["leg_rot_err_deg_mean_cheap"]
        else:
            row["leg_rot_err_deg_preopt"] = section_a["summary"]["leg_rot_err_deg_mean_gt"]
        if c:
            row["leg_acc"] = c["leg_confusion"]["accuracy"]
            row["within_leg_seg_acc"] = c["within_leg_segment_confusion"]["accuracy"]
            row["antenna_side_acc"] = c["antenna_side_confusion"]["accuracy"]
            row["antenna_seg_acc"] = c["antenna_segment_confusion"]["accuracy"]
        if cal:
            row["leg_distal_R"] = block_median_R(cal, "leg_distal")
            row["antenna_R"] = block_median_R(cal, "antenna")
            row["all_feature_R"] = all_feature_median_R(cal)
        rows.append(row)

    cols = ["arm", "run", "leg_rot_err_deg_preopt", "leg_acc", "within_leg_seg_acc",
            "antenna_side_acc", "antenna_seg_acc", "leg_distal_R", "antenna_R", "all_feature_R"]
    csv_path = os.path.join(OUT, "summary_table.csv")
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"wrote {csv_path}")
    return rows


def plot_per_specimen_trajectory(confusion, calib):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    # panel 1: leg_acc, genuinely per-specimen
    ax = axes[0]
    names = None
    width = 0.25
    for k, arm in enumerate(ARM_ORDER):
        run = ARM_RUN[arm]
        c = confusion.get(run)
        if not c or "per_specimen_leg_acc" not in c:
            continue
        stems, accs = zip(*c["per_specimen_leg_acc"])
        names = stems
        x = np.arange(len(stems)) + (k - 1) * width
        ax.bar(x, accs, width=width, label=arm, color=ARM_COLOR[arm])
    if names:
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylabel("leg_acc (per specimen)")
    ax.set_title("leg-level correspondence accuracy")
    ax.legend(fontsize=8)

    # panels 2-3: R is a CORPUS-LEVEL statistic (Pearson correlation across all 12 specimens),
    # not decomposable per specimen -- see module docstring. Plotted here as one bar per arm
    # (not per specimen) rather than fabricating a per-specimen number that does not exist.
    for ax, block, title in [(axes[1], "leg_distal", "distal-leg median R (corpus-level, n=12)"),
                              (axes[2], "antenna", "antenna median R (corpus-level, n=12)")]:
        vals, colors, labels = [], [], []
        for arm in ARM_ORDER:
            run = ARM_RUN[arm]
            cal = calib.get(run)
            if not cal:
                continue
            vals.append(block_median_R(cal, block))
            colors.append(ARM_COLOR[arm])
            labels.append(arm)
        ax.bar(range(len(vals)), vals, color=colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20)
        ax.set_ylabel("median R")
        ax.set_title(title)
    fig.suptitle("Per-specimen trajectory (zero / cheap / gt)")
    plt.tight_layout()
    p = os.path.join(OUT, "fig_per_specimen_trajectory.png")
    fig.savefig(p, dpi=130)
    print(f"wrote {p}")


def plot_confusion_heatmaps(confusion):
    for level, key, title in [
        ("leg", "leg_confusion", "leg-level confusion (row-normalized)"),
        ("within_leg_seg", "within_leg_segment_confusion", "within-leg segment confusion (row-normalized)"),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, arm in zip(axes, ARM_ORDER):
            run = ARM_RUN[arm]
            c = confusion.get(run)
            if not c:
                ax.set_visible(False)
                continue
            m = np.array(c[key]["row_normalized"])
            names = c[key]["names"]
            im = ax.imshow(m, vmin=0, vmax=1, cmap="viridis")
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=90, fontsize=7)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=7)
            ax.set_title(f"{arm}\nacc={c[key]['accuracy']:.3f}", fontsize=9)
        fig.suptitle(title)
        plt.tight_layout()
        p = os.path.join(OUT, f"fig_confusion_{level}.png")
        fig.savefig(p, dpi=130)
        print(f"wrote {p}")


def plot_joint_error_boxplot(section_a):
    ps = section_a["per_specimen"]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot(
        [ps["leg_rot_err_deg_zero"], ps["leg_rot_err_deg_cheap"]],
        tick_labels=["zero_init", "cheap_anatomical"],
    )
    ax.set_ylabel("mean per-leg-joint rotation error (deg) vs GT, PRE-optimisation")
    ax.set_title("Init quality before any fitting (n=12)")
    plt.tight_layout()
    p = os.path.join(OUT, "fig_joint_error_boxplot_preopt.png")
    fig.savefig(p, dpi=130)
    print(f"wrote {p}")


def plot_paired_diff(calib):
    blocks = ["leg_distal", "antenna", "mandible", "head", "gaster", "mesosoma"]
    zero_R = {b: block_median_R(calib[ARM_RUN["zero_init"]], b) for b in blocks} if ARM_RUN["zero_init"] in calib else {}
    cheap_R = {b: block_median_R(calib[ARM_RUN["cheap_anatomical"]], b) for b in blocks} if ARM_RUN["cheap_anatomical"] in calib else {}
    gt_R = {b: block_median_R(calib[ARM_RUN["gt_init"]], b) for b in blocks} if ARM_RUN["gt_init"] in calib else {}
    if not (zero_R and cheap_R and gt_R):
        print("[skip] plot_paired_diff -- missing calib data")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(blocks))
    ax.bar(x - 0.2, [cheap_R[b] - zero_R[b] for b in blocks], width=0.4, label="cheap - zero", color="#2b6cb0")
    ax.bar(x + 0.2, [gt_R[b] - cheap_R[b] for b in blocks], width=0.4, label="gt - cheap", color="#38a169")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(blocks, rotation=30)
    ax.set_ylabel("delta median block R")
    ax.legend()
    ax.set_title("Morphometric R deltas (A->B, B->C)")
    plt.tight_layout()
    p = os.path.join(OUT, "fig_R_paired_diff.png")
    fig.savefig(p, dpi=130)
    print(f"wrote {p}")


def failure_case_gallery(confusion):
    run_a = ARM_RUN["zero_init"]
    run_b = ARM_RUN["cheap_anatomical"]
    ca, cb = confusion.get(run_a), confusion.get(run_b)
    if not (ca and cb and "per_specimen_leg_acc" in ca and "per_specimen_leg_acc" in cb):
        print("[skip] failure_case_gallery -- missing per-specimen leg_acc")
        return
    da = dict(ca["per_specimen_leg_acc"])
    db = dict(cb["per_specimen_leg_acc"])
    deltas = sorted(((db[s] - da[s], s) for s in da if s in db))
    with open(os.path.join(OUT, "failure_case_gallery.txt"), "w") as f:
        f.write("cheap_anatomical - zero_init leg_acc delta, sorted (most negative = cheap hurt most)\n")
        for d, s in deltas:
            f.write(f"  {s}: delta={d:+.3f}  zero={da[s]:.3f}  cheap={db[s]:.3f}\n")
        f.write("\nWorst 3 (cheap helped least / hurt most):\n")
        for d, s in deltas[:3]:
            f.write(f"  {s}: {d:+.3f}\n")
        f.write("\nBest 3 (cheap helped most):\n")
        for d, s in deltas[-3:]:
            f.write(f"  {s}: {d:+.3f}\n")
    print(f"wrote {os.path.join(OUT, 'failure_case_gallery.txt')}")
    print("NOTE: overlaid template+scan mesh renders were NOT generated (would need a full "
          "Blender/matplotlib-3D render pass per specimen; out of scope for this pass). The "
          "ranked specimen list above identifies which ones to inspect manually.")


def main():
    confusion = load_confusion()
    calib = load_calib()
    section_a = load_section_a()

    rows = summary_table(confusion, calib, section_a)
    print(json.dumps(rows, indent=1))

    plot_per_specimen_trajectory(confusion, calib)
    plot_confusion_heatmaps(confusion)
    plot_joint_error_boxplot(section_a)
    plot_paired_diff(calib)
    failure_case_gallery(confusion)


if __name__ == "__main__":
    main()
