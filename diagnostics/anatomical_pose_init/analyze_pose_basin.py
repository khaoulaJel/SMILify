#!/usr/bin/env python3

"""
Analyze whether initialization error predicts final fitting quality.

Question:
    Is the remaining failure primarily explained by the magnitude of
    initial leg-pose error, or does the optimizer exhibit a more
    qualitative basin-of-attraction behavior?

Inputs:
    1. section_A_init_quality.json
       Per-specimen initial leg-pose errors for zero and cheap initialization.

    2. Optional correspondence_confusion.json
       Used if it contains per-specimen correspondence results.

    3. Optional morphometric reliability JSON
       Used for aggregate R values.

The controlled pose-noise sweep is represented directly here because
the injected perturbation magnitude is exactly:
    5, 10, 15, 20, 25, 30 degrees.

The script never treats corpus-level Pearson R as a per-specimen quantity.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Known controlled results from the experiment you reported.
#
# These are intentionally hard-coded because the pose-noise experiment
# used fixed perturbation magnitudes and you already supplied the
# resulting correspondence and morphometric summaries.
# ---------------------------------------------------------------------

NOISE_RESULTS = {
    "GT": {
        "pose_error_deg": 0.0,
        "leg_acc": 0.938,
        "seg_acc": 0.866,
        "antenna_side": 1.000,
        "antenna_seg": 0.828,
        "leg_distal_R": 0.604,
        "antenna_R": 0.677,
        "all_feature_R": 0.779,
    },
    "noise5": {
        "pose_error_deg": 5.0,
        "leg_acc": 0.938,
        "seg_acc": 0.863,
        "antenna_side": 1.000,
        "antenna_seg": 0.812,
        "leg_distal_R": 0.396,
        "antenna_R": 0.743,
        "all_feature_R": 0.755,
    },
    "noise10": {
        "pose_error_deg": 10.0,
        "leg_acc": 0.939,
        "seg_acc": 0.861,
        "antenna_side": 1.000,
        "antenna_seg": 0.817,
        "leg_distal_R": 0.614,
        "antenna_R": 0.773,
        "all_feature_R": 0.762,
    },
    "noise15": {
        "pose_error_deg": 15.0,
        "leg_acc": 0.937,
        "seg_acc": 0.862,
        "antenna_side": 1.000,
        "antenna_seg": 0.820,
        "leg_distal_R": 0.477,
        "antenna_R": 0.752,
        "all_feature_R": 0.779,
    },
    "noise20": {
        "pose_error_deg": 20.0,
        "leg_acc": 0.888,
        "seg_acc": 0.851,
        "antenna_side": 1.000,
        "antenna_seg": 0.821,
        "leg_distal_R": 0.341,
        "antenna_R": 0.461,
        "all_feature_R": 0.731,
    },
    "noise25": {
        "pose_error_deg": 25.0,
        "leg_acc": 0.881,
        "seg_acc": 0.837,
        "antenna_side": 1.000,
        "antenna_seg": 0.828,
        "leg_distal_R": 0.194,
        "antenna_R": 0.749,
        "all_feature_R": 0.689,
    },
    "noise30": {
        "pose_error_deg": 30.0,
        "leg_acc": 0.797,
        "seg_acc": 0.798,
        "antenna_side": 1.000,
        "antenna_seg": 0.843,
        "leg_distal_R": 0.511,
        "antenna_R": 0.692,
        "all_feature_R": 0.674,
    },
}


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 3:
        return np.nan

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    x = pd.Series(np.asarray(x, dtype=float)).rank().to_numpy()
    y = pd.Series(np.asarray(y, dtype=float)).rank().to_numpy()
    return pearson(x, y)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------
# Section A: cheap initializer
# ---------------------------------------------------------------------

def analyze_cheap_initializer(path):
    data = load_json(path)

    p = data["per_specimen"]

    names = p["names"]
    zero = np.asarray(p["leg_rot_err_deg_zero"], dtype=float)
    cheap = np.asarray(p["leg_rot_err_deg_cheap"], dtype=float)

    df = pd.DataFrame({
        "specimen": names,
        "zero_pose_error_deg": zero,
        "cheap_pose_error_deg": cheap,
        "cheap_minus_zero_deg": cheap - zero,
    })

    print("\n" + "=" * 72)
    print("CHEAP INITIALIZER: PRE-OPTIMIZATION POSE ERROR")
    print("=" * 72)

    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\nSummary:")
    print(f"  zero mean pose error  : {zero.mean():.3f} deg")
    print(f"  cheap mean pose error : {cheap.mean():.3f} deg")
    print(f"  mean deterioration    : {(cheap-zero).mean():.3f} deg")
    print(f"  specimens worse       : {(cheap > zero).sum()}/{len(zero)}")

    return df


# ---------------------------------------------------------------------
# Controlled noise sweep
# ---------------------------------------------------------------------

def analyze_noise_sweep():
    df = pd.DataFrame.from_dict(NOISE_RESULTS, orient="index")
    df.index.name = "run"
    df = df.reset_index()

    print("\n" + "=" * 72)
    print("CONTROLLED POSE-ERROR SWEEP")
    print("=" * 72)

    print(
        df[
            [
                "run",
                "pose_error_deg",
                "leg_acc",
                "seg_acc",
                "leg_distal_R",
                "antenna_R",
                "all_feature_R",
            ]
        ].to_string(index=False)
    )

    # Correlations are descriptive here, not proof of a causal linear
    # relationship. The levels are only seven points and the response
    # is clearly not monotonic.
    print("\nCorrelation with injected pose-error magnitude:")

    for metric in [
        "leg_acc",
        "seg_acc",
        "leg_distal_R",
        "antenna_R",
        "all_feature_R",
    ]:
        x = df["pose_error_deg"]
        y = df[metric]

        print(
            f"  {metric:16s}"
            f" Pearson={pearson(x, y): .3f}"
            f" Spearman={spearman(x, y): .3f}"
        )

    # First obvious degradation point.
    baseline = df.loc[df["run"] == "GT", "leg_acc"].iloc[0]

    print("\nDeviation from GT-init:")
    for _, row in df.iterrows():
        if row["run"] == "GT":
            continue

        print(
            f"  {row['run']:8s}"
            f" leg_acc Δ={row['leg_acc']-baseline:+.3f}"
            f" distal_R={row['leg_distal_R']:.3f}"
            f" all_R={row['all_feature_R']:.3f}"
        )

    return df


# ---------------------------------------------------------------------
# Cheap initializer vs controlled perturbation regime
# ---------------------------------------------------------------------

def compare_cheap_to_noise(cheap_df):
    cheap_mean = cheap_df["cheap_pose_error_deg"].mean()

    df = pd.DataFrame.from_dict(NOISE_RESULTS, orient="index")

    print("\n" + "=" * 72)
    print("CHEAP INITIALIZER VS CONTROLLED NOISE REGIME")
    print("=" * 72)

    print(
        f"\nCheap initializer mean initial leg-pose error: "
        f"{cheap_mean:.3f} deg"
    )

    print(
        "\nThe controlled sweep brackets the cheap initializer as follows:"
    )

    for _, row in df.iterrows():
        print(
            f"  {row.name:8s}"
            f"  error={row.pose_error_deg:5.1f}°"
            f"  leg_acc={row.leg_acc:.3f}"
            f"  distal_R={row.leg_distal_R:.3f}"
            f"  all_R={row.all_feature_R:.3f}"
        )

    print(
        "\nImportant:"
        "\nThe cheap initializer cannot be interpreted as simply "
        "\"approximately 28.5 degrees of random noise.\""
        "\nIts error has a structured anatomical direction and also "
        "contains a global-orientation estimate that is diagnostic only."
        "\nTherefore the controlled sweep tests tolerance to magnitude, "
        "while the cheap initializer tests a particular structured error."
    )


# ---------------------------------------------------------------------
# Optional extraction of specimen-level correspondence accuracy
# ---------------------------------------------------------------------

def recursively_find_specimen_accuracy(obj, path=""):
    """
    Try to locate dictionaries/lists that look like specimen-level
    correspondence results.

    This is deliberately conservative. It does not assume a schema.
    """

    hits = []

    if isinstance(obj, dict):
        keys = {str(k).lower() for k in obj.keys()}

        # Common possible structures.
        possible_name_keys = [
            k for k in obj
            if str(k).lower() in {
                "names",
                "specimens",
                "specimen_names",
                "per_specimen",
            }
        ]

        possible_acc_keys = [
            k for k in obj
            if str(k).lower() in {
                "leg_acc",
                "leg_accuracy",
                "accuracy",
                "per_specimen_leg_acc",
            }
        ]

        if possible_name_keys and possible_acc_keys:
            hits.append((path, obj))

        for k, v in obj.items():
            hits.extend(
                recursively_find_specimen_accuracy(
                    v,
                    f"{path}/{k}" if path else str(k)
                )
            )

    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits.extend(
                recursively_find_specimen_accuracy(
                    v,
                    f"{path}[{i}]"
                )
            )

    return hits


def inspect_correspondence_file(path):
    if not path or not os.path.exists(path):
        return

    print("\n" + "=" * 72)
    print("CORRESPONDENCE OUTPUT INSPECTION")
    print("=" * 72)

    data = load_json(path)

    hits = recursively_find_specimen_accuracy(data)

    if not hits:
        print(
            "No obvious specimen-level leg-accuracy structure was found."
        )
        print(
            "This is not an error: the supplied correspondence JSON may "
            "only contain aggregate confusion matrices."
        )
        print(
            "If so, specimen-level pose-error vs leg-accuracy correlation "
            "cannot be computed from that file."
        )
        return

    print(f"Found {len(hits)} candidate specimen-level structures:")

    for path_name, obj in hits:
        print(f"\n  {path_name}")

        for key, value in obj.items():
            if isinstance(value, list):
                print(f"    {key}: list[{len(value)}]")
            else:
                print(f"    {key}: {type(value).__name__}")


# ---------------------------------------------------------------------
# Basin interpretation
# ---------------------------------------------------------------------

def interpret(df):
    print("\n" + "=" * 72)
    print("INTERPRETATION")
    print("=" * 72)

    leg = df["leg_acc"].to_numpy()
    distal = df["leg_distal_R"].to_numpy()
    all_r = df["all_feature_R"].to_numpy()

    # Compare early and late regimes.
    low = df[df["pose_error_deg"] <= 15]
    high = df[df["pose_error_deg"] >= 20]

    print(
        "\nLow-error regime (0–15°):"
        f" mean leg accuracy = {low.leg_acc.mean():.3f}"
        f", mean all-feature R = {low.all_feature_R.mean():.3f}"
    )

    print(
        "Higher-error regime (20–30°):"
        f" mean leg accuracy = {high.leg_acc.mean():.3f}"
        f", mean all-feature R = {high.all_feature_R.mean():.3f}"
    )

    print(
        "\nThis should NOT be described as a precise angular threshold."
    )

    print(
        "The controlled sweep is non-monotonic at the individual-metric "
        "level, particularly for distal-leg reliability."
    )

    print(
        "The robust conclusion is instead that D1 has a finite basin of "
        "successful recovery: sufficiently structured pose errors can "
        "move the optimizer into a worse correspondence basin."
    )

    print(
        "\nThe cheap initializer must therefore be evaluated as a "
        "structured anatomical hypothesis, not merely by its mean angular "
        "distance from ground truth."
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--init-quality",
        required=True,
        help="Path to section_A_init_quality.json",
    )

    parser.add_argument(
        "--correspondence",
        default=None,
        help="Optional correspondence_confusion.json",
    )

    parser.add_argument(
        "--out",
        default=None,
        help="Optional output CSV path for the controlled sweep.",
    )

    args = parser.parse_args()

    cheap_df = analyze_cheap_initializer(args.init_quality)

    noise_df = analyze_noise_sweep()

    compare_cheap_to_noise(cheap_df)

    if args.correspondence:
        inspect_correspondence_file(args.correspondence)

    interpret(noise_df)

    if args.out:
        noise_df.to_csv(args.out, index=False)
        print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()