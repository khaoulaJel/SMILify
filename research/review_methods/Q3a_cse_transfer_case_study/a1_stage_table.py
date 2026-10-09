"""Q3a / A1: where in the pipeline does CSE help or hurt?

Per specimen, per stage, per JAB region: seed-mean of the per-specimen median FK joint error
(% Weber's length), for A_prod and I_cse, and their paired difference. Read straight from JAB's
scored data (no recomputation), so the numbers are JAB's own.

Writes out/A1_stage_table.txt and out/A1_stage_region.csv.
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
JAB = os.path.join(REPO, "diagnostics", "joint_alignment_benchmark", "data")
OUT = os.path.join(HERE, "out")
STAGES = ["H0_body", "H1_legs", "H2_joint", "Stage_2_deform_coarse", "Stage_3_deform_fine"]
SHORT = dict(zip(STAGES, ["H0", "H1", "H2", "S2", "S3"]))
ARMS = ["A_prod", "I_cse"]


def main():
    os.makedirs(OUT, exist_ok=True)
    J = pd.read_csv(os.path.join(JAB, "scores_joints.csv"))
    J = J[J.arm.isin(ARMS)]
    # specimen x stage x region: median over joints within (seed), then mean over seeds --
    # the same "median over joints, mean over seeds" order JAB uses for its specimen summary.
    g = (J.groupby(["arm", "sid", "stage", "region", "seed"]).err_fk.median()
          .groupby(["arm", "sid", "stage", "region"]).mean().unstack("arm"))
    g["delta"] = g["I_cse"] - g["A_prod"]
    g.reset_index().to_csv(os.path.join(OUT, "A1_stage_region.csv"), index=False)

    # all-joint summary, cross-checked against JAB's own specimen table
    allj = (J.groupby(["arm", "sid", "stage", "seed"]).err_fk.median()
             .groupby(["arm", "sid", "stage"]).mean().unstack("arm"))
    S = pd.read_csv(os.path.join(JAB, "scores_specimen.csv"))
    S = S[S.arm.isin(ARMS)].groupby(["arm", "sid", "stage"]).med_fk.mean().unstack("arm")
    dev = (allj - S.loc[allj.index]).abs().max().max()
    assert dev < 1e-6, f"recomputed specimen medians differ from JAB's scores_specimen.csv by {dev}"

    lines = [f"A1 -- seed-mean median FK error (% WL). Cross-check vs scores_specimen.csv: max dev {dev:.1e}", ""]
    sids = sorted(allj.index.get_level_values(0).unique())
    hdr = f"{'specimen':<14}" + "".join(f"{SHORT[s]:>16}" for s in STAGES)
    lines += ["all joints: A_prod / I_cse (delta)", hdr]
    for sid in sids:
        cells = "".join(f"{allj.loc[(sid, s), 'A_prod']:6.1f}/{allj.loc[(sid, s), 'I_cse']:5.1f}"
                        f"({allj.loc[(sid, s), 'I_cse'] - allj.loc[(sid, s), 'A_prod']:+5.1f})"[:16].rjust(16)
                        for s in STAGES)
        lines.append(f"{sid[:14]:<14}{cells}")
    lines.append("")
    for s in STAGES:
        d = (allj.xs(s, level="stage")["I_cse"] - allj.xs(s, level="stage")["A_prod"]).values
        lines.append(f"{SHORT[s]}: CSE better on {(d < 0).sum()}/11, median delta {np.median(d):+.1f}")

    lines += ["", "final stage (S3) and H2 by region: delta I_cse - A_prod (% WL)"]
    regions = sorted(J.region.unique())
    for s in ["H2_joint", "Stage_3_deform_fine"]:
        lines.append(f"-- {SHORT[s]}")
        lines.append(f"{'specimen':<14}" + "".join(f"{r[:12]:>13}" for r in regions))
        for sid in sids:
            row = ""
            for r in regions:
                try:
                    row += f"{g.loc[(sid, s, r), 'delta']:>13.1f}"
                except KeyError:
                    row += f"{'-':>13}"
            lines.append(f"{sid[:14]:<14}{row}")
    txt = "\n".join(lines)
    open(os.path.join(OUT, "A1_stage_table.txt"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
