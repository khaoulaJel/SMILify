"""
run_symmetric_chamfer_experiment_gate_probe.py

Extends TASK7's causal test (fitter_3d/run_symmetric_chamfer_experiment.py) from
the original 3 specimens to 8 more, chosen deliberately from the (density_ratio,
baseline_concentration) map built over the full bench50 corpus -- NOT a random
or dense sample. Same config, same discipline (TASK5's stack held fixed, only
symmetric_chamfer_sampling varies, w_sdf=0 explicit), reusing run_arm from the
original module so the two experiments are directly comparable.

Selection logic (diagnostics/density_ratio_bench50.csv x baseline per-pair
concentration, both already computed, zero new compute for the selection
itself):

Replication picks (nearest neighbor in normalized (ratio, concentration) space
to each of the 3 known outcomes) -- tests whether "nearby in variable-space"
really means "same outcome", the hypothesis's core claim:
  - Cephalotes_simillimus_CASENT0744408   (near Acanthostichus, wins every fix)
  - Strumigenys_sp._appretiata_group_CASENT0744156, Cephalotes_spinosus_CASENT0744158
      (near Acromyrmex, loses every fix)
  - Mayriella_sp._CASENT0745610, Crematogaster_nawai_OKENT0105046
      (near Solenopsis, mixed/loses on count -- note Solenopsis's nearest
      neighbor is still ~5x farther than the other two known points' nearest
      neighbors, since it's an extreme corpus outlier on concentration)

Corner picks (combinations the current 3-point model has no example of):
  - Nesomyrmex_angulatus_CASENT0744820 (ratio=5.12, conc=0.954 -- high enough
      ratio that the fix should be relevant AND high enough concentration that
      collateral risk should be severe. The two known mechanisms make opposite
      predictions here; whichever dominates is informative either way.)
  - Ectatomma_brunneum_CASENT0744575 (ratio=8.00, the single most extreme
      density ratio in the whole 50-specimen corpus, roughly 2x Acanthostichus's
      -- tests whether "higher ratio -> bigger win" keeps extrapolating or
      saturates/reverses.)
  - Eciton_hamatum_CASENT0744580 (ratio=0.87 -- the ONLY specimen in the corpus
      with gaster denser than legs, i.e. the asymmetry item 1 targets is
      inverted here -- AND conc=0.277, the lowest/most-diffuse baseline in the
      corpus, the opposite extreme from Solenopsis on that axis.)

Explicitly excludes the dense middle of the distribution -- not near any
decision boundary the two-variable hypothesis draws, so least informative per
GPU-hour spent.
"""
import os

import pandas as pd

from fitter_3d.run_symmetric_chamfer_experiment import run_arm

SPECIMENS = [
    "Cephalotes_simillimus_CASENT0744408_processed.obj",
    "Strumigenys_sp._appretiata_group_CASENT0744156_processed.obj",
    "Cephalotes_spinosus_CASENT0744158_processed.obj",
    "Mayriella_sp._CASENT0745610_processed.obj",
    "Crematogaster_nawai_OKENT0105046_processed.obj",
    "Nesomyrmex_angulatus_CASENT0744820_processed.obj",
    "Ectatomma_brunneum_CASENT0744575_processed.obj",
    "Eciton_hamatum_CASENT0744580_processed.obj",
]

OUT_ROOT = os.environ.get("SYMCHAMFER_GATE_OUT_ROOT", "fit3d_results_task7_gate_probe")


def main():
    control_dir = os.path.join(OUT_ROOT, "control_raw_src_verts")
    experimental_dir = os.path.join(OUT_ROOT, "experimental_symmetric_sampled")

    run_arm(symmetric_chamfer_sampling=False, out_dir=control_dir, specimens=SPECIMENS)
    run_arm(symmetric_chamfer_sampling=True, out_dir=experimental_dir, specimens=SPECIMENS)

    cols = ["specimen", "penetration_num_penetrating", "penetration_fraction_penetrating",
            "penetration_mean_depth_among_penetrating", "f_score@0.01"]
    eval_control = pd.read_csv(os.path.join(control_dir, "Stage_3_deform_fine_eval_metrics.csv"))
    eval_experimental = pd.read_csv(os.path.join(experimental_dir, "Stage_3_deform_fine_eval_metrics.csv"))
    for df in (eval_control, eval_experimental):
        df["specimen"] = df["specimen"].str.replace(r"\.obj$", "", regex=True)

    print("\n=== control (symmetric_chamfer_sampling=False, TASK5-as-is) ===")
    print(eval_control[cols].to_string(index=False))
    print("\n=== experimental (symmetric_chamfer_sampling=True) ===")
    print(eval_experimental[cols].to_string(index=False))

    combined = eval_control[cols].merge(
        eval_experimental[cols], on="specimen", suffixes=("_control", "_experimental")
    )
    out_path = os.path.join(OUT_ROOT, "control_vs_experimental.csv")
    combined.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
