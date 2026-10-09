"""
run_symmetric_chamfer_experiment_seeded_scale.py

Scale-validation of TASK7's symmetric_chamfer_sampling fix, after its 2-variable
gate hypothesis (density ratio x baseline concentration) failed direct
replication (Strumigenys_sp.appretiata and Acromyrmex sit almost identically in
that 2D space and produced opposite outcomes) while the raw n=11 aggregate
result cleared TASK5's own acceptance bar on both axes. This run answers the
question that actually matters -- does the fix hold at proper scale -- rather
than pursuing the gate further.

Explicit isolation discipline, unchanged from TASK1-6: TASK5's full config held
exactly fixed (pair-scoped gaster-legs w_penetration, w_offset, w_scale/w_trans,
scheme:'all'), w_sdf=0 explicit in every stage, symmetric_chamfer_sampling is
the only variable. Same seeding convention as FINDINGS.md's own reseeded rerun
and fitter_3d/optimise.py's --seed handling (torch/cuda/numpy seeded, warn_only
deterministic algorithms -- some pytorch3d ops lack a deterministic kernel).

10 specimens, NOT curated this time -- 2 fixed (Acromyrmex_coronatus,
Strumigenys_sp._appretiata_group: the pair that directly contradicted the gate
hypothesis, kept in so they sit inside a real distribution rather than only as
curated outliers) + 8 drawn by an actual random sample (Random(42).sample) over
the remaining 48 bench50_clean specimens -- not hand-picked, and the selection
is reproducible from the documented seed=42. This is a genuine random draw, NOT
representative-by-construction of the full 50-specimen population any more
than any single random sample of 10 is -- stated explicitly so this run isn't
later over-read as a population estimate either, same caveat TASK7's n=11
curated sample needed.

3 training seeds (0, 1, 2) x 2 arms (control, symmetric) = 6 batched runs of
10 specimens each (batch_size=10, already validated as fitting this GPU's 6GB
budget by the density-ratio probe). Deliberately NOT resurrecting the
density-ratio/concentration gate mid-run -- that's a separate, secondary
question, addressed only after this run answers whether the fix holds at all.

Output: fit3d_results_task7_seeded_scale/seed{N}_{control,experimental}/, each
with the standard Stage_3_deform_fine_eval_metrics.csv this project's other
aggregation scripts already know how to read.
"""
import os

import numpy as np
import torch

import config
if os.environ.get("SMIL_DISABLE_PLOTTING"):
    config.PLOT_RESULTS = False

from fitter_3d.run_symmetric_chamfer_experiment import run_arm

MESH_DIR = "diagnostics/moonshot/bench50_clean"
OUT_ROOT = os.environ.get("SYMCHAMFER_SCALE_OUT_ROOT", "fit3d_results_task7_seeded_scale")
SEEDS = [0, 1, 2]

FIXED_SPECIMENS = [
    "Acromyrmex_coronatus_CASENT0744365_processed.obj",
    "Strumigenys_sp._appretiata_group_CASENT0744156_processed.obj",
]
# Reproducible from SELECTION_SEED=42 -- see module docstring. Fixed list, not
# re-drawn at runtime, so a rerun of this exact script is deterministic in
# WHICH specimens it uses even though the training itself is only seeded via
# SEEDS above, not by re-sampling.
RANDOM_SPECIMENS = [
    "Strumigenys_alberti_CASENT0878103_processed.obj",
    "Centromyrmex_brachycola_CASENT0744052_processed.obj",
    "Acromyrmex_lobicornis_CASENT0878007_processed.obj",
    "Eciton_hamatum_CASENT0744580_processed.obj",
    "Dilobocondyla_fouqueti_CASENT0745576_processed.obj",
    "Cyphomyrmex_cf.minutus_CASENT0744280_processed.obj",
    "Cephalotes_minutus_CASENT0709253_processed.obj",
    "Carebara_trechideros_CASENT0877591_processed.obj",
]
SPECIMENS = FIXED_SPECIMENS + RANDOM_SPECIMENS


def seed_everything(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    print(f"[determinism] seeded RNGs with {seed}, deterministic algorithms requested (warn_only)")


def main():
    for seed in SEEDS:
        for symmetric, arm_name in [(False, "control"), (True, "experimental")]:
            seed_everything(seed)
            out_dir = os.path.join(OUT_ROOT, f"seed{seed}_{arm_name}")
            print(f"\n{'='*80}\nseed={seed} arm={arm_name} -> {out_dir}\n{'='*80}")
            run_arm(symmetric_chamfer_sampling=symmetric, out_dir=out_dir, specimens=SPECIMENS)


if __name__ == "__main__":
    main()
