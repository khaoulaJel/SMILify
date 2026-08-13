# TASK 3 handoff — GWN disagreement-rate diagnostic, baseline vs gentle

Written by a Claude Code session on 2026-08-13 to hand off to a fresh session picking
this up. Not a finished writeup — TASK1 and TASK2 in this directory each got a
synthesized `TASK{1,2}_*.md`; this is the equivalent raw-results note for TASK3,
one level short of that.

## What TASK3 is

Per the docstring in `diagnostics/moonshot/eval_gwn_disagreement.py`: ship the
GWN (generalized winding number) disagreement-rate diagnostic — previously validated
in `scripts/penetration_joint_study/FINDINGS.md` Step 6 (r=0.947 correlation with
[whatever Step 6 correlated it against — check that file]) — into the moonshot metric
suite as **read-only, CPU-only instrumentation**. No training, no loss weights
touched. For every specimen and every anatomically non-adjacent part pair (both
directions), it reports a GWN-vs-proximity-test disagreement rate, joinable with
`Stage_3_deform_fine_per_pair_penetration.csv` on the same key columns.

Two earlier interactive attempts (`gwn_eval_baseline_out.txt`, `gwn_eval_gentle_out.txt`,
this dir) **crashed partway through** (traceback after specimen 34/50) — do not reuse
those numbers, they're incomplete/superseded.

## What's done

Two SLURM jobs, submitted via `scripts/penetration_joint_study/submit_gwn_disagreement_eval.sbatch`,
both **COMPLETED cleanly** (all 50 specimens) on 2026-08-13 at 10:06:

- `gwn_base` — job **15525266** — run_dir `fit3d_results_all_baseline` — log: `gwn_eval_gwn_base_15525266.log`
- `gwn_gentle` — job **15525267** — run_dir `fit3d_results_all_gentle` — log: `gwn_eval_gwn_gentle_15525267.log`

Per-specimen, per-pair raw CSVs (the actual joinable deliverable):

- `fit3d_results_all_baseline/Stage_3_deform_fine_gwn_disagreement.csv`
- `fit3d_results_all_gentle/Stage_3_deform_fine_gwn_disagreement.csv`

## Population-level summary (baseline vs gentle), recomputed from the two logs

`gentle` = the same TASK1/TASK2 gentle-penetration-weight arm; `baseline` = w_penetration=0.
Sorted by magnitude, delta = gentle − baseline (positive = gentle disagrees *more*):

```
pair/direction                             baseline     gentle      delta
mandible  A_into_B  antenna                  0.0479     0.1221    +0.0742
mandible  B_into_A  antenna                  0.0925     0.0492    -0.0433
gaster    A_into_B  legs                     0.0195     0.0844    +0.0649
head      A_into_B  legs                     0.0036     0.0377    +0.0341
antenna   A_into_B  legs                     0.0097     0.0250    +0.0153
mandible  A_into_B  legs                     0.0020     0.0220    +0.0200
antenna   B_into_A  thorax                   0.0017     0.0148    +0.0131
gaster    B_into_A  legs                     0.0131     0.0127    -0.0004
mandible  B_into_A  thorax                   0.0010     0.0100    +0.0090
antenna   B_into_A  legs                     0.0019     0.0091    +0.0072
antenna   A_into_B  thorax                   0.0031     0.0056    +0.0025
head      B_into_A  legs                     0.0016     0.0036    +0.0020
mandible  A_into_B  thorax                   0.0033     0.0034    +0.0001
mandible  B_into_A  legs                     0.0014     0.0031    +0.0017
antenna   B_into_A  gaster                   0.0000     0.0007    +0.0007
(remaining gaster-pairs: both ~0.0000, tied)
```

**16 of 19 non-trivial pair/directions get worse (higher disagreement) under gentle;
2 improve; 1 tie.** This is population-level (mean over 50 specimens per row), not
independently recomputed from the per-specimen CSVs yet — that recompute is the
first thing to do before trusting this table for a real verdict (project convention:
never trust a training/eval log over the raw file when both exist — see
`.claude/CLAUDE.md` "Diagnostics & Validation").

## Why this matters / how it connects to TASK1 and TASK2

TASK1 already found: gentle weight makes contact **shallower but not less frequent**
(gaster-legs count −20.8%, but *all-pairs* count **+16.8%**). This GWN disagreement
result is directionally consistent — a metric sensitive to overlap/asymmetry going up
under gentle on most pairs looks like the same phenomenon from a different angle,
not a new one. But "directionally consistent with a metric I haven't cross-checked
yet" is not the same as confirmed — that's exactly the gap TASK1's Hard Rule panel
exists to prevent (see `REVIEW_penetration_update.md` for the standing "looked
better, measured worse" trap named there).

## What's left to do (in order)

1. **Independently recompute** the summary table above from the two per-specimen
   `*_gwn_disagreement.csv` files directly (not from the log's printed summary) —
   standing project discipline, not optional.
2. **Read `FINDINGS.md` Step 6** (`scripts/penetration_joint_study/FINDINGS.md`) to
   get the exact prior claim (r=0.947 against what?) this diagnostic was validated
   against, so the new baseline/gentle comparison is interpreted against the right
   prior result, not assumed.
3. **Decide what this means for the standing recommendation** in
   `REVIEW_penetration_update.md` ("ship gentle as an available config option with a
   required per-specimen check, not unconditional") — does the GWN-disagreement
   result strengthen, weaken, or just corroborate that caveat?
4. **Write `TASK3_gwn_disagreement_baseline_vs_gentle.md`**, same format as TASK1/TASK2
   (goal, config/jobs, headline table, Hard-Rule-style verification, a clear verdict
   line), and update the appendix/index if one exists.
5. Only if the discipline items above hold up: fold this into whatever the actual
   penetration-loss shipping decision ends up being — not before.
