# Deviations from PREREGISTRATION.md (sha256 f4f265e2…)

## D1 — CPU execution under the `default` account (2026-10-09, approved by Khaoula)

**What:** the D1_PROD fit may run on CPU (partition c23ms, account `default`) instead of a c25g GPU
under account rwth2151.

**Why:** the GPU job 4886461 (rwth2151, c25g) had a scheduler start estimate of Mon 12 Oct 20:06 to
Tue 13 Oct 00:04, past the Mon 12 Oct deadline. Dry runs (`sbatch --test-only`, 2026-10-09 11:35) found
no earlier GPU route: rwth2151 c23g 26 Oct; rwth2151 `_low` partitions not permitted; `default` c25g
denied; `default` c23g 25 Oct. CPU: `default` devel (1 h limit) today ~13:20, `default` c23ms
(48 cores, 12 h) Sat 10 Oct ~02:20. The standing rule "always rwth2151, never default" was lifted by the
user for this task only.

**What is unchanged:** recipe (both commands verbatim, same A_prod.yaml), model file and md5, code
state, seed 0, specimen set (all 20), figure specimen 13, every metric and gate.

**What can differ:** floating-point results (CPU vs GPU kernels). The fitter is already not
bit-deterministic run-to-run (~0.01 F on gen@20, memory Z4/Z5), so a CPU fit is a different draw of
the same procedure, not a different procedure.

**Procedure:**
1. 10-iteration-per-stage timing probe on devel (`probes/cpu_timing_PROBE/`), to check that every
   op runs on CPU and to extrapolate runtime.
2. If the extrapolated full run fits the c23ms limit: full run to
   `/hpcwork/nao48500/fig_registration_stages/D1_s0_cpu{,_hier}`, scored with the same scripts.
3. GPU job 4886461 stays queued. The first complete, gate-passing result goes into the figure; the
   CSV and PROVENANCE.md state which device produced it. If both finish, the GPU-vs-CPU difference is
   reported as a reproducibility check.

**Jobs (2026-10-09):** timing probe 4890106 (devel, `default`); full CPU fit 4890115 (c23ms, `default`,
`submit_cpu.sbatch`) and CPU scoring 4890116 (afterok; writes to `run_cpu/` so it cannot overwrite
the GPU job's outputs in this folder). The full CPU fit was queued before the timing probe returned,
to keep its queue position; it is cancelled before it starts if the probe fails or extrapolates past
the 12 h limit. GPU fit 4886461 and GPU scoring 4886482 stay queued (outputs in this folder).

**Update:** timing probe 4890106 cancelled before it started (2026-10-09 ~11:45): the full CPU fit 4890115
started first (11:33 on r23m0101), so its own log measures the real per-iteration rate on the real workload.

## D2 — one specimen per CPU job (batch 1, 20-task array) (2026-10-09, under the user's "do what is best")

**What:** the same two D1_PROD commands, run once per specimen (mesh_dir = a folder holding one
symlinked scan), as a 20-task c23ms array (`submit_cpu_single.sbatch`), then stacked back into the
20-specimen layout by `merge_singles.py` (pure concatenation, label-checked) and scored unchanged by
`submit_score_cpu1.sbatch` into `run_cpu1/`.

**Why:** the 20-batch CPU fit 4890115 is single-thread bound (sstat: 35.9 CPU-min in 8.9 min wall, ~4
of 48 cores busy) and slower than ~5.7 s/iteration (no `100/900` line after 9.5 min), so 3000
hierarchical + 2000 moonshot iterations very likely exceed its 12 h limit. Per-iteration cost scales
with batch size, so batch-1 fits in parallel finish many times sooner.

**What can differ:** each specimen's random draw (point sampling depends on batch composition) and
Adam's epsilon-level batch-scale effects. Specimens share no parameters, so this is another draw of
the same per-specimen procedure. Recipe, model, code, seed value, specimen set, figure specimen,
metrics and gates are unchanged. 4890115 keeps running; if it finishes, it is scored too and its
difference to the batch-1 result is reported.

**Jobs:** array 4890224 (20 tasks), merge+score 4890225 (afterok).

**Scoring rerun:** 4890225 failed in merge_singles.py's label assertion (moonshot metrics.csv labels specimens '01.obj', not '01'); both lookups now strip the extension; fits untouched; rescored as job 4890848.

**Close-out (2026-10-09, user's instruction):** GPU jobs 4886461/4886482 and the 20-batch CPU jobs
4890115/4890116 cancelled (4890115 mid-run; its partial outputs in `/hpcwork/.../D1_s0_cpu{,_hier}` are
unused). The run_cpu1 figure and CSVs were promoted to this folder (figure byte-identical; CSVs = raw
scorer output plus provenance columns via `annotate_csv.py`, values verified identical).

**Final render (2026-10-09):** style B re-render in Blender (job 4892744, c23ms, `default`; re-render only, no fitting).
