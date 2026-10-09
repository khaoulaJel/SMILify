# Joint-Alignment Benchmark (JAB)

Answers: *which SMILify registration strategies align the model skeleton best with expert-annotated
joints?* Start with **`REPORT.md`**.

| path | contents |
|---|---|
| `REPORT.md` | results, answer, limitations |
| `PREREGISTRATION.md` · `PREREGISTRATION.FREEZE` | design frozen before any arm ran (SHA-256 + timestamp; one citation-only amendment logged) |
| `DEVIATIONS.md` | D1 FK-pivot extraction fix (before scoring) · N1 Dolichoderus stays excluded |
| `LITERATURE.md` | methods sources, each tied to the design decision it grounds |
| `tools/gt_registration.py` | fit-independent expert joints → fitter frame (reflection + L/R swap, chained ICP, exclusion) |
| `tools/blend_dump.py` | headless-Blender dump of annotation scenes (bpy venv: `/hpcwork/nao48500/bpy_venv`, needs `LD_LIBRARY_PATH=<bpy>/lib`) |
| `tools/model_joints.py` | FK / REG / SKIN model joints with checkpoint-reproduction guard |
| `tools/score.py` · `tools/analyze.py` · `tools/figures.py` | scoring, pre-registered statistics, figures |
| `tools/oracle.py` · `tools/make_learned_init.py` | oracle O0/O1/O2 runs; arm H input |
| `submit_arms.sbatch` · `submit_oracle.sbatch` · `cfg/` | the 12 arms × 3 seeds and their verified configs |
| `data/` | GT in fitter frame, registration audit, per-joint and per-specimen scores, code-state patch |
| `results/results.json` | every reported statistic |
| `figures/` | F1–F13 (PNG + SVG) |
| `probes/` | instrument audit (old vs new ruler, swap test) and the GT-on-scan visual check |

Fits live outside the repo in `/hpcwork/nao48500/jab_runs/`; annotation-scene dumps in
`/hpcwork/nao48500/jab_blend*`.
