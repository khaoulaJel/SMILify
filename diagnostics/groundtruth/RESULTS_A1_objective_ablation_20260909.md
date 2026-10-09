# A1 — the objective's hostility to correct anatomy is causally removable, and removing it costs nothing geometrically.

Job 3891381, 3m39s, c25g. 12 annotated specimens, R9's machinery. Five weight configurations; for
each, the CORRECT arm is **re-derived under that configuration** (not reused from production), then
that configuration's objective is evaluated at both states.

---

## Verdict

> **Removing `w_offset` + `w_normal` REVERSES the sign of the hostility gap** — the production
> objective scores the anatomically correct solution **+67.1% worse**; without those two terms it
> scores it **−12.9% BETTER**. And this is not bought with geometric quality: the correct-anatomy
> solution reaches **0.673×** production's Chamfer, a *better* surface fit, not a worse one.
> R9's finding is therefore not a structural property of the fitting problem. It is caused by two
> specific regularisers, and it is removable.

## The table

| config | hostility gap | vs production | anat err (CORRECT) | chamfer ratio |
|---|--:|--:|--:|--:|
| production | +0.000557 (+67.1%) | — | 0.2% | 1.251× |
| **no_offset** | +0.000186 (+27.2%) | **−66.7%** | 0.1% | **0.760×** |
| no_normal | +0.000338 (+47.4%) | −39.3% | 0.2% | 1.266× |
| **no_offset_normal** | **−0.000074 (−12.9%)** | **−113.2%** | **0.1%** | **0.673×** |
| half_offset | +0.000478 (+63.3%) | −14.1% | 0.1% | 1.093× |

`hostility gap` = objective(CORRECT) − objective(production), under that config's own weights.
Negative means the objective prefers the anatomically correct solution.

**Three readings:**

1. **`offset` is the primary culprit, matching R9's decomposition.** R9 attributed 43% of the gap to
   `offset` and 26% to `normal`; A1's causal test agrees — removing `offset` alone removes 66.7% of
   the hostility, `normal` alone 39.3%.
2. **It is a threshold, not a smooth knob.** Halving `w_offset` recovers only 14.1% of the gap,
   while removing it recovers 66.7%. Reweighting is not a substitute for removal.
3. **The geometric cost is negative.** This is the result that makes it a real candidate rather
   than a trade: under `no_offset_normal` the anatomically correct solution fits the surface
   *better* than the production fit does (0.673×). The regularisers were not protecting fit
   quality; they were preventing the deformation the correct anatomy requires while the surface
   term was, on net, indifferent or favourable.

## What this does and does not establish

**Does:** the objective can be made compatible with correct anatomy, cheaply, by removing two
terms. R9's negative was about the objective as shipped, not about the problem.

**Does NOT:** show that an unsupervised fit would *find* that solution. The CORRECT arm remains
landmark-supervised in every configuration — A1 changes whether the objective *fights* the correct
answer, not whether the optimiser can locate it unaided. The remaining barrier is anatomical
identity/search, which is R11/R12's subject.

This completes the causal chain:

| | finding |
|---|---|
| **R3/R4** | correct part identity moves anatomical error 34.7% → 0.2% |
| **R9** | the production objective scores that correct solution +67% worse |
| **A1** | that hostility is caused by `offset`+`normal` and is removable at negative geometric cost |
| **M2** | but no geometric diagnostic — not even oracle ones — predicts morphometric error |
| **R11/R12** | so: can the required identity be supplied automatically? |

## Mechanism checks

- **Production state constant across configs.** `anat prod = 35.5%` identical in all five rows, as
  required — the production parameters never change, only the objective evaluating them.
- Each CORRECT arm re-derived under its own configuration (800 Adam iterations), so no arm inherits
  another's optimisation.

## Limits

n=12, one run per arm, same limits as R3/R4/R9 (reachability under full landmark supervision, not
generalisation; correct-part assignment is anatomical judgement). Removing `offset` has known
side-effects elsewhere in the project (it is the term bounding free-form deformation, added because
unbounded `deform_verts` destroys correspondence — see `trainer_moonshot.py`'s own rationale). A1
says the anatomy/objective conflict is removable; it does **not** say `w_offset=0` is a safe
production recipe. That needs its own corpus-scale test before any recipe change.

## Artifacts

`a1_objective_ablation.py`, `a1_results.json` (full per-config term breakdowns),
`sbatch_logs/R9_3891381.log` (log filename inherited from the R9 template it was derived from).
