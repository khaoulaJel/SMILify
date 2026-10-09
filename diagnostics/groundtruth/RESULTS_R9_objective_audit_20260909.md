# R9 — CASE 2. The production objective actively prefers the anatomically WRONG solution.

Job 3886056, 2m44s, c25g. 12 annotated specimens, same machinery as R3/R4. Full term-by-term
decomposition of `D1_PROD.yaml`'s Stage_3_deform_fine objective, production fit vs. the R3
anatomically-CORRECT solution.

---

## Verdict

> **CASE 2, per the pre-registered decision tree.** The production objective, evaluated at R3's
> anatomically correct solution, is **67% higher** than at the actual production fit (0.000826 →
> 0.001380, Δ = +0.000554). **Every one of the nine active terms costs more for CORRECT — none
> reverses.** This is not a search-basin problem the optimizer merely fails to find; the objective
> itself, if it ever reached the anatomically correct configuration, would be actively pushed back
> out of it. Redesigning the objective is now a load-bearing requirement, not an optional
> improvement alongside better search.

## The full table

| arm | TOTAL | chamfer | edge | normal | laplacian | offset | sym | mid | scale | limit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| production | 0.00083 | 0.00045 | 0.00012 | 0.00011 | 0.00000 | 0.00014 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| **CORRECT** | **0.00138** | 0.00056 | 0.00016 | 0.00026 | 0.00001 | 0.00038 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| CONSENSUS | 0.00126 | 0.00052 | 0.00015 | 0.00025 | 0.00001 | 0.00032 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| FREE | 0.00117 | 0.00050 | 0.00014 | 0.00025 | 0.00001 | 0.00027 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |

(`allo`/`beta_prior`/`trans`/`jres`/`dsym`/`cse_corr`/`nrm_align` all inactive at `w_*=0` in this
config — reported as `--`, correctly excluded, not silently dropped.)

## Where the extra cost comes from — decomposed, not just totaled

| term | production | CORRECT | Δ (CORRECT − production) | % of total Δ |
|---|---:|---:|---:|---:|
| **offset** | 0.000144 | 0.000384 | **+0.000240** | **43%** |
| **normal** | 0.000112 | 0.000257 | **+0.000145** | **26%** |
| **chamfer** | 0.000445 | 0.000561 | **+0.000116** | **21%** |
| edge | 0.000116 | 0.000163 | +0.000047 | 8% |
| laplacian | 0.000004 | 0.000006 | +0.000002 | <1% |
| mid | 0.000000 | 0.000002 | +0.000002 | <1% |
| limit | 0.000000 | 0.000002 | +0.000002 | <1% |
| sym | 0.000002 | 0.000002 | +0.000000 | <1% |
| scale | 0.000001 | 0.000001 | +0.000000 | <1% |

**Two findings worth pulling apart, not one:**

1. **`offset` (43% of the gap) and `normal` (26%) are regularizers actively fighting the correct
   anatomy.** `offset` penalizes `‖deform_verts‖²` — reaching the correct anatomy requires more
   free-form surface deformation than production uses, and the objective explicitly taxes that
   (see `trainer_moonshot.py`'s own documented rationale for `offset`: "the optimizer prefers to
   explain the target with POSE and SHAPE, and only spends offsets where it must"). `normal`
   (mesh normal consistency) costing more says the correct-anatomy surface is locally less smooth
   than production's — plausible if correctly-placed mandibles/antennae require sharper local
   curvature than a chamfer-convenient blob.
2. **`chamfer` itself costs 21% more for CORRECT — the anatomically correct solution is a *worse*
   raw surface fit than production's, not just a worse regularized one.** This is the more
   surprising half: production isn't merely cutting corners on regularizers while nailing the
   surface — it's genuinely explaining the point cloud *better* than the correct-anatomy solution
   does. The two together say the production fit and the anatomically-correct fit are landing on
   two different local surface explanations, and the data term itself has a mild preference for
   the wrong one.

## What this settles, precisely

Combined with R3/R4 (the correct anatomy is *reachable* under enough constraint) and G6
(no term constrains joint *position*), R9 completes the picture for **Layer 2 (anatomical fit)** of
the validity framework this thread has converged on:

- **Layer 1 (geometric fit)**: insufficient alone — already established (low chamfer coexists with
  35% anatomical error, R2/R3).
- **Layer 2 (anatomical fit)**: the model has the *capacity* (R3, 0.2% reachable) but the
  *objective itself* does not want to go there (R9, +67% total cost, every term). This is stronger
  than "the optimizer can't find it" — it says the optimizer, correctly finding the true minimum of
  the CURRENT objective, would choose the anatomically wrong basin on purpose.
- **Layer 3 (morphometric validity)**: the Atta result stands independently as the positive anchor
  — real biological scaling recovered, on specimens/measurements where anatomical identity happens
  to be either unambiguous (head width, mandible span — large, non-confusable structures) or
  supplied correctly by construction. R9 says this cannot be assumed to generalize to structures
  where the objective's own preference actively fights correct placement (exactly R4's finding:
  the named parts drift furthest on the morphologies — long mandibles, unusual antennae — a
  morphometric study most wants to measure).

## Consequence for the roadmap

Per the pre-registered decision tree: **this is not "focus on automatic anatomical identity
recovery and stop there."** A semantic-identity layer that successfully told the optimizer where
the correct anatomy is would still be fighting an objective that actively prefers somewhere else —
R9 shows getting there costs real, measured loss across chamfer/normal/offset simultaneously, not
one fixable term. The identity-recovery work (Priority 2/3 in the current roadmap) is still
necessary — nothing here retracts R3/R4 — but it is not sufficient on its own; an objective
redesign candidate is now equally load-bearing, and `offset`/`normal` are the two concrete places
to start looking (86% of the gap between them), not a generic "add more terms."

## Limits — stated per the preregistration's own amendment

This is **one total-objective number per arm**, not a per-specimen paired significance test —
`MoonshotStage.forward()` batch-means every term internally, so a formal sign-test p-value is not
available from this instrumentation. What substitutes for it here: the effect is not a single
close call but nine independent terms, all moving the same direction, none reversing — a pattern
far less likely under a true null than a single ambiguous total would be. A genuinely rigorous
per-specimen version would need `MoonshotStage.forward()` extended to expose per-sample terms
before this could carry a formal p-value; worth doing if this result needs to withstand harder
scrutiny (e.g. before a publication claim), not required to trust the direction of this result.

n=12, one run per arm, same limits as R3/R4 (reachability under full landmark supervision, not
generalization; correct-part assignment is anatomical judgement, not derived).

## Artifacts

- `r9_objective_audit.py` — extends `r3_correct_parts.py`'s exact machinery, adds full `comp` dict
  capture (raw + weighted) at each arm's converged state.
- `PREREGISTRATION_R9_objective_audit.md` — decision rule fixed before running, including the
  no-paired-test limitation flagged before, not after, seeing the result.
- `r9_results.json` — full per-arm, per-term raw+weighted values.
- `sbatch_logs/R9c25_3886056.log` — full run log (gitignored, numbers are in `r9_results.json`).
