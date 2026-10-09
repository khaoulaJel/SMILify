# Pre-registration — C11 circumferential hard-negative mining (job 3243639)

Fixed **before** the run. Written to disk so the bar cannot drift after seeing results.

## Rationale

The LRF check (`lrf_wellposedness_20260827.py`, metric toy-verified via `verify_metric_on_toy.py`)
found `circum_gap = (l2-l3)/l2` of **0.38–0.44** on real `tr`/`fe` neighbourhoods against a perfect
cylinder's finite-sample floor of **0.031** — 12–14× above it. The circumferential disambiguating
signal is **present in raw geometry**. Yet circumferential retrieval sits near chance.

Therefore the deficit is **present-but-unlearned**, not unrepresentable — a training problem, not an
architecture problem. Uniformly sampled negatives are almost all on other segments and are solved
long before circumferential detail matters. C11 replaces half the negative budget with vertices in
the **same segment at a similar axial position** — i.e. differing mainly circumferentially.

Pools cover 4701/10235 vertices, mean 22.5 candidates each. Everything else is identical to C3
(same corpus, same held-out split, same B2 warm start, 120 epochs), so the comparison is unconfounded.

## Pre-registered success criterion

**PASS iff circumferential gap-closed on `tr`/`fe` improves by ≥ 0.05 absolute over C3, with a
paired sign test p < 0.05** across (specimen-pair, segment-class) units, measured by
`eval_C3_vs_B2_retrieval_20260827.py` on the same 12 held-out specimens.

C3 baseline: **tr 0.317, fe 0.280**. So PASS requires tr ≥ 0.367 and/or fe ≥ 0.330.

Reported alongside sign test, Wilcoxon and paired-t, with the outlier-excluded delta beside the
full-sample one, per standing project rule.

## Declared in advance

- Axial gains do **not** count. Axial was never the problem.
- A drop in axial or overall retrieval is a **cost** and must be reported even if circumferential improves.
- `ta`/`pt` are excluded from the criterion: C7 showed correspondence identity on `ti`/`ta` is
  irrelevant to the fitter, so improving them would buy nothing.
- If C11 fails, it is a negative result about hard-negative mining specifically, **not** evidence
  that the circumferential signal is unlearnable — and must not be cited as such.
