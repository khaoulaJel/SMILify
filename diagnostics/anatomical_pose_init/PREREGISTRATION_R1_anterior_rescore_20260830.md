# Pre-registration — R1: did the shipped scale bound fix the anterior, or only its statistics?

**Written 2026-08-30, before the re-scoring runs.** New series letter `R` (representation);
`A`–`H` are in use by earlier threads in this directory.

This arm tests a **representation bound**, not a data-term or initialisation intervention. That
choice is not stylistic — see §1.

---

## 1. Why this arm, and why it is the one worth running

`diagnostics/moonshot/REPORT.md` §6.10 records a domain observation from Fabian, made from
renders, and then quantifies it:

> *"the head is simply not moved forward enough so the mandibles and antennae explode to form the
> head, while the head shrinks into the thorax."*

| group | joints | constrained axes | deform ratio vs thorax |
|---|---|---|---|
| thorax (`b_t`) | 1 | 3/3 | 1.00× (reference) |
| **head** (`b_h`) | 1 | **0/3** | **0.72×** |
| gaster (`b_a_1…5`) | 5 | **0/15** | 1.05× |
| waist (`w_*`) | 4 | **0/12** | — |
| mandible (`ma_*`) | 2 | 6/6 | 1.24× |
| antenna (`an_*`) | — | — | **1.72×** |
| each leg | — | — | 1.05× |

Three mechanisms were named in §6.10. Two have been tested:

- **rotation limits (E1)** — the authored limits cannot reach the anterior *by construction*:
  30 of 33 body axes are unconstrained. NULL.
- **anterior partition (E3, `--split_anterior`)** — head/mandible/antenna given their own
  partition groups. Head ratio **0.72× → 0.73×**. NULL on the pre-registered outcome; mandible
  got worse.

The third was never tested: **per-joint scale, which nothing bounds.** The stock baseline lets an
antenna joint span **889×** and a head joint **138×**, and the only terms touching
`log_beta_scales` anywhere are the L/R symmetry tie and an opt-in L2.

**Since §6.10 was written, a bound on exactly that quantity shipped.** `w_scale = 0.052`
(scale_cap) is in `D1_PROD.yaml`, promoted on a pre-registered 838-specimen A/B
(`diagnostics/SHIPPED_RECIPE.md`). Its validated claim is *"anterior joint-scale outlier tail
measurably reduced — max ratio down 58–70% head/mandible/antenna."*

**It was never scored against the symptom it would be expected to fix.** The A/B measured genus
classification, five integrity scalars, and the anterior *scale ratios*. It did not compute the
§6.10 **head-carried deform ratio**.

So the question is not speculative and needs no new mechanism:

> Did the representation bound correct the anatomical deformation, or did it only make the scale
> statistics look healthier?

This also sits on the strongest pattern the project has recorded. REPORT.md §6.10.1, on its own
evidence and before any of the correspondence work:

> Three pre-registered nulls in a row — rotation limits, soft partition, anterior partition —
> **all of them interventions on how the data term is shaped.** The one intervention that has ever
> moved correspondence (removing `w_beta_prior`) acted on **what the model is allowed to
> represent.**

R1 is a test *inside* the winning family, on an intervention that already ships.

## 2. Prior evidence, declared before the bar

I have already looked at `ab_scale_cap/integrity_anterior_full_corpus.json`, so the bar below is
set with this in hand rather than in ignorance of it. **Anterior joint-scale ratio, head, n = 838
per arm** (A = no scale_cap, B = `w_scale 0.052`; the four values are mean / median / p95 / max):

| arm | mean | median | p95 | max |
|---|---|---|---|---|
| A (baseline) | 1.6069 | 1.3874 | 2.1849 | 8.9114 |
| B (scale_cap) | 1.5290 | 1.3853 | 2.1168 | 3.7735 |
| Δ | **−4.9%** | **−0.15%** | −3.1% | **−57.7%** |

This is a **tail-only signature**: the maximum falls 58% while the median does not move at all.
It biases me toward expecting FAIL or PARTIAL on the endpoint below. Declaring it here so the bar
is not tuned to it after the fact, and so a FAIL cannot later be presented as a surprise.

It is **not** the test. These are joint-*scale* ratios; the endpoint is the deform-magnitude
head-carried ratio, a different quantity measured on the fitted surface.

## 3. Data availability — the blocker, stated before the method

The endpoint needs per-specimen `deform_verts`. **The A/B outputs in this repo do not contain
them.** `ab_scale_cap/out_A/` and `out_B/` hold only `analysis.json` (genus/subfamily signal,
controls, ICC, cross-corpus) and one figure. The raw fits were written to JSC scratch
(`/p/scratch/cias-7/jellal1/...`) and are not on this branch.

`integrity_anterior_full_corpus.json` stores `deform_mag_mean` **globally, not stratified by
anatomical group** — even though `SCHEMA.md`'s own secondary endpoint specified *"stratified by
SDF/skinning-weight anatomical region"*. The stratified form was not retained.

Three routes, in order of preference:

- **(a) Recover the JSC scratch fits.** Pure re-scoring, zero GPU. Preferred if they survived
  purge. Verify count = 838 per arm before scoring anything.
- **(b) Refit a matched subset.** The 81 `ALL_ANTS_CLEAN` specimens, both arms, `D1_PROD.yaml` vs
  `D1_PROD_SCALECAP.yaml`, single variable. Two short jobs. Underpowered relative to 838 but a
  genuine paired test; **must be reported as n = 81, not presented as the A/B result** — the
  standing rule from `SCHEMA.md` ("do not substitute a smaller corpus and call it this result").
- **(c) Local fits.** 126 `Stage_3_deform_fine.npz` exist under `diagnostics/`, but they are
  scattered smoke tests and task arms with no matched A/B pair. **Not usable for this endpoint**;
  listed only to record that it was checked.

If none of (a)–(c) is available, R1 does not run and this document stands as an unrun
registration. It is not to be quietly rescoped onto a different endpoint.

## 4. Method

Per specimen, per anatomical group *g*:

```
r_g = deform_mag_mean(g) / deform_mag_mean(thorax)
```

`deform_mag_mean(g)` is the mean L2 norm of `deform_verts` over vertices assigned to *g*.
Grouping comes from `fitter_3d.trainer_hierarchical.vertex_groups(weights, joint_names,
split_anterior=True)` — the existing function, so head/mandible/antenna are separate groups and
`an_*` is not silently pooled into `body` (the exact error §6.10 records an earlier pass making).

Paired A vs B on the same specimen. Report sign test, Wilcoxon, and paired t.

## 5. The bar — fixed before the run

**Primary endpoint: `r_head`, arm A → arm B.**

- **PASS** — `r_head` rises by **≥ 0.10 absolute** (0.72 → ≥ 0.82, i.e. ≥ 36% of the gap to 1.0
  closed), with **sign p < 0.05** paired.
- **PARTIAL** — `r_head` rises with sign p < 0.05, but by **< 0.10**.
- **FAIL** — no significant rise, or a fall.

**Mechanism check, reported alongside the endpoint (standing rule from the 2026-08-28 session).**
Report `r_antenna` (1.72× baseline) and `r_mandible` (1.24×) in the same pass. A rise in `r_head`
bought by *further* inflating the appendages is a renormalisation, not an anterior correction.

> If `r_head` and `r_antenna` rise **together**, the result is reported as a global deformation-scale
> shift and **not** as an anterior fix — regardless of what the primary endpoint does.

The intended anatomical direction is `r_head` → 1.0 **and** `r_antenna` → 1.0 from above.

**Controls.**

- **Void control (definition fidelity).** Compute `r_head` on arm A alone. It must reproduce
  §6.10's **0.72×** within ±0.03. If it does not, this grouping/definition is not the one §6.10
  used, the numbers are not comparable to it, and **the arm is VOID** — no verdict is issued and
  the discrepancy is investigated first. (Precedent: F4's first implementation was recorded VOID
  rather than patched.)
- **No-op control.** `r_thorax` must be exactly 1.000 by construction. `r_leg` should reproduce
  ~1.05×. A deviation in either means the deform field or the grouping is being read wrongly.

## 6. What each outcome rules out

- **PASS** → representation bounds are the mechanism behind a whole-animal failure, demonstrated
  on an intervention **already in production**. The next arm is a stronger anterior-scoped bound
  (the current one is global), and the global-deformation-prior path has its first concrete
  working instance rather than only a literature argument.
- **PARTIAL** → the bound acts in the right direction but is too weak or too global. Same next
  arm, weaker prior.
- **FAIL** (scale tail down 58%, symptom unmoved) → this is the **seventh** instance of the
  session's central pattern: a component improving on its own measure while the thing that measure
  stood for does not. It would also mean the anterior defect is **neither partition nor scale** —
  all three of §6.10's named mechanisms exhausted — and the anterior becomes a genuinely open
  problem requiring a learned global prior rather than a hand-specified bound. That is a more
  expensive path, and it should not be started until this cheap arm has ruled the alternative out.

## 7. Scope limits, stated rather than glossed

- `r_head` is a **deformation-magnitude ratio on the fitted surface**. It is not a correspondence
  measure. A PASS says the bound corrected the deformation symptom; it says **nothing** about
  whether correspondence improved. Any correspondence claim needs `run_audit.py` scoring as a
  separate arm and is not registered here.
- The A/B corpus is `worker_ALT` (757) + `ALL_ANTS_CLEAN` (81), real scans with **no
  correspondence ground truth**. Same honest limit as the bench50 work: geometric and
  deformation-structural quantities only.
- §6.10's 0.72× was measured on the moonshot arms (`LIM_1x`), not on `D1_PROD`. The void control
  in §5 exists precisely because a baseline drift between those recipes would invalidate the
  comparison to §6.10, and must be caught before any verdict rather than explained afterwards.
- `probe_22_anterior_blindspot_PROBE.py`, which produced §6.10's table, is **not on this branch**.
  The definition above is reconstructed from the report's text plus the local `vertex_groups`
  function. The void control is the check on that reconstruction.
