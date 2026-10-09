# V1 — pre-registration. Does the joint-position term survive on real scans, and is free
# deformation currently absorbing the anatomical error?

Written 2026-09-07, BEFORE the run. Job/results appended after.

## Why this and not the surface-vs-skeleton comparison

The proposed "does the fit beat reading a predicted skeleton directly" test is unfair by
construction on a surface endpoint: a skeleton predictor emits joint coordinates and cannot
produce a surface at all, so the baseline arm does not exist. Dropped.

What has genuinely never been measured is the SHAPE and SURFACE side of G7's arms. G7 reported
joint residual, chamfer and betas |z| only. It never asked:

1. does the joint term degrade the biologically useful measurements (T4: surface-derived traits,
   which carry 2.8x the genus signal of rig-derived ones), and
2. is the joint term being satisfied honestly, or by abusing `deform_verts` / `log_beta_scales`?

Question 2 opens a mechanism that has never been tested and that G6 makes plausible: if the
surface objective is invariant to the skeleton because free per-vertex deformation can absorb any
mismatch, then deformation is the reason the skeleton is unconstrained, and freezing it should let
the joint term reach further.

## Arms — 14 human-annotated specimens, real production objective (D1_PROD Stage_3_deform_fine),
## all starting from the production parameters, betas clamped |z| <= 1 throughout

| arm | lambda | deform_verts |
|---|---|---|
| `production` | — | as fitted (no re-optimisation; reference row) |
| `L0_free` | 0 | free — **VOIDING CONTROL** |
| `L01_free` | 0.1 | free |
| `L03_free` | 0.3 | free |
| `L10_free` | 1.0 | free |
| `L0_dfroz` | 0 | frozen at the production value |
| `L01_dfroz` | 0.1 | frozen at the production value |
| `L0_dzero` | 0 | held at ZERO |
| `L01_dzero` | 0.1 | held at ZERO |

The two lambda=0 deform arms are controls: they separate "the joint term reaches further without
deformation" from "removing deformation changes the fit anyway".

## Measured, per arm

- **joint residual** vs the human annotations (% of mesosoma length), same metric as G1/G6/G7
- **chamfer** and the full production objective
- **surface traits** via `trait_extract.traits()` — the GLAD ratios T4 identified as the
  biologically useful class: HW/HL, ML/HL, SL/HL, GL/WL, PetL/WL, HL/WL, HW/WL
- **trait plausibility** against T1's pre-set biological bounds (count of impossible values)
- **bilateral asymmetry** of ML, SL, WL, FL — the only accuracy-like signal available with no
  surface ground truth
- **shape abuse**: mean betas |z|, RMS `deform_verts` magnitude as % of mesosoma length,
  mean |log_beta_scales|

## Pre-registered bars

**VOIDING CONTROL (must pass or nothing else is readable).** `L0_free` must stay at production:
chamfer within 50% and joint residual within 5 percentage points. Fails => the sweep measures the
optimiser, not lambda, and every row is void.

**H1 — the joint term does not damage the biological surface representation.** At `L01_free`:
impossible-trait count does not increase, and median bilateral asymmetry does not rise by more
than 50% relative. PASS => Outcome A/B: the term is a real candidate improvement. FAIL => a genuine
trade-off, and the term must be staged rather than applied throughout.

**H2 — the joint term is satisfied honestly.** At `L01_free`: RMS `deform_verts` does not increase
by more than 50% relative to production, and mean |log_beta_scales| does not increase by more than
50%. FAIL => the joint targets are being met by local surface hacking, and the reported joint
improvement is not an anatomical improvement.

**H3 — free deformation is absorbing the anatomical error.** `L01_dfroz` improves joint residual
by >= 25% relative to `L01_free`, WHILE the corresponding control pair (`L0_dfroz` vs `L0_free`)
moves by < 25%. PASS => deformation is a mechanism, not a nuisance: it is what lets the surface
objective ignore the skeleton, and freezing it during the anatomical stage is a shippable change.
FAIL => deformation is not the escape route and the under-determination is elsewhere.

## What CANNOT be concluded, stated in advance

There is **no surface ground truth**. The five expert surface landmarks T4 asks for do not exist,
so trait ACCURACY cannot be scored here. V1 can show the traits stay biologically valid, stable
and self-consistent; it cannot show they get closer to truth. n = 14, one optimisation per arm,
no seed replication — treat differences under ~10% relative as noise.
