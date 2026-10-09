# CSE feasibility on the frozen B2 backbone — retrieval accuracy, axial vs circumferential

**Date:** 2026-08-26 · **Script:** `cse_feasibility_retrieval_20260826.py` ·
**Output:** `out_cse_feasibility_20260826/cse_feasibility_retrieval.json` ·
**Log:** `out_cse_feasibility_20260826_run.log`
**Cost:** zero GPU. Frozen checkpoint, eval mode, forward passes only, run on CPU.
**Checkpoint:** `out_B2_correspondence_net_20260826/best_model.pt` (epoch 91, val_total 0.4229,
trained n_points 2048, 37 classes — taxonomy asserted identical to the rebuilt one at load time).
**Data:** 12 specimens from the **held-out** val tail of `synth_b2_train_corrb06.npz`
(`val_frac=0.05` — the same split B2 trained under, so these meshes were never trained on).

---

## Why this ran

`PROGRESS_SYNTHESIS_20260826.md` Part 5 pre-registered three things that had to happen before any
GPU-heavy CSE retrain, because the only encouraging signal (r=0.547) was a distance correlation on
44 points, one band, one segment (`fe`), one leg, one specimen, via a coarse feature proxy:

  (a) cover `tr`, the worst offender everywhere else — **done**, and it is the headline result;
  (b) measure real nearest-neighbour **retrieval accuracy**, not a linear correlation — **done**;
  (c) replicate across specimens — **done**, 12 held-out specimens.

It also removed the approximation for free: `smil_correspondence_net.py:137` shows both heads decode
from `l0_points = fp1(...)`, shape (B,128,N) — a **true per-point** feature. The r=0.547 probe used
each point's nearest FPS-centroid from `sa1`'s 512 centroids. This probe reads `l0_points` directly,
so it measures the exact tensor a CSE head would sit on. No retrain required to fix this.

## Two facts verified live (not assumed)

1. **`chain_pos` is degenerate, reproduced in current code.** Class 1 (`l1_r_co`) has
   `min == max == 0.0` — constant within the class. This confirms the settled finding directly.
   Consequence for this probe: the axial/circumferential split is derived from **posed-geometry PCA**
   of each segment's own vertices, *not* from `chain_pos`, which would have been identically zero.
   (It silently zeroed every segment length on the first run — caught, diagnosed, fixed.)
2. **Metric floor and ceiling are measured, never assumed** (per the 0.974-not-1.0 correction):
   a RANDOM same-segment partner gives the chance floor, and a nearest-neighbour partner gives the
   achievable ceiling, which is **not** zero error.
   `gap_closed = (rand − feat) / (rand − ceiling)`: 1.0 = at the achievable ceiling, 0.0 = chance.

## Pre-registered criteria (fixed before the first run)

- **P1** feature-NN beats the random floor, paired sign test p<0.05 (Wilcoxon + paired-t also reported).
- **P2** feature-NN closes ≥25% of the random→ceiling gap. **PASS = P1 ∧ P2.**
- **P3** (directional, SurfEmb-derived, does **not** gate PASS): axial gap-closed > circumferential.

---

## Result 1 — within-specimen retrieval looks near-perfect, and is MISLEADING

| seg | n | feat | rand | 3dNN | gap_closed | axial | circ | sign p |
|-----|---|------|------|------|-----------|-------|------|--------|
| tr | 72 | 0.053 | 0.334 | 0.043 | **0.963** | 0.985 | 0.846 | 4.2e-22 |
| ti | 31 | 0.080 | 0.350 | 0.072 | **0.971** | 0.981 | 0.786 | 9.3e-10 |
| fe | 72 | 0.070 | 0.325 | 0.057 | **0.950** | 0.985 | 0.782 | 4.2e-22 |
| ta | 28 | 0.142 | 0.325 | 0.130 | **0.938** | 1.012 | 0.685 | 2.2e-07 |
| co | 72 | 0.140 | 0.447 | 0.113 | **0.920** | 0.956 | 0.886 | 4.2e-22 |

All five PASS at ~0.92–0.97 of the achievable ceiling. Outlier-excluded companions are effectively
identical (e.g. `tr` 0.968 with 10/72 dropped; `co` 0.929 with 5/72 dropped), so this is not an
outlier artifact. (`ta` axial 1.012 > 1 is a decomposition artifact of taking a ratio of means on a
projected component, not genuine superiority to the ceiling.)

**This number must not be reported as CSE feasibility.** Within-specimen retrieval can be aced by
features that merely encode **absolute position in the current pose**, which is useless for
correspondence. That is why the cross-specimen test below was added — and it changes the picture.

## Result 2 — cross-specimen retrieval, THE decisive test

Query point on specimen A → feature-nearest point on a **different** specimen B (different pose
*and* different shape). Error is measured between the two **true template vertices in the canonical
rest frame**, so it is pose-independent by construction.

| seg | n | feat | rand | tmplNN | **gap_closed** | **axial** | **circ** | sign p | wilcoxon | t-test | verdict |
|-----|---|------|------|--------|---------------|-----------|----------|--------|----------|--------|---------|
| **tr** | 72 | 0.177 | 0.346 | 0.039 | **0.551** | 0.567 | 0.090 | 3.1e-20 | 1.9e-13 | 2.7e-31 | **PASS** |
| **ti** | 19 | 0.182 | 0.347 | 0.045 | **0.545** | 0.542 | 0.107 | 7.3e-04 | 5.3e-05 | 2.3e-05 | **PASS** |
| **fe** | 72 | 0.224 | 0.320 | 0.044 | **0.348** | 0.345 | 0.144 | 7.0e-13 | 5.7e-09 | 7.2e-11 | **PASS** |
| **co** | 72 | 0.383 | 0.478 | 0.118 | **0.265** | 0.338 | 0.200 | 5.8e-12 | 9.2e-12 | 2.2e-17 | **PASS** |
| **ta** | 13 | 0.291 | 0.327 | 0.063 | 0.138 | 0.140 | −0.153 | 2.7e-01 | 2.4e-01 | 3.2e-01 | **FAIL** |

Performance roughly **halves** relative to within-specimen. The flattering number was inflated, as
suspected. 4 of 5 segments still clear the pre-registered bar on all three tests.

### Two findings that matter

**(i) `tr` is the BEST cross-specimen segment (0.551, p=3e-20).** `tr` has been the *worst* offender
in every previous check in this investigation (2.03×→3.85× bone-length bias under centroid and
furthest-point conversion). That inversion is strong evidence the earlier `tr` failures were caused
by **the conversion/decode step, not by the backbone's features** — consistent with the settled
conclusion that centroid conversion carries a real bias, and with the falsification of the
furthest-point "smarter statistic" family.

**(ii) The SurfEmb prediction holds on all 5 segments, with a large margin.**
Axial gap-closed 0.14–0.57 versus circumferential **0.09–0.20** — circumferential is close to
chance everywhere, and *negative* on `ta`. The backbone localises **along** the limb but essentially
**not around** it. This is precisely the near-rotational-symmetry ambiguity SurfEmb
([arXiv:2111.13489](https://arxiv.org/abs/2111.13489)) identifies as the failure mode of a
*deterministic* per-point embedding, and it independently corroborates the earlier finding that a
leg segment is a 2-D tube whose circumferential dimension no scalar can address.

The prediction was made **from the literature before the probe was written**, then confirmed —
not fitted after the fact.

---

## Verdict, and the one named fix

**CSE-family embedding head is supported as the next step**, on evidence rather than hope:
cross-specimen retrieval is significantly above chance on 4/5 segments by all three tests, on
held-out specimens, with a measured (not assumed) ceiling.

The specific recipe the evidence agrees with:

- **Head:** per-point continuous embedding, matched by nearest neighbour to per-template-vertex
  embeddings — **CoE**-style ([arXiv:2412.05557](https://arxiv.org/abs/2412.05557), 3DV 2025), which
  is point-cloud-native, unlike image-based CSE ([arXiv:2011.12438](https://arxiv.org/abs/2011.12438)).
- **Loss:** **SurfEmb**-style contrastive (InfoNCE query/key) against true vertex identity —
  explicitly **not** plain L2/triplet regression. Justification is now empirical, not stylistic: the
  residual is *specifically circumferential*, and the contrastive-distributional form is what exists
  to handle exactly that ambiguity.
- **Unchanged:** sa1/sa2 backbone, corrected sampler, training pipeline, and the leg_acc/seg_acc +
  sign/Wilcoxon/paired-t + live-computed B3 bar evaluation harness.

### Honest limits — this justifies the bet, it does not predict success

- Cross-specimen gap-closed of **0.27–0.55 is real signal well above chance, but nowhere near the
  ceiling.** A CSE head starts from a partially-informative backbone, not a nearly-solved one.
- **`ta` FAILS** (0.138, p=0.27, n=13) — no usable cross-specimen signal on the tarsus. `pt` never
  entered the analysis at all (only 8 template vertices, below the 8-point floor). The two most
  distal segments are unsupported by this evidence and must not be claimed.
- These are **synthetic** specimens from one corpus with shared topology. Nothing here speaks to
  bench50 real scans, where the recorded gotcha stands: feeding the network a point sample whose
  distribution does not match training **breaks it outright**.
- Retrieval quality is a *precondition* for correspondence quality, which is itself upstream of
  morphometric reliability. Neither downstream link is demonstrated by this probe.

### Explicitly NOT concluded here

The two queued D1 fits (`3173467` Network_correspondence, `3175274` GTLabel_IK_oracle — both still
SLURM-pending as of this writing, estimated start 20:24) use the **centroid conversion already shown
to carry a 2.03× bone-length bias**. Whatever they produce evaluates that superseded conversion
route, not correspondence-network quality, and will be reported as historical/comparative only.
