# Pre-registration — C12: did hard-negative mining change `ti`'s ROLE in the fitter?

Fixed **before** generating correspondences or submitting any fit.

## The question

C7 established that the CSE head does **two different jobs**: genuine correspondence on `tr`/`fe`,
and a coarse regional pull on `ti`/`ta` — shuffling `ti`/`ta` retrieved vertices *within segment*
cost nothing (`+0.0010`, sign p = 1.0, Wilcoxon 0.91) while removing them cost ~0.02.

C11 (circumferential hard-negative mining) failed its own circumferential bar, but moved `ti` direct
retrieval from **at chance** (0.340 vs 0.339 random) to **clearly beating chance** (0.246).

So: has `ti` changed from supplying "any plausible point in the region" to supplying "the actually
correct vertex"? That is a categorical change in mechanism, not a metric wiggle — and it is exactly
the distinction C7's shuffle control was built to detect.

## Arms (synth_clean, identical D1 recipe, same corpus/split as C7 so results are comparable)

| arm | `ti` treatment | correspondences from |
|---|---|---|
| `C12_c11_all` | real predicted identity | C11 |
| `C12_c11_shufti` | replaced by random same-segment vertex | C11 |
| `C12_c11_noti` | no `ti` constraint at all | C11 |

Only `ti` is manipulated. All other segments are identical across the three arms, so the contrast
isolates `ti`'s contribution.

## Pre-registered readings

**MECHANISM CHANGED — `ti` now supplies genuine correspondence.** Requires BOTH:
- `real` beats `shuffled` with paired sign p < 0.05 (i.e. destroying identity now *costs* something), AND
- `real` beats `no-ti` (the segment still contributes at all).

**MECHANISM UNCHANGED — retrieval improved without the fitter caring.** `real ≈ shuffled`
(sign p ≥ 0.05, |Δ| < 0.005), while both beat `no-ti`. This would be the third documented instance
this session of retrieval gains failing to predict fitter gains, and is a genuine finding, not a null.

**SEGMENT CONTRIBUTES NOTHING.** `real ≈ shuffled ≈ no-ti`. Would contradict C7 and require
re-examination of that result rather than acceptance of this one.

## Declared in advance

- Metric: `per_specimen_seg_acc` from the correspondence audit, the same measure C7 used. Reported
  with sign test, Wilcoxon and paired-t together, plus the outlier-excluded delta.
- The **direct comparison to C7's `+0.0010` shuffle cost** is the headline number, since the whole
  question is whether that value has changed.
- Absolute fscore/seg_acc gain versus zero-init is secondary here and must not be substituted for
  the shuffle contrast if the shuffle contrast is null.
- A positive result on `ti` says nothing about `ta`, which remains below chance.
