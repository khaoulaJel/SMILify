# C4 — CSE predicted correspondence fed to the fitter as direct vertex identity

**Date:** 2026-08-27 · **Arm:** `CSE_correspondence` · **Job:** 3214121
**Recipe:** identical to `SYN_clean_zero_wsl` / `Dense_GT_oracle` (zero pose init, D1_low), differing
ONLY in the added `--cse_correspondence_from` term.
**Audit:** 59 → 60 rows, 0 lost, 0 duplicates. All numbers computed live from
`out/correspondence_confusion.json`, none copied from a frozen document.

## Result — seg_acc, n=12 paired specimens, baseline 0.8201

| run | mean | Δ | pos | sign | wilcoxon | paired-t | outlier-excl Δ |
|---|---|---|---|---|---|---|---|
| **CSE_correspondence** | **0.8657** | **+0.0456** | 9/12 | 0.146 | **0.0122** | **0.0337** | +0.0208 |
| Network_correspondence (centroid) | 0.8488 | +0.0287 | 9/12 | 0.146 | 0.0522 | 0.122 | +0.0128 |
| GTLabel_IK_oracle (centroid, GT-fed) | 0.8522 | +0.0321 | 11/12 | 0.0064 | 0.0024 | 0.0908 | +0.0090 |
| *Dense_GT_oracle* (oracle reference) | 0.8745 | +0.0544 | 12/12 | 0.0005 | 0.0005 | 0.0128 | +0.0304 |
| *GT_as_fitted_ceiling* (measured ceiling) | 0.9769 | +0.1568 | 12/12 | 0.0005 | 0.0005 | 6.4e-05 | +0.1270 |

### What this establishes

1. **84% of the oracle's gain, from predictions.** +0.0456 against `Dense_GT_oracle`'s +0.0544,
   without any ground truth at inference, and from only `tr`/`fe` (~10% template-vertex coverage).
2. **Predicted-direct beats GT-fed-converted.** `CSE_correspondence` (0.8657) exceeds
   `GTLabel_IK_oracle` (0.8522), which is fed TRUE labels but pushed through the centroid/IK
   conversion. An arm with worse correspondence information and a better consumption mechanism beats
   an arm with perfect information and the biased one. This is the cleanest confirmation that the
   CONVERSION was the bottleneck — independently predicted by the `tr` inversion in the retrieval
   probe (worst segment under conversion, best under direct retrieval) and by the measured ~41%
   gain loss between `GTLabel_IK_oracle` and `Dense_GT_oracle`.
3. **Beats the centroid-route network arm** by +0.0169 on an otherwise identical recipe.

### Limits, stated plainly

- **The sign test is NOT significant** (9/12, p=0.146) while Wilcoxon (0.0122) and paired-t (0.0337)
  are. The gain is carried by magnitude on a subset of specimens rather than a uniform win —
  materially weaker than `Dense_GT_oracle`'s 12/12. Two of three tests, not three of three.
- **Outlier-excluded Δ falls to +0.0208**, under half the full-sample figure. The effect is real but
  smaller than the headline.
- **leg_acc is not moved**: +0.0657 nominal, but **−0.0004 outlier-excluded**, sign p=0.388. Every
  arm here shows the same leg_acc null, `Dense_GT_oracle` included. No leg-level claim is made.
- Coverage is partial and segment-restricted **by design** (`tr`/`fe` only, on held-out evidence
  fixed before this fit). `co` weak, `ti` at chance, `ta` worse than chance were all excluded.
- Synthetic corpus, shared topology. Nothing here transfers to bench50 real scans, where the
  recorded distribution-mismatch failure mode still applies.

## Negative result carried forward

The pre-registered confidence check FAILED: cosine similarity between query and retrieved key is
**anti-correlated with correctness** — it retained `ta` at 95.7% and `ti` at 94.5% (the two segments
at/below chance) while suppressing `tr` to 55.6% and `co` to 18.2% (the reliable ones). Confidence
filtering was therefore disabled rather than swapped for another unvalidated heuristic. Finding a
usable confidence signal is open work, and matters for any deployment where segments cannot be
whitelisted from held-out data in advance.

## Engineering discipline

`--cse_correspondence_from` / `--w_cse_correspondence` are additive and flag-gated. Verified
**byte-identical with the flag off** across all four hierarchical stages, every array,
maxdiff exactly 0.0 — measured against a stashed baseline, not asserted.
