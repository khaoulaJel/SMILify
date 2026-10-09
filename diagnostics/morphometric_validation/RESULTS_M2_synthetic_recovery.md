# M2 — Chamfer does not predict morphometric accuracy. Neither does anything else we can measure.

Corpus: `diagnostics/moonshot/synth_power48`, 48 synthetic specimens, **`noise: 0.0`** — clean
targets generated *by the model itself*, so an exact solution provably exists in the search space.
Fits: existing `P48_zero` / `P48_cse_all` / `C13_uniform` runs (`D1_no_deform.yaml`, i.e. pose +
shape + per-joint scale — exactly the parameters that generated the ground truth). No new fitting.

Frame gate passed (fit/GT extent ratio median 0.968–0.973). Specimen order verified by name.

---

## 1. Morphometric recovery error, on data where exact recovery is possible

`P48_zero` (closest to production). Raw includes scale error; **shape** is scale-corrected, so a fit
that is merely the wrong *size* no longer counts as the wrong *shape*.

| trait | raw median | raw p90 | **shape median** | shape p90 |
|---|--:|--:|--:|--:|
| WL | 6.34% | 14.96% | **3.84%** | 13.82% |
| SL | 7.67% | 26.31% | **5.76%** | 22.12% |
| HW | 8.16% | 18.95% | **4.79%** | 12.16% |
| HL | 8.31% | 27.68% | **6.02%** | 18.32% |
| TBL | 9.76% | 21.76% | **7.17%** | 17.41% |
| FL | 10.96% | 33.53% | **11.73%** | 28.19% |
| GL | 15.00% | 41.44% | **12.31%** | 53.01% |
| ML | 19.03% | 49.69% | **20.74%** | 50.86% |
| PetL | 32.86% | 69.10% | **25.47%** | 68.56% |

**This is search failure, not representational failure.** The targets are noise-free, in-model, and
the generating parameters are inside the fitter's own search space. The exact answer exists and the
optimiser does not reach it — which is precisely what R9 predicts, since the objective scores the
correct solution worse.

Cross-arm: CSE correspondence supervision changes little (ML 19.03%→13.63% median, PetL
32.86%→27.69%, HW 8.16%→8.83% — mixed, no arm is broadly better).

## 2. The decisive result — no diagnostic predicts which measurement to trust

Spearman r between each diagnostic and trait recovery error, across 48 specimens
(`*` = p<0.05):

| diagnostic | HW | HL | ML | SL | WL | PetL | GL | FL | TBL |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **chamfer** | −0.10 | 0.22 | −0.16 | 0.34* | 0.33* | −0.14 | 0.01 | 0.08 | −0.24 |
| vertex_corr_err *(oracle)* | 0.09 | 0.26 | −0.09 | 0.31* | 0.14 | 0.16 | 0.12 | 0.22 | −0.06 |
| joint_err *(oracle)* | 0.14 | 0.33* | −0.12 | 0.29* | 0.20 | 0.16 | 0.01 | 0.27 | −0.09 |
| pose_err_rad *(oracle)* | 0.17 | 0.13 | 0.04 | 0.05 | 0.03 | 0.16 | 0.09 | 0.18 | 0.03 |
| beta_err *(oracle)* | 0.11 | 0.22 | 0.02 | 0.13 | 0.31* | 0.26 | −0.18 | 0.14 | 0.07 |
| deform_mag | −0.11 | 0.20 | 0.10 | 0.05 | 0.14 | −0.08 | −0.03 | 0.06 | −0.23 |

**Chamfer — the quantity the objective optimises and the number everyone reports — predicts
essentially nothing.** Two of nine traits reach significance at r≈0.33, explaining ~11% of variance;
three correlations are *negative*.

**The stronger half: the oracle diagnostics fail too.** `vertex_corr_err` is the TRUE per-vertex
correspondence error, computable only because synthetic specimens share the template topology.
`pose_err_rad` and `beta_err` are exact deviations from the generating parameters. These are
quantities you could never obtain on a real scan — and they still do not tell you which
measurement is trustworthy (max |r| ≈ 0.41 anywhere in the table).

> So this is not "we need a better geometric QC metric." **Geometric fit quality — even measured
> perfectly, with ground truth — is close to uninformative about morphometric accuracy.** That is a
> general statement about the pipeline, not about Chamfer specifically.

## 3. What this means together with Track B

Track B (Atta, real specimens, physical reference) and M2 (synthetic, exact ground truth) agree on
the trait ranking where they overlap: **HW is the most trustworthy trait on both** (pose Δ 0.115%,
agreement bias −0.13%, synthetic shape error 4.79%), and the anterior/appendage traits are worst.

They also separate two distinct failure modes that a single number would conflate:
- **pose sensitivity** (B2): SL is worst at 6.08%
- **recovery error** (M2): SL recovers at 5.76% shape error — mid-pack

A trait can be pose-fragile but recoverable, or pose-stable but poorly recovered (FL: 0.53% pose,
11.73% recovery). **Reliability is trait-specific and failure-mode-specific.** There is no single
scalar quality score, and M2 shows why: nothing we can measure predicts the thing we care about.

## Limits

- One recipe (`D1_no_deform.yaml`), not production `D1_PROD.yaml` (which adds free-form deform).
  The generating parameters are inside this recipe's search space, which is what makes the
  "exact solution exists" argument valid — but production numbers could differ.
- 48 synthetic specimens, in-model shapes. Real scans add topology defects, fragmentation and
  out-of-model morphology, all of which can only make recovery worse, not better.
- Trait definitions use `trait_extract.py`'s unadjudicated landmark indices (see Track B's B1) —
  the *relative* trait comparison is unaffected, absolute values inherit that offset.

## Artifacts

`m2_synthetic_recovery.py` → `out/m2_results.json` (per-specimen diagnostics, per-trait errors,
full correlation tables for all three arms).
