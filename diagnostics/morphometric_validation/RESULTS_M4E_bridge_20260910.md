# M4-E — the bridge: validated scalars mark a genus-structured direction in the dense phenotype

Run 2026-09-10 per `PREREGISTRATION_M4_corpus_analysis.md` §6, on the frozen M4-A admissible set.
Scale-free on both sides, so this does not depend on the unresolved absolute-scale question. Trait
roles are those frozen in M3/M4-A (HW primary, HL secondary) and were **not** modified by M4-C.

## Setup

Dense phenotype = fitted vertices in **canonical pose** (joint rotation zeroed, everything else as
fitted), centred, scaled to unit norm, Procrustes-rotated to the template — pose and size both
removed, putting it on the same scale-free footing as the scalar ratios. 757 × 30 705, reduced to
50 PCs capturing **93.0%** of dense shape variance.

## Result

| | HW/WL (primary) | HL/WL (secondary) |
|---|--:|--:|
| dense shape predicts the scalar, R² | 0.719 | 0.746 |
| genus η² of the scalar itself | 0.331 | 0.326 |
| genus η² of its dense-aligned direction | **0.350** | 0.357 |
| best *single* unsupervised dense PC | 0.243 (PC18) | 0.273 (PC18) |
| random direction, same space (median) | 0.150 | 0.160 |
| p vs random direction | 0.0005 | < 0.0005 |

**Cross-validated** (direction fitted on a random half, η² evaluated on the held-out half, 50
splits):

| | held-out η² | random-direction held-out η² | Wilcoxon p |
|---|--:|--:|--:|
| **HW/WL** | **0.409 ± 0.044** | 0.236 ± 0.049 | 1.8×10⁻¹⁵ |
| HL/WL | 0.412 ± 0.034 | 0.238 ± 0.042 | 1.8×10⁻¹⁵ |

The in-sample version is partly circular by construction — the direction is fitted to predict a
scalar that is itself genus-structured. The cross-validated version removes that: the direction is
never fitted on the specimens it is scored on, and the effect **increases** rather than collapsing.
The cross-validated number is the one to report.

## What this establishes

Two things, and the second is the more interesting:

1. **The validated scalars are a genuine component of the dense phenotype, not something beside
   it.** Dense shape predicts HW/WL at R²=0.72 — they share most of their information, so the
   scalar is a projection of the dense shape rather than an independent measurement of it.
2. **The direction the scalar picks out is more genus-structured than any direction unsupervised
   decomposition surfaces.** The HW-aligned direction reaches η²=0.35 in-sample / 0.41 held-out,
   against a best single dense PC of 0.243 and a random-direction null of 0.15/0.24. PCA maximises
   *variance*, not taxonomic structure — so a validated morphometric identifies a biologically
   structured axis of the dense shape space that the dense representation does not hand you on its
   own.

The claim this licenses, in the wording fixed **before** the result (§6):

> **Validated interpretable morphometrics recover a measurable component of biological structure
> already present in the dense 3D representation.**

Explicitly **not** "the scalar traits explain the biological structure." The dense reference signal
is itself modest (Z11/Z12: top-1 0.179, nest-mate agreement 18.9%), and η²≈0.41 held-out leaves the
majority of dense shape variation unaccounted for by these two scalars.

## On HL

HL/WL performs comparably to HW/WL **on this axis** (0.412 vs 0.409 held-out). That does **not**
promote it, and the frozen roles are unchanged. HW and HL have separated on three prior independent
axes — M4-A admissibility (99.5% vs 93.9%), the genus-structured HL exclusion (p=0.025), and M4-C's
singleton robustness check (HL's genus/species ordering reverses, HW's does not). A single axis on
which they agree does not overturn three on which they do not.

## Limitations

1. **Genus η² here is not comparable to M4-C's variance components.** This is a rank-based η² on
   genera with n ≥ 5; M4-C is a REML nested decomposition on the full hierarchy. They answer
   different questions and their numbers should not be differenced.
2. **The dense representation is the fitted model's shape space, not the scan.** It inherits every
   limitation of the registration, including the noise floor (Z6/Z7) and the anatomical-identity
   problem (Phase A). A direction being genus-structured *in the fitted shape space* is not
   evidence it is anatomically correct.
3. **Size is not controlled**, per M4-B/M4-C. Procrustes removes isotropic scale, but allometric
   shape variation remains and is not separable here.
