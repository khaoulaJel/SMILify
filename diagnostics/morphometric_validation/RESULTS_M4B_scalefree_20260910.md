# M4-B — scale-free head shape carries strong genus structure

Run 2026-09-10 on the **frozen** M4-A admissible set, per `PREREGISTRATION_M4_corpus_analysis.md`
§2. These are **shape** claims, not allometric ones. B-iii (pooled absolute allometry) remains
prohibited.

## B-i is NOT EXECUTABLE — a further narrowing forced by evidence

§2 permitted within-genus allometry "for genera with n ≥ 5, where scale is internally comparable."
That premise was checked before running and **fails**: the evidence for internal comparability was
within-*species* voxel homogeneity, and genera span species. Of the 45 genera with n ≥ 5 and any
scan metadata, **12 contain mixed voxel sizes internally** — and they are the largest ones:
*Pheidole* (n=39), *Strumigenys* (33), *Cephalotes* (28), *Dorylus* (21), *Camponotus* (16),
*Hypoponera* (14), *Acromyrmex* (10), *Leptogenys* (10), *Odontomachus* (9), *Carebara* (8),
*Neoponera* (8), *Polyrhachis* (8).

Two reasons this closes B-i rather than restricting it:

1. The premise fails exactly where the statistical power is.
2. Even a verified single-voxel genus would not rescue it — within the single 2.44 µm group the
   implied scale is still wrong by 2–3× in both directions (M3b), so voxel homogeneity is
   *necessary but not sufficient* for scale comparability.

**Absolute-scale allometry is therefore not defensible anywhere on `worker_ALT`**, at any grouping
level, until an external per-specimen size reference exists. B-ii carries M4-B alone.

## B-ii — the result

All three scale-free ratios show highly significant between-genus structure. Permutation tests
(20 000 shuffles, Kruskal–Wallis H as statistic, seed 20260910); asymptotic p not reported, same
reasoning as M4-A. Genera with n ≥ 5 only.

| ratio | n | tested | genera | H | p (perm) | rank η² |
|---|--:|--:|--:|--:|--:|--:|
| **HW/WL** | 753 | 488 | 45 | 161.0 | < 5×10⁻⁵ | **0.331** |
| HL/WL | 711 | 451 | 42 | 146.8 | < 5×10⁻⁵ | 0.326 |
| HW/HL | 707 | 443 | 41 | 114.6 | < 5×10⁻⁵ | 0.259 |

**26–33% of rank variance in validated head shape sits between genera.** For two scalar
measurements surviving a QC gate, on a corpus whose registration sits at its noise floor (Z6/Z7),
that is a substantial effect.

| ratio | median | IQR |
|---|--:|--:|
| HW/WL | 0.637 | [0.533, 0.792] |
| HL/WL | 0.557 | [0.465, 0.685] |
| HW/HL | 1.129 | [1.086, 1.192] |

## The extremes are biologically coherent — an independent plausibility check

Nothing in the pipeline was selected using taxonomy, so where the genera land is a free check:

- **Broadest heads (HW/WL)**: *Acromyrmex* 1.086, *Sericomyrmex* 1.016, *Mycetophylax* 0.943,
  *Cephalotes* 0.926 — attine fungus-growers and turtle ants, all genuinely broad-headed;
  *Cephalotes* famously so (phragmotic head discs).
- **Narrowest**: *Leptogenys* 0.376, *Dolichoderus* 0.504, *Neivamyrmex* 0.509, *Megaponera* 0.521
  — slender ponerines and army ants.
- **HW/HL lowest**: *Parasyscia* 1.057, *Acanthostichus* 1.058, *Cerapachys* 1.064, *Eciton* 1.080
  — dorylines with elongate heads.

The ordering recovers known head morphology without being told any of it. This is a plausibility
signal, not evidence — it was not pre-registered and is reported as a check, not a result.

## Limitations — two that matter

1. **Shape differences may partly BE size differences.** These ratios are scale-free by
   construction, but head proportions genuinely scale with body size in ants. Because absolute
   size is unidentifiable here, the between-genus shape signal **cannot be decomposed into
   "shape difference" versus "size difference expressed as shape"**. M4-D's size correction is
   therefore also constrained; the honest claim is that *validated head shape differs between
   genera*, not that it differs *independently of size*.
2. **HL results carry the M4-A exclusion.** HL's n=711 is **not** a random subset of 757: exclusion
   is genus-structured (permutation p=0.025), concentrated in *Blepharidatta*, *Eciton*,
   *Parasyscia*, *Tetramorium* and others. Any HL genus comparison inherits that, and those genera
   are under-represented. HW (n=753, 99.5%) is effectively corpus-wide and does not carry it —
   which is why **HW/WL is the primary M4-B result and HL/WL the secondary**.
