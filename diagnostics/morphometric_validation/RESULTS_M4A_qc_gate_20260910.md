# M4-A — the QC gate on all 757: admissible set FROZEN

Run 2026-09-10 against `PREREGISTRATION_M4_corpus_analysis.md` §1. No biology was run. The
admissible set defined here is frozen and is not revisited by M4-B/C/D/E.

## The fit

All **757** specimens fitted under unmodified `D1_PROD` (`submit_M4_fit757.sbatch`), 2 GPUs,
**1h52m39s** wall. Zero chunk failures; 12/12 chunks produced `Stage_3_deform_fine.npz`, 757
specimens recovered.

## The gate

| trait | admissible | rate |
|---|--:|--:|
| **HW** | **753 / 757** | **99.5%** |
| **HL** | **711 / 757** | **93.9%** |

HL failure classes: `AP_inversion` **46**, `extraction_failure` **0**.

**M3b's prediction holds at corpus scale.** The 40-specimen audit found HW robust 40/40 and HL
failing ~7.5% by head-axis inversion; the full corpus gives **HW 99.5%, HL 93.9% (6.1% inverted)**.
The headline M4-A result is therefore confirmed on the full corpus:

> **HW is robust to the dominant real-corpus head-orientation failure that compromises HL.**

HW and HL are **not** interchangeable primaries, despite both clearing M3 on the 20 *Atta*. HW is a
Type III part extent computed after rigid alignment and does not depend on landmark AP ordering; HL
*is* the distance between the two landmarks that invert.

Distributions (admissible specimens only), dimensionless against WL:

| | median | IQR | 5–95% |
|---|--:|--:|--:|
| HW/WL | 0.637 | [0.533, 0.792] | [0.393, 1.129] |
| HL/WL | 0.557 | [0.465, 0.685] | [0.342, 0.979] |
| head lateral ratio | 1.007 | [0.992, 1.023] | [0.839, 1.160] |

The head straddles its midline at 1.007 median across 753 specimens — the cleanest number in the
audit chain.

## Failure structure — the pre-registered test, and it is POSITIVE

§1 fixed in advance that the gate is not a filter and that failure must be tested for taxonomic
structure before any exclusion. It is structured:

| level | groups (n≥5) | p | verdict |
|---|--:|--:|---|
| **genus** | 46 | **0.025** | **structured** |
| subfamily | 9 | 0.621 | no signal detected at this coverage |

**Test used.** Permutation test — 20 000 shuffles of the inversion labels across the tested
specimens, χ² as the test statistic, seed 20260910. An asymptotic χ² p-value would be invalid at
these counts (30 inversions over 46 groups, 29 of them zero, so most expected counts fall far below
5) and is therefore **not reported**; see the implementation note in `m4a_qc_gate.py` for why the
asymptotic form was rejected during development.

Most-affected genera: *Blepharidatta* 2/5 (40%), *Eciton* 3/9 (33%), *Parasyscia* 2/7 (29%),
*Tapinoma* 1/5, *Cyphomyrmex* 2/10, *Gnamptogenys* 1/5, *Nomamyrmex* 1/5 (20%), *Tetramorium*
7/37 (19%).

**Reading, per §1.** Head-axis inversion is concentrated in particular **genera** but does **not**
aggregate to subfamily (p=0.62) — it is not a deep-clade property. This is reported as a scientific
finding in its own right: a set of body plans the fitter systematically mis-orients. Consequently,
**every downstream analysis using HL must declare these genera under-represented in its admissible
sample.** They are not silently dropped, and the HL results for them are a failure class, not
missing data.

Caveat on the subfamily test: `GENUS_TO_SUBFAMILY` leaves **210 of 757 (28%)** specimens UNKNOWN,
so the subfamily null is weaker than the genus positive. It should be read as "no subfamily signal
detected at this coverage", not "subfamily structure excluded".

## Consequence for M4-B/C/D/E

- **HW analyses**: n=753, effectively the whole corpus, unbiased.
- **HL analyses**: n=711, with a **known genus-structured** 6.1% exclusion that must be stated
  wherever HL enters a taxonomic comparison.
- The admissible set is frozen at `out/m4a_admissible_set.json` (per-specimen flags
  `HW_admissible`, `HL_admissible`, `HL_class`) and is not recomputed downstream.
