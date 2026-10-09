# R3 — R6/R7/R8 re-scored in the corrected frame

**2026-09-17.** Re-runs of `r6_sdf_instances_real_v2.py`, `r7_bilateral_split_v2.py`,
`r8_symmetry_plane_v2.py` in `diagnostics/gt_contamination_audit_20260916/`. All three use a
shared frame-corrected, on-surface-asserted, `gt_expert`-only landmark loader (median on-surface
error 0.121% of diagonal, vs 2.272% uncorrected — the same 19x improvement as R1). n = 11
(Dolichoderus excluded). Originals in `diagnostics/groundtruth/` are untouched, VOID, left as the
historical record. **No comparison to their numbers is made — per the standing rule, a
comparison against pre-correction numbers is meaningless. Every verdict below stands on its own.**

---

## R6 — SDF part-instance recovery: FAILS

SDF-based part segmentation does not cleanly separate appendage instances on real scans.

- Mandible L vs R in different SDF components: **2/11**
- Antenna L vs R in different components: 10/11 (this part separates fine)
- Mandible vs antenna in different components: 9/11
- **All four appendage landmarks in distinct components: 1/11**
- Median component count on the thin/appendage side: 35 (vs ~9 on the clean template)

Verdict unchanged in character from the void original: template-level part purity does not
transfer to fragmented real scans. The mandibles are the specific failure — they merge into one
component on 9 of 11 specimens.

## R7 — Attachment-based lateralisation: FAILS

Even with a valid midline (R8's plane, valid on 10/11), attachment/connectivity alone cannot tell
left from right at the mandible.

- Midline B (R8 plane) valid on 10/11 specimens
- **Mandibles correctly lateralised: 1/10**
- Scapes (antenna base) correctly lateralised: 9/10

So the antenna problem is a midline problem (fixed by R8) and now resolves; the mandible problem
is not — it is the SDF merge from R6 propagating through. Fixing the reference frame did not fix
this arm's actual bottleneck.

## R8 — Body symmetry plane: **PASSES**

The one geometric signal that reliably works.

| metric | value | bar |
|---|---:|---|
| straddle (head-width landmarks split by the plane) | **10/11** | ≥9/11 |
| median reflection residual | 0.87% of scale | — |
| median plane stability (50% subsampling) | **1.4°** | — |

**Reconciled with the earlier landmark-free probe, not contradicted by it.** That probe
(`r8_roll_degeneracy_PROBE.py`, full mesh) found the reflection residual varies a median 53.5% of
its best value across roll — i.e. a preferred roll exists, the body is not rotationally
degenerate. This run measures roll-spread on the **body mask only** (legs/antennae excluded,
since R8's own plane-finding already restricts to the body) and gets 106.2% — a *stronger*
preference, consistent in direction. Both agree: **do not claim rotational degeneracy about the
long axis.** The symmetry plane itself is well-determined; that was never in tension with the
roll finding, which is about a different rotational axis (about the long axis) than the one the
symmetry plane fixes (the sagittal reflection).

---

## Net verdict for the talk

**R8 survives and is presentable as a real, quantified result: the raw scan alone supplies a
stable bilateral reference frame (10/11, 1.4° stable).** R6 and R7 do not survive — the specific
failure in both is the same structure, the mandibles, which fragment under an SDF prior and
cannot be lateralised by attachment even given a correct midline. That convergence (R6's
component-merge finding directly explaining R7's residual failure once the midline itself is
fixed) is worth stating as one coherent story rather than three separate negatives:

> **The scan supplies a reliable body-symmetry frame, but the mandibles are unrecoverable from
> local geometry: they fragment under distance-based part segmentation and cannot be
> lateralised by connectivity even once a correct midline is supplied.**

This is a narrower and more defensible version of "identity does not come from geometry alone"
than the void original claimed, and it is now quantitatively true in the corrected frame rather
than asserted from contaminated numbers.
