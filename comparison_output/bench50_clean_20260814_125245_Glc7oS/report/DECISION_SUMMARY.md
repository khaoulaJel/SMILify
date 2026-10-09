# alpha_wrap decision summary — all numbers, one file

Corpus: bench50_clean (50 ant specimens). Baseline run: 2026-08-14. Offset fix applied + reruns: 2026-08-16.

## 1. Original bench50 run (offset = alpha/30)

| Metric | Value |
|---|---|
| Total specimens | 50 |
| Pure `alpha_wrap` success | 28/50 (56%) |
| ManifoldPlus fallback (no formal guarantee) | 22/50 (44%) |
| Failed pipeline's own fidelity gate at gen time | 10/50 (20%) |
| Errors during analysis | 0 |
| `original` baseline not watertight | 50/50 (100%) |
| `original` baseline fragmented (>5 components) | 48/50 (96%), median 27 components, max 69 |
| alphawrap output with >1 component | 14/50 (28%) |
| Chamfer-distance outliers (>5 MAD) | 8/50 |

**Thin-feature thickness ratio (alphawrap/original, p5, all 50):** median 0.047, Wilcoxon p=1.1e-8 (catastrophic, significant collapse)
- alpha_wrap-only subset (n=28): median 0.208, p=4.2e-4 (real collapse, less severe)
- manifold_plus-fallback subset (n=22): median 0.0015, p=4.8e-7 (near-total destruction)

**Vertex count:** alphawrap median +87,848 vertices vs original, p=6e-14 (expected/by design, not a defect)

**Fallback root cause breakdown (22 specimens):**
- 15/22 explicit CGAL subprocess timeout (300s limit)
- 2/22 CGAL OOM-killed during bisection
- 5/22 bisection coarse-bound heuristic failure (`alpha_fused_start=pre_gap*15` didn't fuse the gap) — unrelated to offset, separate bug, still unfixed

## 2. Offset fix (offset_alpha_ratio 30→2.5, i.e. offset=alpha×0.4 not alpha/30)

Code change: `custom_processing/prepare_antscan_data_for_mesh_fitting_alphawrap.py`
(`_bisect_alpha_wrap_params.offset_alpha_ratio`, `_derive_alpha_wrap_params.offset_factor`)

### Ground-truth sanity check (2 specimens, `diagnostics/offset_fix_sanity_20260816/`)

| Specimen | Before | After |
|---|---|---|
| Cephalotes_spinosus_CASENT0744158 | timeout → ManifoldPlus fallback | pure alpha_wrap, 226s, gate=True |
| Dorylus_sp._CASENT0744703 (968K pre-recon verts) | (first run under new code) | pure alpha_wrap, 2293s, gate=True, gap `preserved` across all 8 bisection steps |

### Rerun of the 22 previously-fallback specimens (full bisection, `diagnostics/offset_fix_rerun_20260816/`)

| | Before | After |
|---|---|---|
| Pure `alpha_wrap` | 0/22 | **13/22 (59%)** |
| Still ManifoldPlus fallback | 22/22 | 9/22 |

Runtime for the 13 newly-fixed: 43s–1189s (median ~250s). Two (Ectatomma_brunneum, and others in
the 9 still-failing) got further into bisection before failing than before, but didn't fully
resolve — consistent with the 5 "bounds-heuristic" root-cause specimens being a separate,
unaffected failure mode.

**Thin-feature quality on the 13 newly-fixed specimens** (re-measured against same `original` baseline):

| Metric | Original batch (§1) | 13 newly-fixed |
|---|---|---|
| p5 thickness ratio, median | 0.047 (collapse) | **2.05** (no collapse) |
| Wilcoxon p | 1.1e-8 | 0.040 (opposite direction) |

### Rerun of the 28 previously-passing specimens (full bisection, `diagnostics/offset_fix_rerun28_20260816/`)

Requested check: does the looser offset regress any of the 28 specimens that already worked?
(A cheap frozen-alpha-only gap recheck was attempted first and found unreliable — OBJ
export/reimport round-trip precision loss flips ~1/3 of borderline verdicts on its own,
independent of offset; confirmed CGAL itself is deterministic. Full bisection rerun was required
for a trustworthy answer.)

| | Before | After |
|---|---|---|
| Pure `alpha_wrap` | 28/28 | **24/28 (86%)** |
| Regressed to ManifoldPlus fallback | 0/28 | **4/28 (14%)** |

Regressed specimens (all 4 still have `fidelity_gate_passed=True` even in fallback mode):
- Acanthostichus_aff.brevicornis_CASENT0744328
- Mayriella_sp._CASENT0745610
- Acromyrmex_coronatus_CASENT0744365
- Anochetus_risii_CASENT0877609

Runtime range: 18s–784s (median ~150s).

## 3. Net combined result across all 50 (offset fix applied everywhere)

| | Before (§1) | After (offset fix, both reruns) |
|---|---|---|
| Pure `alpha_wrap` | 28/50 (56%) | **37/50 (74%)** |
| ManifoldPlus fallback | 22/50 (44%) | **13/50 (26%)** |

Net: +9 specimens moved from fallback to guaranteed alpha_wrap (13 gained, 4 lost).

**Not yet re-measured:** full-corpus thin-feature/Chamfer/volume metrics for the new 37/50 split
(only the 13-newly-fixed subset was re-measured, §2). A full `compare_pipelines.py` rerun across
all 50 with the new outputs would give the equivalent of §1's numbers under the fix.

## 4. Still open, unaddressed by this fix

- 13/50 (26%) specimens still fall back to ManifoldPlus (no formal guarantee):
  - 9 from the original 22 (5 bounds-heuristic bug, 4 other/unresolved timeout-adjacent)
  - 4 new regressions from the 28 (root cause not yet investigated — could be the same
    bounds-heuristic issue newly triggered by the shifted fuse/preserve boundary, not confirmed)
- The 5/22 (and now possibly some of the 4/28) bounds-heuristic failures are a distinct bug:
  `alpha_fused_start = pre_gap*15` doesn't fuse the gap on those specimens. Separate fix, not started.

## 5. Full-batch metric suite under the fix, all 50 (closes gap noted in §3)

Run: `comparison_output/bench50_offsetfix_20260816/report/` — same `compare_pipelines.py`, same
`original` baselines as §1, paired against the new post-fix alphawrap outputs (13 from the
22-rerun, 24 from the 28-rerun, 4 regressions + 9 unresolved still on ManifoldPlus = 13 fallback).

| Metric | §1 baseline (all 50) | §5 post-fix (all 50) |
|---|---|---|
| Pure `alpha_wrap` | 28/50 (56%) | **37/50 (74%)** |
| ManifoldPlus fallback | 22/50 (44%) | **13/50 (26%)** |
| Failed fidelity gate at gen time | 10/50 (20%) | 8/50 (16%) |
| p5 thickness ratio, median (all 50) | 0.047 (collapse) | **1.000** (no collapse) |
| p5 thickness ratio, Wilcoxon p (all 50) | 1.1e-8 (significant collapse) | 0.052 (not significant) |
| p5 thickness ratio, alpha_wrap-only subset | 0.208, p=4.2e-4 (n=28) | **2.04, p=0.002 (n=37)** — reversed, now thicker not thinner |
| p5 thickness ratio, manifold_plus-only subset | 0.0015, p=4.8e-7 (n=22) | 0.0047, p=0.021 (n=13) — still catastrophic |
| Chamfer distance, median | 43.97 | 44.95 (unchanged) |
| Vertex count, median diff (aw-orig) | +87,848, p=6e-14 | +61,422, p=5.3e-9 (still by-design, smaller gap) |

**Reading:** the thin-feature collapse documented in §1 is no longer a batch-wide problem after
the fix (p=0.052, not significant, median ratio exactly ~1.0 = no systematic loss). It is now
fully concentrated in the 13/50 still on ManifoldPlus — that subset alone remains as bad as it
ever was (median ratio 0.0047, still highly significant collapse, p=0.021 even at n=13). The
alpha_wrap-guaranteed subset (37/50) shows no collapse at all — if anything the opposite,
consistent with §2's 13-specimen finding.

Full per-specimen data: `comparison_output/bench50_offsetfix_20260816/report/per_specimen.csv`,
`flagged_for_review.txt`.

## 6. Fabian's original (unpatched) pipeline vs post-fix alpha_wrap, all 50

`custom_processing/fabian_processing.py` is NOT the same code as the "original" baseline used in
§1-5 (`prepare_antscan_data_for_mesh_fitting.py`) - it's missing `filter_small_components`,
`bridge_nearby_islands` (thin-anatomy reconnection), and the Weld face-loss safety check that were
added on top of it later. Run via `diagnostics/fabian_processing_20260817/` (SLURM array, 50/50
succeeded, all meshes in `.../all_meshes/`), compared against the SAME post-fix alpha_wrap outputs
used in §5.

| Metric | §5 (patched "original" baseline) | §6 (Fabian's unpatched pipeline) |
|---|---|---|
| baseline watertight | 0/50 | 0/50 (same) |
| baseline fragmented (>5 components) | 48/50, median 27 | **7/50, median 2** |
| p5 thickness ratio, median (all 50) | 1.000 | 1.000 (same) |
| p5 thickness ratio, Wilcoxon p (all 50) | 0.052 | 0.388 (even less significant) |
| p5 thickness ratio, alpha_wrap-only subset | 2.04, p=0.002 (n=37) | 1.51, p=0.011 (n=37) |
| p5 thickness ratio, manifold_plus-only subset | 0.0047, p=0.021 (n=13) | 0.00069, p=0.0002 (n=13) |
| Chamfer distance, median | 44.95 | 30.19 (lower - Fabian's baseline itself is closer to alphawrap) |
| Vertex count, median diff (aw-orig) | +61,422, p=5.3e-9 | +35,459, p=1.1e-4 |

**Reading:** Fabian's unpatched pipeline is far LESS fragmented than the patched "original"
(median 2 vs 27 components) - `find_largest_component` throws away everything but the single
largest island, so it never accumulates the patched version's debris, but by the same mechanism
it can also silently discard real disconnected anatomy (a leg or antenna that ray-cast cleaning
happened to sever) rather than reconnecting it like `bridge_nearby_islands` does. Against this
baseline, alpha_wrap's advantage looks slightly smaller in the alpha_wrap-only subset (1.51 vs
2.04) but the qualitative story is unchanged: no batch-wide collapse post-fix, and the ManifoldPlus
fallback subset is still catastrophic (if anything worse relative to Fabian's baseline: 0.00069
vs 0.0047). The offset-fix conclusion in §5 holds regardless of which "original" baseline is used.

Full data: `comparison_output/fabian_vs_alphawrapfix_20260817/report/`.
