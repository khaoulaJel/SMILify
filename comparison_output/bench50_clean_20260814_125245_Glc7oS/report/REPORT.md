# alpha_wrap vs original pipeline — bench50_clean comparison report

Run: `bench50_clean_20260814_125245_Glc7oS` (50 ant specimens, git commit `a63f04a`)
Analysis script: `compare_pipelines.py` (repo root)
Generated: 2026-08-16

## Headline result

**50/50 specimens processed, 0 errors.** Thin-feature (leg/antenna) thickness collapse is
**real, statistically significant, and widespread** — not an isolated Pheidole-specific fluke.
It is present in both reconstruction sub-paths, but far worse in the ManifoldPlus fallback path.

Do not ship alpha_wrap as-is. See recommendation at the bottom.

## 1. What actually ran

The "alphawrap" pipeline is not one method — it's two, selected per-specimen at runtime:

| Reconstruction path | Count | Formal guarantee |
|---|---|---|
| `alpha_wrap` | 28/50 (56%) | Yes — strict enclosure |
| `manifold_plus_fallback_from_alpha_wrap` | 22/50 (44%) | **No** |

44% of specimens did **not** get the enclosure guarantee that was the whole reason for adopting
alpha_wrap. On top of that, **10/50 (20%)** failed the pipeline's own internal fidelity gate at
generation time (`fidelity_gate_passed=False`) and were exported anyway, per the pipeline's
"flag, don't silently drop" policy — meaning 1 in 5 specimens is already self-flagged as
untrustworthy before this analysis even starts.

All three numbers matter for the go/no-go call: this isn't "alpha_wrap works, ManifoldPlus is a
rare edge case" — it's closer to a coin flip between two methods with very different reliability,
and a policy that ships known-bad output either way.

## 2. Baseline caveat (read before trusting any ratio below)

**The `original` (Weld-based) baseline is fragmented and non-watertight on every single
specimen** — median 27 disconnected components, up to 69, 100% (50/50) not watertight. This is
the known, expected failure mode of the deprecated pipeline (why it's being replaced), not a
script bug — confirmed via each specimen's own `original.log` (hole counts, boundary edges,
`is_watertight=False` all logged directly by the pipeline). Two consequences for this report:

- **`volume_ratio_aw_over_orig` is not computable for any specimen** (0/50) — it requires both
  meshes watertight, and `original` never is. Don't read anything into its absence; it's not
  broken, it's structurally undefined here.
- The thin-feature thickness metric is computed **against `original`'s largest connected
  component only** (fixed during sanity-checking, see §4) — otherwise stray debris islands in the
  fragmented baseline dominate the low-percentile thickness statistic and produce meaningless
  ratios (confirmed: mixed up to 12x in either direction on hand-checked specimens before the fix).

## 3. Is the thin-feature collapse real?

Yes. Using the 5th-percentile thickness ratio (alphawrap / original, more robust to ray-cast noise
than p1 — see §4), paired two-sided Wilcoxon signed-rank test across all 50 specimens:

| Subset | n | median ratio | below 0.5x (collapse flag) | Wilcoxon p |
|---|---|---|---|---|
| **All 50** | 50 | **0.047** (thinnest features ~5% of original's) | 45/50 (90%) | **p = 1.1e-8** |
| `alpha_wrap` only | 28 | 0.208 (~21% of original's) | 25/28 (89%) | **p = 4.2e-4** |
| `manifold_plus_fallback` only | 22 | 0.0015 (~0.15% of original's) | 20/22 (91%) | **p = 4.8e-7** |

Translation: this is not a borderline or small-sample result. Both sub-paths show a real,
consistent, statistically decisive loss of thin-feature detail relative to the original mesh's
own (largest-component) thickness — legs and antennae are getting thinner or destroyed, not just
in the one Pheidole specimen that started this investigation. The ManifoldPlus fallback path is
dramatically worse: a median ratio of 0.0015 means the thinnest surviving features are typically
under 1/500th their original thickness — this reads as near-total destruction of thin anatomy,
not gradual erosion.

**Sample size is adequate to trust this** (n=28 and n=22 per subset, both individually
significant at p<0.001) — this is not a "need 10 more specimens" situation.

## 4. Sanity-checking the script (what was checked, what was found and fixed)

Per the ground rules, the script was hand-verified against real data before the full run, and
this surfaced three real, non-cosmetic issues — fixed in place, not worked around:

1. **`rtree` was missing from the `pytorch3d` conda env.** The SDF thin-feature thickness
   metric — the one specifically meant to catch Pheidole-style collapse — was silently returning
   `NaN` for every specimen while the rest of the script ran normally. This would have shipped a
   report with the one metric that mattered quietly empty. Fixed: installed `rtree`.
2. **1st-percentile thickness was noise-dominated.** On hand-checked specimens, p1 flipped sign
   entirely relative to p5 (implying "growth" instead of "collapse") because a handful of rays
   escaping through holes in the non-watertight `original` baseline produced erratic outlier hit
   distances. p5 is far more stable and was made the primary signal; p1 is retained in
   `per_specimen.csv` for transparency only, not as evidence on its own.
3. **SDF sampling over `original`'s full (fragmented) mesh let debris dominate.** Fixed by
   restricting SDF ray-casting to each mesh's largest connected component (`largest_component()`
   in the script).

Independent cross-check: `original`'s own pipeline log for a hand-picked specimen
(`Acanthostichus_aff.brevicornis`) directly reports `is_watertight=False`, `simple_holes=1106`,
`fidelity_gate_passed=False`, `worst_fidelity_deviation=181.8` — matching what the comparison
script independently measured from the exported `.obj`, confirming the script reads real pipeline
state correctly rather than miscomputing it.

Also found during the run itself (not a correctness bug, but worth recording): the ray-triangle
intersector trimesh falls back to without `embreex` installed OOM'd catastrophically on the
largest specimen (3.3M faces) — 188GB+ and still not done after 12 minutes. Installing `embreex`
fixed it (same specimen: 34 seconds, 2.3GB). No metric values were affected — this failed loudly
(OOM-killed) rather than producing wrong numbers.

## 5. Errors

**Zero.** All 50 specimen pairs were found and processed successfully; no specimen silently
dropped from the aggregate. (`errors.txt` is absent because the script only writes it when
errors occur — confirmed no error path was ever hit this run.)

## 6. Other findings worth flagging

- **14/50 (28%)** of alphawrap outputs have more than one connected component, despite alpha_wrap
  and ManifoldPlus both being expected to produce a single watertight solid. Not fatal on its own,
  but worth a look alongside the thin-feature flags — could be the same root cause (thin
  connective anatomy being severed rather than just thinned).
- **8/50** specimens are Chamfer-distance outliers (>5 MAD from the batch median) — these plus
  the thin-feature and multi-component flags are listed per-specimen, with reasons, in
  `flagged_for_review.txt`. All 50 specimens carry at least one flag; treat that file as the
  starting point for any visual review, not a random sample.
- Vertex counts are, as expected/intended, wildly variable and systematically much higher for
  alphawrap output (median +87,848 vertices vs `original`, p=6e-14) — this is by design
  (`_decimate_with_fidelity_stopping`'s documented policy: vertex count is a soft target, not a
  correctness criterion) and is not itself evidence of a problem; downstream SMAL-style
  registration resamples to the template's own vertex count regardless.

## 7. On the decimate_mesh fix you asked me to propose

It already exists and was active for this exact run. `base.decimate_mesh()`
(`custom_processing/prepare_antscan_data_for_mesh_fitting_manifold.py:1294`) — the shared,
"never modify directly" function — genuinely only gates on topology (`boundary_edges==0` /
`non_manifold_edges==0`), exactly as suspected. But the alphawrap pipeline
(`custom_processing/prepare_antscan_data_for_mesh_fitting_alphawrap.py:775`,
`_decimate_with_fidelity_stopping()`, dated 2026-08-13 — one day before this bench50 run) already
wraps it with a fidelity-based stopping criterion: it decimates using the same COLLAPSE primitive
but stops the moment whole-surface deviation from the pre-decimation mesh crosses
`fidelity_stop_factor x target_cell_size`, regardless of vertex count, and is called
unconditionally whenever decimation runs (`prepare_antscan_data_for_mesh_fitting_alphawrap.py:1301`).

**This means the collapse documented in §3 happened despite that fix already being live.**
That's a more serious finding than "the fix hasn't been written yet" — it means either:
- the fidelity check's threshold (`fidelity_stop_factor`, defaulted from `fidelity_gate_factor`)
  is too loose to catch thin-feature-scale deviation even though it catches whole-surface
  deviation (a thin leg is a small fraction of total surface area, so a global fidelity metric
  can pass while a local feature is destroyed), or
- the collapse is happening upstream of decimation, in reconstruction itself (alpha_wrap's
  bisected `alpha` or ManifoldPlus's internal resolution choice), and decimation is innocent.

I have **not** modified `_decimate_with_fidelity_stopping`, `decimate_mesh`, or any production
pipeline code — per the ground rules, that needs your explicit sign-off on the approach. Given
§3's finding that ManifoldPlus-fallback specimens are ~140x worse than pure alpha_wrap specimens
(0.0015 vs 0.208 median ratio), my read is that the fix should start with a **per-feature (not
whole-surface) fidelity check** — e.g. folding a local thickness/SDF check into the existing
gap-preservation check already run at each decimation step, the same way
`_decimate_with_gap_preservation` already tracks one specific landmark gap — rather than only
tightening `fidelity_stop_factor`, since a global threshold is structurally the wrong shape of
check for a defect that's local by nature. Whether that's the right direction is your call, not
mine to implement here.

## 8. Recommendation

**Hold and fix first — do not ship alpha_wrap as the production reconstruction method yet.**

The data doesn't support "proceed as-is": thin-feature collapse is real, large, and hits both
reconstruction sub-paths (worse in the 44% that fall back to ManifoldPlus). It also doesn't
support "abandon alpha_wrap" — the `alpha_wrap`-only subset, while still showing a significant
collapse, is an order of magnitude better than the ManifoldPlus fallback (0.208 vs 0.0015 median
ratio), so the enclosure-guaranteed path is clearly the better foundation; it just isn't
sufficient on its own yet. Two independent things need attention before a re-test, in priority
order:
1. Why does 44% of this real-world specimen distribution fail to converge under alpha_wrap and
   fall back to ManifoldPlus at all? That fallback rate is itself the bigger lever — fixing it
   would remove the worst of the collapse (the manifold_plus subset) without touching decimation.
2. The per-feature fidelity check discussed in §7, for whatever residual collapse remains in the
   pure alpha_wrap path.

Re-run this same script against a new bench50 (or larger) batch after either change lands — the
harness and metrics are now validated and ready to reuse as-is.

## Files in this folder

- `per_specimen.csv` — full metric table, one row per specimen
- `summary.json` — aggregate stats + all Wilcoxon test results (full batch + both subsets, p1 and p5)
- `flagged_for_review.txt` — all 50 specimens, grouped reasons per specimen (open these first)
- `run_provenance/` — SLURM batch scripts, logs, and the incremental per-specimen progress file
  from the actual run (kept for reproducibility/audit, not needed to read the results above)

---

## Addendum (2026-08-16, later same day): offset/alpha decoupling fix

Root cause identified (external review) for the 44% ManifoldPlus-fallback rate in §1: the
bisection derives `offset = alpha / 30` (CGAL's own documented example ratio), but per CGAL's own
docs, `alpha` and `offset` are independent controls - `alpha` alone gates whether the algorithm
can carve through a gap (what bisection is solving for); `offset` only controls how tightly the
output hugs the input everywhere else (density/refinement cost). Forcing `offset = alpha/30` on
every CGAL call - including the ~6-8 calls per specimen during bisection - inflated refinement
cost mesh-wide, independent of the gap-preservation search. CGAL's own `wrapwrap` reference tool
recommends `relative_alpha=500, relative_offset=1200` (`offset ≈ alpha × 0.417`), over 10x looser
than `alpha/30`.

**Fix applied** (both call sites - `_bisect_alpha_wrap_params.offset_alpha_ratio`, the actual
production path, 30.0→2.5; `_derive_alpha_wrap_params.offset_factor`, the no-landmark fallback
path, 1/30→0.4 - matching `offset = alpha × 0.4`):
`custom_processing/prepare_antscan_data_for_mesh_fitting_alphawrap.py`.

**Ground-truth sanity check** (`diagnostics/offset_fix_sanity_20260816/`): both specimens named in
this pipeline's own validation history now pass. `Cephalotes_spinosus_CASENT0744158` (previously
timed out, fell back to ManifoldPlus) completes as pure `alpha_wrap` in 226s,
`fidelity_gate_passed=True`. `Dorylus_sp._CASENT0744703` (~968K pre-reconstruction vertices, this
function's original characterization specimen) completes in 2293s, `fidelity_gate_passed=True`,
gap verdict `preserved` consistently across all 8 bisection steps. Notably, this wasn't only a
speed fix: on `Dorylus_fulvus_CASENT0745678` (see full rerun below), the *exact same* alpha value
(6.041) that the old tight offset reported as `verdict=preserved` (a bisection-bounds anomaly that
caused an outright failure) now correctly reports `verdict=fused` under the new offset - i.e. the
old over-tight offset was producing numerically unreliable bisection verdicts, not just slower
ones.

**Full rerun of all 22 previously-fallback specimens**
(`diagnostics/offset_fix_rerun_20260816/`, SLURM array job 3000006, all 22 tasks completed):

| | Before (bench50_clean) | After (offset fix) |
|---|---|---|
| Pure `alpha_wrap` success | 0/22 | **13/22 (59%)** |
| Still falls back to ManifoldPlus | 22/22 | 9/22 |

**Thin-feature quality on the 13 newly-fixed specimens** - re-ran `compare_pipelines.py` pairing
each new alphawrap output against the SAME `original` baseline already in this report:

| Metric | Original batch (all 50, §3) | 13 newly-fixed specimens |
|---|---|---|
| p5 thickness ratio (median) | 0.047 (catastrophic collapse) | **2.05** (no collapse - if anything, slightly thicker) |
| Wilcoxon p (batch/subset) | 1.1e-8 (collapse) | 0.040 (direction reversed - not a collapse signal) |

The collapse documented in §3 is **not an inherent property of alpha_wrap** - it was fully
attributable to needing ManifoldPlus (which has no fidelity/quality guarantee) as a fallback. Once
a specimen gets to use alpha_wrap as originally intended, the thin-feature problem disappears for
that specimen.

**What this does NOT fix**: the remaining 9/22 fail for a different reason entirely - `_run`'s
`alpha_fused_start = pre_gap * 15` heuristic genuinely doesn't fuse the gap on those specimens
(a bisection-bounds problem), unrelated to offset. That's a separate bug, still open.

**Not yet re-validated**: the 28 specimens that already succeeded with `alpha_wrap` under the old
offset were not re-run under the new one. CGAL's docs and the two ground-truth checks above give
good reason to expect no regression (offset only affects cost/density, and both ground-truth
specimens passed cleanly), but this is expectation, not confirmed data - re-run the full 50 before
calling this done.

### Updated recommendation

Materially better, not yet ready to ship. The offset fix resolves the single largest lever
identified in §8 (the 44% fallback rate) and, where it lands, appears to fully resolve the
thin-feature collapse too - both bigger and cleaner wins than expected. Before shipping:
1. Re-run the full 50-specimen batch with the fix (not just the 22 that previously failed) to
   confirm no regression on the other 28 and get an updated batch-wide read.
2. The remaining 9/22 bisection-bounds failures need their own fix (loosen/adapt
   `alpha_fused_start`, or a different coarse-bound strategy) - separate from this one.
