# Task 4 — does a better distance metric improve local taxonomic retrieval?

**Verdict: HOLD.** Neither Mahalanobis (fold-safe covariance) nor a lightweight Fisher-weighted
diagonal metric beats the Euclidean baseline by a margin distinguishable from noise, in-corpus or
cross-corpus. Mahalanobis is directionally *worse* in-corpus at every genus-count threshold
tested; the Fisher-weighted scheme is statistically indistinguishable from Euclidean everywhere.
Stay on Euclidean 1-NN in PCA space. This closes the distance-construction question raised by the
prior finding that genus information is local (1-NN works, UMAP/t-SNE/HDBSCAN do not,
`FINAL_REPORT.md` sec 3.1) — the answer is that the metric itself is not the bottleneck.

## 0. What was fixed, what was varied

Representation held constant throughout: log-shape-ratios (Mosimann, on the 6 `CORE_BLOCKS`
already selected on ground-truth reliability) → PCA(10) → 1-NN, exactly `absolute_scale/
analyse_source.py`'s existing protocol. Only the distance on top of the PCA(10) coordinates
varies. All three arms share the identical lot-blind leave-one-lot-out split, the identical
permutation seed, and the identical worker corpus (`morphometrics_source.csv`, 757 worker + 81
`ALL_ANTS_CLEAN` specimens — the already-computed deliverable table, no fit outputs re-run).

Full code and numbers: `task4_distance.py`, `out/results.json`, `out/run.log`.

## 1. Fixed baseline comparison (n=616, 82 genera, 34 lots, min_n≥3, npc=10)

| Method | Accuracy | Null | p | Lift | 95% CI |
|---|---|---|---|---|---|
| **A. Euclidean** (baseline) | 13.0% | 2.1±0.7% | <0.0001 | 6.13× | 7.1–15.5% |
| **B. Mahalanobis**, Σ re-estimated per fold from training specimens only | 12.3% | 2.1±0.7% | <0.0001 | 5.82× | 6.6–14.6% |
| **D. Fisher-weighted** diagonal, weights re-estimated per fold from training genus labels only | 13.3% | 2.1±0.7% | <0.0001 | 6.25× | 7.6–15.3% |

n=616/82 genera (vs. the 149/29 in `REPORT_ABSOLUTE_SCALE.md`) because this task doesn't need
the absolute-scale-calibratable subset — shape alone needs no calibration, so the full worker
corpus after the `min_n≥3` filter is usable. All three CIs overlap almost completely; none of the
three-way spread (12.3–13.3%) is distinguishable from sampling noise at this n.

**Mahalanobis is not an improvement here** — it loses 0.7pp of accuracy and 0.31× of lift versus
Euclidean, the *opposite* of the hoped-for direction, though within the overlapping CIs. Shrinkage
(α=0.15 toward the diagonal, `shrink_inv_cov`) was applied specifically to avoid a version of this
failure mode caused by covariance noise in the smaller lot-blind folds; the result held with
α swept informally at 0.05/0.15/0.3 (not tabulated — direction did not change).

**Fisher-weighted is a marginal, non-significant gain** (+0.3pp, +0.12× lift) — the CIs overlap
essentially entirely with Euclidean's. Reported honestly as a non-finding, not a discovery.

## 2. Accuracy by number of genera (min_n sweep — Euclidean vs. Mahalanobis)

| min_n | n | genera | Euclidean | Mahalanobis |
|---|---|---|---|---|
| 3 | 616 | 82 | 13.0% (6.13×) | 12.3% (5.82×) |
| 5 | 496 | 46 | 18.3% (5.77×) | 16.9% (5.23×) |
| 8 | 386 | 27 | 21.8% (4.36×) | 21.8% (4.44×) |
| 12 | 221 | 9 | 39.4% (3.19×) | 34.8% (2.85×) |

Mahalanobis is never ahead of Euclidean at any genus-count threshold — tied once (min_n=8,
27 genera), behind at the other three, and furthest behind at the *fewest*-genera / most-
per-genus-training-data setting (min_n=12: −4.6pp). That is the opposite of what a covariance
correction should buy: with more training examples per fold, Σ estimation should get *more*
reliable, not less useful. The more likely explanation is that at npc=10 the PCA coordinates are
already close to isotropic (each axis independently standardised, then whitened again by SVD), so
there is little anisotropy left for a learned Σ to correct — Mahalanobis is mostly adding
estimation variance without removing a real correlation structure. (Fisher-weighted was not
re-run across every threshold — the top-line result already showed no effect worth chasing across
the sweep.)

## 3. Reliability-weighted distance (item C) — not run as a decisive arm

Two independent reasons, either sufficient on its own:

1. **Task 2's own verdict is HOLD** (`diagnostics/reliability/REPORT.md`): no weighting scheme
   beat `equal` with evidence clearing this project's pre-registered bar at n=24 synthetic
   specimens, and the *one* statistically distinguishable cell (`mesosoma`/`continuous_part`,
   ΔR=−0.113) was a regression, not an improvement. This task's own instruction — "if Task 2
   establishes a defensible reliability measure, test..." — is explicit that C is conditional on
   Task 2 clearing that bar. It did not.
2. **Even setting (1) aside, it is not computable at this task's scale.** `continuous_global`/
   `continuous_part` weights are built from `measure.mesh_quality`/`part_quality`, which need
   per-specimen fitted-mesh npz outputs. Task 2 §5 already documents that the full-corpus fit
   outputs (`MORPH_W*` runs) do not exist on this cluster — only `diagnostics/moonshot/runs/
   D1_N50_SEED0` (50 specimens, 2 genera after the standard filter) does. That is the same
   blocker Task 2 hit, not a new one.

No exploratory pass is included for C, since a scheme with no defensible free-parameter choice
and no computable inputs at this scale would not add information — it would just be a second
place to accidentally overclaim.

## 4. Cross-corpus retrieval — the decisive test (n=46 clean specimens, 38 shared genera)

Worker genus centroids (Euclidean mean / Mahalanobis Σ / Fisher weights, all fit on the worker
corpus *only*) scored against `ALL_ANTS_CLEAN` specimens — different specimens, preparation,
scanner from the worker corpus, so this is the test the task instructions flag as the one that
actually bears on whether a distance change is *biologically* better, not just numerically better
on the same corpus's own folds.

| Method | Top-1 | Null | p | Δ vs. Euclidean | 95% CI on Δ |
|---|---|---|---|---|---|
| Euclidean | 8.7% | 2.3% | 0.0125 | — | — |
| Mahalanobis | 10.9% | 2.3% | <0.0001 | +2.2pp | **[−6.5, +10.9]pp** |
| Fisher-weighted | 8.7% | 2.0% | 0.0100 | +0.0pp | [−6.5, +4.3]pp |

Chance is 2.6% (1/38 genera). All three methods individually clear their own permutation null —
the cross-corpus signal itself replicates, consistent with the existing finding in
`analyse_source.py` sec. 4. But the **paired** comparison — same 46 test specimens, resampled
together so the comparison isn't confounded by which specimens happen to be easy — is what
answers the actual question here, and Mahalanobis's apparent +2.2pp edge has a 95% CI of
[−6.5pp, +10.9pp]: it comfortably contains zero. At n=46 specimens, a 1-specimen swing in raw hit
count is a ~2pp change in top-1, so this gap is well inside noise. **Per the pre-registered
constraint in the task instructions, this is not treated as evidence the Mahalanobis
representation is biologically better** — and per the same logic, item D's clean CI is even
tighter around zero, so the small in-corpus edge it showed in §1 does not carry over here either.

## 5. Conclusion

Every arm was tested lot-blind, on the identical split, against the identical permutation
protocol, at multiple genus-count thresholds and — for the two arms with a computable input —
cross-corpus. None of it moves the needle:

- **Mahalanobis**: consistently flat-to-negative in-corpus (4/4 thresholds not ahead), a
  directionally positive but statistically noise-level cross-corpus gap.
- **Fisher-weighted (lightweight metric learning)**: statistically indistinguishable from
  Euclidean everywhere it was tested.
- **Reliability weighting**: not testable at this task's scale, and not justified by Task 2's own
  HOLD verdict even where it would be.

This is consistent with, and extends, the standing local-not-global finding: the taxonomic signal
in this feature space is carried by which specific points are nearest, not by any exploitable
large-scale anisotropy or region-specific noise structure that a smarter metric could correct for.
The representation (log-shape-ratios → PCA) is the limiting factor, not the distance function on
top of it — consistent with `FINAL_REPORT.md`'s framing that the signal is local. Improving
genus retrieval further likely needs better or additional *features* (e.g. resolving the blocked
absolute-scale/full-corpus items in Tasks 1–2), not a better metric on the existing ones.

## Reproducing

```
conda activate pytorch3d
python diagnostics/distance_metric/task4_distance.py
```
Reads `diagnostics/absolute_scale/morphometrics_source.csv` (already on disk); writes
`out/results.json`. No fit outputs, GPU, or pytorch3d-dependent import required — `measure.py`'s
functions used here (`log_shape_ratios`, `parse_taxonomy`) are pure numpy/re.
