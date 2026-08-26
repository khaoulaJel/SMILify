# Penetration handling in fitter_3d — consolidated findings and standing recommendation

Start here. This is the one document that says plainly what was tried, what shipped, why the
underlying problem is genuinely hard, and what to do next. Everything else in
`diagnostics/khaoula_review/` and `scripts/penetration_joint_study/FINDINGS.md` is the evidence
trail behind this — read them for the detail and the numbers, not to find out what to do.

## The problem in one paragraph

SMIL-fitted meshes can look accurate from the outside (good chamfer, good silhouette) while a
leg passes straight through the gaster on the inside. `fitter_3d/penetration_loss.py` adds a
differentiable penalty for this. Turning it on reliably makes existing contact **shallower**
(depth improves) but does **not** reliably make it **less frequent** (count doesn't improve, and
sometimes gets worse) — the count/severity split that started this whole investigation. Two
independent, confirmed root causes were found for that split, and this project tried fixing
each of them, plus tried resolving penetration by other means entirely. None of it produces an
unconditional win. That's not a failure to find the right fix — see "Why this is genuinely hard"
below.

## What shipped today (adopted, not experimental)

- **`penetration_soft_num_penetrating` is now the standard metric for comparing any two
  penetration-related arms or seeds.** Not opt-in, not gated, computed unconditionally
  alongside the existing hard count in every eval CSV
  (`fitter_3d/penetration_loss.py`/`trainer.py`). The hard count (`num_penetrating`) has
  substantial seed-to-seed noise (mean CV ~40% across a 10-specimen x 3-seed panel) because
  vertices near the classification boundary flip in/out of the count with tiny position
  changes between otherwise-identical runs. The soft count is a sigmoid-relaxed version of the
  same threshold (measured ~25-45% relative CV reduction, same panel) — use it for "did this
  arm actually help", keep the hard count for "how many, exactly, right now."
- **`symmetric_chamfer_sampling` is available now as an opt-in, gated `Stage` flag** (default
  `False`, zero effect on any existing config). It reliably improves surface fit (mean F-score
  +0.028 to +0.058, positive in every one of 3 seeds, 10 specimens). It does **not** reliably
  reduce penetration count — under the standard (soft) metric the average effect is a real,
  modest improvement (mean -3.2%), but one of three seeds still shows a worse result even under
  the low-noise metric, so this is not an unconditional win on that axis. Recommended as a
  standalone quality improvement, independent of whatever else is or isn't shipped for
  penetration count specifically.

## What was tried, and what happened

| approach | targets | result |
|---|---|---|
| Gentle proximity loss (existing `w_penetration`), pair-scoped + `w_offset` + `w_scale`/`w_trans` barriers (TASK1-5) | resolve penetration directly via the existing nearest-centroid matching test | Best config found (-16.9% aggregate count) still fails the F-score bar on every arm tried. The count win on the targeted pair is real but comes with collateral elsewhere. |
| GWN-based matching primitive (TASK6) | replace the nearest-centroid test (confirmed less accurate specifically where count fails to improve, FINDINGS.md step 6) with a true winding-number signal | Not a clean fix. Weaker on the pair it trains than the existing loss (2 of 3 specimens), and still generates real collateral on pairs it never trains — pure geometric coupling from deforming the mesh, not misattributed gradient credit. Gentler in both directions, not a better primitive. |
| torch-mesh-isect, original fork (`EthanFifle/torch-mesh-intersection`) | SMPLify-X's field-standard BVH + conical distance field | Compiles clean, kernel **never detects any collision** — confirmed via a maximally unambiguous test (two exactly coincident triangles, still not flagged). Not usable, full stop. |
| torch-mesh-isect, maintained fork (`wonjongg/torch-mesh-isect`, SIGGRAPH 2025's own) | same | Detection genuinely works in isolation (real intersections, multiple simultaneous pairs, all correct). Wired in as an actual `w_penetration_bvh` training signal (TASK8) — **crashes with a CUDA illegal memory access**. Originally read as training-graph-specific; TASK9's independent reproduction (see below) shows it's actually triggered by *repeated invocation in any loop*, caller-agnostic. Not usable as-is. |
| Symmetric chamfer sampling (TASK7) | the OTHER confirmed root cause: chamfer vertex-density asymmetry (legs 2.36x denser than gaster in raw template vertices) | Shipped (see above) — real F-score win, no reliable count win. |
| Soft-count metric | the seed-instability itself | Shipped (see above). |
| Post-hoc mesh repair (TASK9, `wonjongg/instant-mesh-intersection-repair`, the actual SIGGRAPH 2025 method, not just its detection primitive) | sidestep training-time instability entirely by repairing an already-converged mesh, no gradient competition, deterministic by construction | All infrastructure blockers resolved (`cholespy` segfault fixed with a solver swap; the CUDA BVH's repeated-invocation fault, same one TASK8 found, bypassed with a validated CPU detector; determinism confirmed bit-identical throughout). Tried two ways, both worse than every training-time arm in this investigation. **Whole-mesh, unfiltered: mean F-score -0.152.** Hypothesis: it repairs legitimate part-to-part contact right alongside genuine problems, so refined to **part-pair-aware filtering** (same non-adjacent-pairs convention as `penetration_loss.py`) — count then improved dramatically on all 3 specimens (as predicted), but **mean F-score cost nearly doubled to -0.326**, with one specimen collapsing outright (edge distortion 0.32, F-score 0.885→0.222). Concentrating the same iteration budget onto fewer, deeper pairs made the push more violent, not gentler. Closed on outcome grounds, tried both the naive and the mechanistically-motivated refined version — the refinement made it worse, not better. |

## Why this is genuinely hard, not just under-solved

Three separate, confirmed mechanisms, each independently evidenced, not guessed:

1. **Near-tie matching instability.** Any hard threshold on a continuous geometric quantity
   (proximity-based inside/outside, GWN's `w=0.5` boundary) has vertices sitting right at that
   boundary, and small position changes between training runs flip their classification. Direct
   per-vertex evidence (TASK7 Part 4): flipped vertices sit measurably closer to the boundary
   than stable ones in 7/10 specimens, sometimes 6-7x closer. This affects the **count metric
   itself**, independent of which loss produced the fit — confirmed on two unrelated mechanisms
   (gentle-proximity, per FINDINGS.md's own reseeded rerun, and symmetric-chamfer, TASK7). It is
   not fixed by switching matching primitives, because the instability is in the act of
   thresholding, not in which primitive gets thresholded.
2. **Rough gradients from discrete geometric primitives.** GWN's gradient near its own
   inside/outside boundary is known-rough in the literature (motivating smoothed variants like
   GauWN, SIGGRAPH Asia 2024). This is consistent with what TASK6 measured directly: GWN is
   gentler in both directions than the existing proximity loss — smaller wins where it should
   win, smaller-but-still-real collateral where it shouldn't touch anything — rather than a
   sharper, better-conditioned signal.
3. **A repeated-invocation fault in the one primitive that's actually well-conditioned.**
   BVH + conical distance field is the field's own answer to problem #2 above — exact detection,
   a purpose-built smooth push. It's the one candidate that isn't undermined by a
   gradient-quality argument. It fails anyway, for an unrelated, lower-level reason: the
   `bvh_cuda` kernel crashes when called repeatedly inside any iterative loop — confirmed
   identically whether the caller is this project's own multi-loss training step (TASK8) or a
   much simpler, single-energy repair loop with no training graph at all (TASK9) — caller-
   agnostic, not a training-graph-composition issue as first suspected. Single/isolated calls
   are reliable every time. This is an implementation bug in a third-party, effectively-
   unmaintained-for-this-context kernel, not a conceptual problem with the method — genuinely
   different in kind from mechanisms #1 and #2, and the only one of the three that a future
   attempt (kernel-level debugging, or a from-scratch reimplementation of the same
   conical-field idea) could plausibly resolve without hitting the same wall again.

There's a fourth pattern worth naming separately, because it isn't about the fitting problem's
own difficulty the way #1-3 are — it's about the tooling: **every third-party GPU-accelerated
geometry-processing dependency this investigation tried to adopt hit some version of a
prebuilt-CUDA-extension incompatibility with this project's environment** (torch 2.5.1+cu124,
Ampere/compute_86) — the original `torch-mesh-isect` fork (silently inert), `cholespy`
(segfaults on any CUDA input), and the maintained `torch-mesh-isect` fork's `bvh_cuda` kernel
(crashes under repeated invocation, confirmed caller-agnostic across TASK8 and TASK9's
independent reproductions). All three were root-caused to a specific, named cause, not
hand-waved, and all three were either fixed or fully worked around (a working fork, a one-line
solver swap, a validated CPU detection swap) — every infrastructure blocker this investigation
hit was eventually resolved. **None of that ended up mattering for the outcome.** Once post-hoc
repair could actually run end to end, its result (mean F-score -0.152) was worse than any arm
blocked by infrastructure ever got the chance to be. Worth remembering as its own lesson:
"the tool doesn't work" and "the tool works and the method doesn't help" are different findings,
and this investigation reached the second one, not the first.

## Standing recommendation

- **Default stays D1 / no penetration loss** (`w_penetration=0`). No arm across any approach
  tried — existing proximity loss variants, GWN, BVH — clears the simultaneous bar (aggregate
  count improves AND mean F-score doesn't regress) reliably enough to become the default.
- **If penetration reduction is specifically required for a given use case**, the TASK5 config
  (pair-scoped gentle proximity + `w_offset` + `w_scale`/`w_trans`) is the least-bad available
  option — ship it as a config choice, gated behind a required per-specimen check, never
  silently defaulted on. This was already the standing recommendation before this round of
  work; it still is, now on more evidence.
- **Use `penetration_soft_num_penetrating`, not `penetration_num_penetrating`, for any future
  arm/seed comparison in this area.** This is the one unconditional process change from this
  round of work.
- **`symmetric_chamfer_sampling=True` is recommended now**, independent of the penetration
  question, for anyone who wants the surface-fit improvement it reliably delivers.
- **Not recommended, not worth another attempt without new evidence or a changed environment**:
  GWN and BVH as penetration-loss matching primitives, for the two separate, specific reasons in
  mechanisms #2 and #3 above — not "nothing worked," but concretely different reasons why, each
  pointing at a different kind of future fix if this gets picked up again (gradient smoothing
  would need a GauWN-style primitive, untried; the CUDA fault needs kernel-level work, untried).
- **Post-hoc repair (TASK9) is closed, on outcome grounds — the strongest close of any arm
  tried, because the one mechanistically-motivated refinement was tried too, not left as a
  hypothesis.** Every infrastructure blocker got fixed or bypassed (`cholespy`'s segfault, the
  `bvh_cuda` repeated-invocation fault, via a validated CPU detection swap), and the actual
  repair ran to completion, deterministically (confirmed bit-identical), in two configurations:
  whole-mesh unfiltered (mean F-score -0.152) and part-pair-aware (same `non_adjacent_pairs`
  filtering `penetration_loss.py` already uses — count improved dramatically on all 3
  specimens as hypothesized, but mean F-score cost nearly doubled to **-0.326**, one specimen
  collapsing outright). The structural "no training instability" argument was correct and is
  directly demonstrated both times; concentrating the same iteration budget onto fewer, more
  severe pairs made the push more violent, not gentler. Not recommended in either configuration
  without a fundamentally different constraint/optimizer design (e.g. a per-iteration
  displacement cap) — a new method, not a variant of this one.

## Where the detail lives

`scripts/penetration_joint_study/FINDINGS.md` (the original root-cause investigation, steps
1-6) -> `diagnostics/khaoula_review/DELIVERABLE_penetration_fix_TASK1-5.md` (TASK1-5, the
existing-loss tuning arms, plus the torch-mesh-isect compat findings and corrections) ->
`TASK6_gwn_matching_primitive_causal_test.md`, `TASK7_symmetric_chamfer_sampling.md`,
`TASK8_bvh_penetration_attempt.md`, `TASK9_posthoc_repair_attempt.md` (the four causal/
training-signal/repair attempts this document summarizes). Each TASK doc's own retractions and
corrections are left in place and marked prominently where they occur — read the doc, not just
this summary, before relying on any specific number.
