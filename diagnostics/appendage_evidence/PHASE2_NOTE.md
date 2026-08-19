# Task 3, Phase 2 — candidate measurement implementations

`diagnostics/appendage_evidence/candidates.py`: C0 (baseline, unchanged — imported from
`measure.bone_table`/`bone_lengths`, filtered to the exact `leg_distal` columns via
`calib_features.block_of`), C1 (joint-chain sum: `chain_distal` = seg_ti+seg_ta, the direct
like-for-like span vs C0; `chain_full` = seg_co..seg_ta, a different quantity — whole-leg length
— kept separate, never compared to C0 as if it answered the same question), C2 (geodesic/mesh-edge
shortest path via `scipy.sparse.csgraph.dijkstra`, `geo_distal`/`geo_full` matching C1's spans).

`diagnostics/appendage_evidence/score.py`: one scoring path for all candidates, formula copied
verbatim from `calib_features.py`'s own R/SNR/bias loop (not reimplemented). `score_run(run,
corpus)` loads the fitted-vertex array and the ground-truth vertex array ONCE each, runs BOTH
through `candidates.measure_all` (the same function, same code, no branch on which one it's
given), and scores. This guarantees no candidate's number can come from a different definition of
"ground truth" or a different scoring rule than any other's.

## Endpoint selection for C2 (per your instruction to document this precisely)

Proximal/distal endpoint vertex indices are computed ONCE from the template rest-pose geometry
and frozen — reused unchanged for every specimen and for both fitted and ground-truth arrays
(candidates.py's module docstring has the exact rule and reasoning). Confirmed by construction:
since correspondence in this synthetic ceiling test is by vertex index, freezing two integers
from the template is "the same rule on both" by definition, not something that could drift.

## Implementation sanity checks (not the Phase 3 decision)

1. On the template mesh (all specimens = identical geometry), every candidate is finite and the
   geodesic length is consistently *slightly longer* than the corresponding straight chain length
   (e.g. `geo_distal` 0.232 vs `chain_distal` 0.229 for leg 1) — the physically-required
   inequality (geodesic ≥ straight path) holds for every leg/side, a basic correctness check a
   bug would very likely violate.
2. **Smoke run only, on the single existing clean/pose=0.25 condition** (`SYN_clean_w5` /
   `synth_clean` — no new fit needed), purely to catch implementation bugs before spending compute
   on Phase 3's full battery:

   | family | median R | median SNR | median &#124;bias&#124; |
   |---|---|---|---|
   | C0 baseline | 0.141 | 0.35 | 12.0% |
   | C1 chain_distal (like-for-like) | 0.110 | 0.35 | 8.6% |
   | C2 geo_distal (like-for-like) | 0.155 | 0.42 | 24.1% |
   | C1 chain_full (different quantity) | 0.595 | 0.92 | 2.9% |
   | C2 geo_full (different quantity) | 0.527 | 0.89 | 11.7% |

   On the like-for-like distal span, neither C1 nor C2 clears C0 by a margin that means anything
   at n=12 on one condition — consistent with Phase 1's finding that this is a correspondence
   problem the length definition alone doesn't fix. **This is not the decision** — it is one
   condition out of the three the plan requires (clean / pose-varied — already covered by this
   same condition — / damage), and Phase 3 still needs the pose_scale=0 rescoring and the damage
   corpus before any verdict. Reported here only so a candidates.py bug would have been caught
   now rather than after a multi-hour Phase 3 fitting run.

## Not yet done (Phase 3)

- Rescore all candidates on `SYN_clean_pose0_w5` / `synth_clean_pose0` (already fit, from Phase
  1b) — this is a legitimate additional data point, not a decision on its own either.
- Build the damage/missing-geometry corpus (distal-vertex dropout before fitting) and fit it.
- Full comparison table + verdict per candidate, and only then the Phase 4 cross-corpus gate for
  anything that clears Phase 3.
