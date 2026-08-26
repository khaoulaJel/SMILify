# A2 — Dense Correspondence Label-Generator Validation: Results (2026-08-25)

## What was checked

B2 needs per-point ground-truth correspondence labels to supervise a network. The label rule
already exists and is validated in production use (`w_dense_gt`, finding #11): for a target point
`p`, its label is `argmin_v ||p - dense_gt_verts[v]||` — nearest true-mesh vertex in 3D space.
That rule was never checked at the point level before now; it was only validated indirectly, via
the *aggregate* seg_acc ceiling (0.974, not 1.0). This probe checks the rule directly: sample
20,000 points from a posed specimen with a custom area-weighted sampler that (unlike the public
`sample_points_from_meshes` API) also records which face each point truly came from, then asks
whether the nearest-vertex label actually belongs to that face.

## Result

| | value |
|---|---:|
| Labels consistent with the point's own sampled face | 89.99% |
| Mislabeled (nearest vertex is NOT one of the true face's 3 vertices) | 10.01% (n=2002/20000) |
| — of those, same anatomical segment anyway (soft failure) | 92.16% |
| — of those, different segment (real correspondence error) | 7.84% (n=157/20000 overall = **0.79%** of all points) |
| Mislabel distance (assigned vertex to nearest true-face vertex) | mean 0.020 (1.08% of mesh extent), median 0.013, p95 0.064 |

## What it means

The label rule is sound at the level finding #11's aggregate ceiling implied, and now confirmed
directly rather than inferred: ~10% of individual point labels miss their exact originating
triangle (an expected artifact of finite mesh resolution — no triangulated surface has zero
Euclidean/geodesic gap between adjacent faces), but the practically relevant failure mode — a
label that crosses into the WRONG anatomical segment (the thin-leg/tightly-packed-geometry risk
this probe was built to catch, see script docstring) — affects **well under 1% of points**. This
is a small, quantified, non-zero error floor, consistent with (and likely a contributor to)
finding #11's own residual 0.974-not-1.0 seg_acc ceiling rather than an unrelated new problem.

**Conclusion: the dense-label generator is fit for B2's purpose as-is.** No fix is needed before
using it to generate B2's training labels — the ~0.8% wrong-segment rate is a floor to be aware
of (it will show up as a small amount of label noise in training, and caps how close any network
trained on these labels could ever get to 100% even with a perfect architecture), not a defect
to patch. Recorded here per this investigation's discipline of never assuming a pipeline
component is correct without a direct check (`CLAUDE.md`: "Verify byte-equivalence on I/O swaps",
generalized here to "verify label-equivalence on a nearest-neighbor correspondence rule").

Plot: `fig_A2_dense_label_validation.png` (left: consistency breakdown bar chart; right:
mislabel-distance histogram for the 10.01% inconsistent points).
