# Open thread — the D1 stage degrades the coxa it was handed

Opened 2026-08-28 as a **placeholder**, from Control 2 of
`PREREGISTRATION_E1E2_frozen_pose_20260828.md`. Deliberately **not scoped yet**: E1/E2 (job
3264569) close the coxa-via-pose-freeze thread first, and this is not a variant of that question.

## The observation

Auditing the hierarchical stage's own output (`H2_joint.npz`) instead of the post-D1 fit:

| run | stage | leg_acc | **co** |
|---|---|---|---|
| `C13_uniform` (CSE correspondence) | H2_joint | 0.9649 | **0.2564** |
| `C13_uniform` (CSE correspondence) | after D1 | 0.9415 | **0.4115** |
| `C14p_gtinit_nocse` (GT-seeded pose) | H2_joint | 0.9413 | **0.4564** |
| `C14p_gtinit_nocse` (GT-seeded pose) | after D1 | 0.9468 | **0.4114** |

With correspondence, the hierarchical stage reaches **0.2564** — far better than any post-D1
number ever recorded — and D1 then degrades it to 0.4115, taking `leg_acc` down with it
(0.9649 → 0.9415). Against the coxa's measured floor of 0.0908 that is a loss of ~0.155 of a
~0.166 gain: **D1 gives back almost everything the hierarchical stage won at the coxa.**

## Why this is a different question, and a better-posed one

It does not ask "is the coxa reachable" — the hierarchical stage demonstrably reached it. It asks
**which specific D1 term moves the coxa away from a position already found**, and whether that term
can be reweighted or scoped to leave the coxa alone. That is a concrete, bounded search over terms
that run in D1 but not in the hierarchical stage (edge, laplacian, symmetry, offset, and the
free-form `deform_verts` pass, which is disabled in the hier recipe via `--deform_its 0`).

`deform_verts` is the first suspect precisely because the hier arms ran with it off.

## Not yet established

- Whether the effect is D1's loss terms, its free-form deformation, or its initialization handoff.
- Whether it generalizes past `C13_uniform` — the GT-seeded arm moves the *other* way (0.4564 →
  0.4114), so this is not yet a uniform "D1 harms the coxa" claim, and may be specific to arms
  where the hier stage had already placed it well.
- Anything about real scans. Both rows are synthetic, model-generated P48.

Do not cite the 0.2564 as a result until this thread is actually scoped and tested.
