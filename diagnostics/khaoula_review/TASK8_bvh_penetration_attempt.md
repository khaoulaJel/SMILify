# TASK 8 — BVH+conic-field penetration loss: real attempt, real crash, not shelved on faulty grounds anymore

Attempts "the original spec's TASK 4" (SMPLify-X's field-standard approach) as an actual training
signal for the first time, after confirming the BVH detection primitive itself works on a
different fork than the one tested first (`wonjongg/torch-mesh-isect`, see
`DELIVERABLE_penetration_fix_TASK1-5.md`'s 2026-08-24 update). This is not another compat probe —
it's the real unblock attempt the confirmed-working primitive made possible.

## What was built

`fitter_3d/bvh_penetration_loss.py` — reuses (not reimplements) `mesh_intersection.bvh_search_tree.BVH`
for detection and `mesh_intersection.loss.DistanceFieldPenetrationLoss` for the differentiable
conical push, both from the compiled extension. Filters BVH's raw face-pair collisions down to
whichever (part_a, part_b) pairs are requested — same purpose as `penetration_train_pairs`,
reimplemented because BVH's output shape (face pairs) doesn't match the existing per-vertex
query/surface split. Wired into `fitter_3d/trainer.py` as `w_penetration_bvh`/`bvh_penetration_pairs`,
lazily imported so no existing config is affected by `mesh_intersection` not being on the default
path. `fitter_3d/bvh_weight_calibration_probe.py` — same discipline as GWN's calibration probe,
found `w_penetration_bvh=0.0005` lands single-digit % of total loss (stage2 mean 1.3%/max 1.9%,
stage3 mean 2.3%/max 4.7%), comfortably below the 70% blowup GWN's first uncalibrated attempt hit.

## What happened when it actually ran

Two small-scale smoke tests passed cleanly: a 2-specimen, 15-iteration run with the BVH loss
active trained without error (loss decreased 0.046->0.016, `penetration_bvh` component genuinely
nonzero and moving, `compute_eval_metrics` produced sane hard/soft counts). The calibration probe
(3 specimens, 20-iteration snapshots, 8 candidate configs) also ran clean throughout.

**The real 10-specimen, seeded, matching-primitive-swap experiment (`run_bvh_penetration_experiment_seeded_scale.py`,
same specimens/seeds as TASK7) crashed on the very first `Stage_2` iteration**:
`Cuda failure .../bvh_cuda_op.cu:990: 'an illegal memory access was encountered'`. GPU state was
verified healthy afterward (a fresh CUDA context/matmul worked immediately) — the crash was
contained to that process, not a lasting corruption.

Isolated the trigger as precisely as time allowed before stopping, per this project's own
"don't debug someone else's kernel open-endedly" discipline (already stated in the DELIVERABLE
addendum this task follows up):

- The raw `BVH.forward()` detection call, run standalone on the exact same real 10-specimen
  warmed-up state (post Stage_0/Stage_1), works fine — 7554 real collisions detected, no crash,
  across `max_collisions` 8/16/32/64.
- `bvh_penetration_loss_batched` (detection -> filter -> `DistanceFieldPenetrationLoss` forward
  -> `.backward()`) called MANUALLY and IN ISOLATION on that same state, at iteration=0 (ramp
  factor exactly 0), also works fine — no crash, gradients flow.
- The SAME function, called through the real `Stage.step(0)` -- i.e. combined with chamfer, edge,
  normal, laplacian, w_offset, w_scale, w_trans all active simultaneously, in the same forward/
  backward graph as the actual experiment used -- **crashes immediately, on the very first call,
  scale identical to the isolated test that just passed.**

Reading: this is not a scale problem (10 specimens standalone was fine), not a first-iteration
ramp-value problem (isolated iteration=0 was fine), and not a buffer-size problem (varying
`max_collisions` up to 8x didn't matter for detection). Whatever's wrong is specific to something
about how this kernel's CUDA state interacts with the REST of a real training step's combined
loss/backward graph -- outside what a compat probe or a weight-calibration probe is built to
catch, and outside what this task is scoped to root-cause. CUDA's asynchronous kernel launches
mean the reported crash *location* (inside `bvh_cuda_op.cu`) is not necessarily where the fault
actually originates -- it's just wherever the next synchronization point caught it.

## Correction to the 2026-08-24 DELIVERABLE update

That update said the BVH detection primitive is "confirmed working on this environment" and that
wiring it into an actual loss "is a further step, not done here" — read as cautiously optimistic
about that further step. It should not be. **The detection primitive works in isolation; it does
not currently work as part of a real combined training step**, and that's a materially different,
more serious finding than a packaging or calibration issue. Flagging this correction here with
the same weight as the original claim, not buried, per this project's own standing discipline
after TASK7's Strumigenys retraction.

## Verdict

Still not usable as this project's penetration training signal, and for a different, more
fundamental reason than "believed non-functional" (the belief that motivated shelving it was
wrong for detection, right in effect for actual training use — just for a different underlying
cause). Root-causing the crash is real, scoped work — likely needs stepping through the CUDA
kernel directly (`cuda-gdb`/`compute-sanitizer`) or a minimal standalone repro isolating exactly
which OTHER loss term's presence triggers it, neither attempted here. Standing recommendation
(gated option, not default, `w_penetration` stays the shipped primitive) is unchanged and now
rests on stronger evidence than before: two matching-primitive alternatives attempted as real
training signals (GWN in TASK6, BVH here), neither a clean replacement -- GWN on outcome grounds,
BVH on basic stability grounds.
