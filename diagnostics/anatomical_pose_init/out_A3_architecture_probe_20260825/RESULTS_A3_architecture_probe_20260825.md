# A3 — Architecture Probe (SMILPointNet2 sa1/sa2): Results (2026-08-25)

## Provenance correction (stated up front)

The handoff assumed an "already computed" `SMILPointNet2` checkpoint existed among this repo's
`diagnostics/anatomical_pose_init/*/best_model.pt` files. Checked directly, none of them are:
every one of those belongs to a different, simpler network
(`train_leg_pose_regressor.py::LegPoseRegressor`, a single global-max-pool PointNet with no
sa1/sa2 hierarchy — state-dict keys confirm this). The actual, already-trained `SMILPointNet2`
checkpoint is `checkpoints/checkpoint_epoch_10.pth` (`sa1.conv_blocks.*` keys confirm
`PointNetSetAbstractionMsg`; n_pose=54, n_betas=13, output_size=511 ⇒ n_joints=55,
rotation_representation='6d', include_scales=True — all read from the checkpoint's own tensors
and cross-checked against `config.N_POSE` before use). This is the checkpoint the probe below
actually uses — genuinely already-trained, no new training performed, just not the file path the
handoff assumed.

## Method

24 synthetic specimens, 2048 points each, sampled with A2's validated area-weighted
face-tracked sampler (so every input point's true leg/segment is known exactly, not
approximated). Fed through the frozen, already-trained network's `sa1` and `sa2` set-abstraction
layers only — **never `sa3`**, since `sa3`'s `group_all=True` collapse into one pooled 1024-d
vector is exactly the architectural flaw finding #1 diagnosed. Each region centroid's true label
is recovered exactly (not estimated) via `pointnet2_utils.py`'s own guarantee that FPS centroids
are literal input-point indices, never interpolated — checked directly in the source before
relying on it. Two **linear** probes (deliberately linear, not MLP, so a positive result means
the DensePose-style split is already close to linearly readable in the frozen feature, not merely
recoverable given enough probe capacity):

## Results

| region | task | metric | value | n_test | chance/baseline |
|---|---|---|---:|---:|---|
| sa1 (512 regions, radius 0.1–0.4) | leg-identity (7-way: body + 6 legs) | accuracy | **0.850** | 3687 | ~0.14 (uniform) |
| sa1 | within-leg chain position (co=0..pt=1) | R² | **0.838** | 1478 | 0.0 |
| sa2 (128 regions, radius 0.2–0.8) | leg-identity (7-way) | accuracy | **0.866** | 922 | ~0.14 |
| sa2 | within-leg chain position | R² | **0.706** | 418 | 0.0 |

Confusion matrices and regression scatter plots: `fig_A3_sa1_sa2_linear_probes.png`.

## What it means

Both leg-identity and within-leg-position information are **already substantially, near-linearly
present** in `sa1`/`sa2`'s per-region features — well before the network's own regression head
ever sees them, and well before `sa3` destroys that locality by pooling everything into one
vector. This is a positive result for the Phase 10 design's premise (§4 of
`SESSION_SYNTHESIS_20260825.md`): a DensePose/CSE-style head reading from `sa1`/`sa2` instead of
`sa3` is not just architecturally plausible in the abstract, the information it would need is
demonstrably *available* at that point in this exact, already-trained network — the bottleneck is
specifically `sa3`'s collapse, not an earlier representational failure. sa2's slightly lower
chain-position R² (0.706 vs sa1's 0.838) is consistent with its coarser receptive field
(radius 0.2–0.8 vs 0.1–0.4) pooling across more of each leg's length, not a contradiction of the
main result.

**Caveat, stated plainly:** this network was trained end-to-end for parameter regression, not for
correspondence — so this shows the *information survives training incidentally*, not that a
DensePose-style head trained explicitly for it would reach these numbers or higher. It is a
lower-bound/existence argument for B1, not a substitute for actually building and training that
head.
