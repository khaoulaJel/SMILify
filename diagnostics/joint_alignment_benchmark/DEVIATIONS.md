# Deviations from PREREGISTRATION.md

Each entry is dated and was recorded before the affected analysis was looked at, unless stated.

## D1 — 2026-09-14 — FK pivot extraction bug (fixed before any strategy arm was scored)

**What.** §5.1 defines the primary model joint as `J_transformed + trans`. The first
implementation of `tools/model_joints.py` used the joints returned by `SMAL3DFitter(return_joints=True)`.
For this model file (`static_joint_locs` absent → `config.STATIC_JOINT_LOCATIONS = False`), those are
`J_regressor` applied to the skinned **pre-deform** vertices, not the FK pivots.

**Caught by** a semantics check: at rest `J_transformed` equals `J_regressor @ v_template` to 9e-8,
but the returned joints differ from `J_transformed + trans` by up to 0.0095 (normalised units) after
posing and per-joint scaling.

**Effect on the frozen text.** §5.1's sentence "the two differ by a median 0.6% WL on production"
compared SKIN with REG, not FK with REG. With true FK pivots, FK vs REG differs by a median **3.8% WL**
per joint on production. The per-specimen medians barely move (production: FK 25.0%, REG 24.7%,
SKIN 25.1%). The swap corroboration is unchanged (5/5). The primary definition is unchanged. The
previously mislabelled quantity is kept as a third, labelled definition `SKIN`.

**Status.** No strategy arm had been fitted or scored when this was fixed (array 4092031 still pending).

## Note N1 — 2026-09-14 — Dolichoderus stays excluded (no deviation)

A re-annotation of Dolichoderus was attempted. The uploaded scene holds the correct scan, but only
the 12 surface landmarks were placed; the joint markers are unplaced placeholders. The user decided
not to re-annotate the joints. The pre-registered exclusion (§3.3, n = 11) therefore stands
unchanged. The uploads are filed, not installed, in `annotation/landmarks_recheck_20260914/`.
Recorded limitation: no Dolichoderinae in the benchmark.
