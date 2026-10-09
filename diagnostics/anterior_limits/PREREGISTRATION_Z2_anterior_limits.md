# Pre-registration — Z2: does closing the anterior/body-axis rotation gap move the head defect?

**Written and committed BEFORE the run.** Date 2026-08-31. Series `Z`. Folder
`diagnostics/anterior_limits/`.

## 1. The claim under test

X1 closed the "add another scale regulariser" path and named the next lead as a **representation
gap**, not a magnitude gap. Z1b measured it exactly:

- `OmniAnt_25PCs_joint_limited.pkl` constrains **99 of 165** rotation axes.
- The unconstrained non-leg joints are `b_a_1..b_a_5`, `b_h`, and all six antenna joints.
- **`b_h` carries 2180.69 skinning mass — the highest of any joint in the model — with all three
  axes free.** Fitted rotations on bench50 reach **101.1°** at the head, **174.0°** at `an_2_l`,
  **135.9°** at `b_a_5`. Medians are 2–6° on the body axis: this is a pure tail.
- Where limits *are* authored the hinge binds exactly: `l_1_fe_r` max 70.1° against a 70° limit.

> **H: the head's under-deformation (REPORT §6.10's head-carried ratio, 0.70–0.79 across every
> open-shape arm measured) is sustained by unconstrained head/body-axis rotation — the fitter
> satisfies anterior geometry by rotating and scaling rigid parts rather than by deforming them.**

## 2. Arms

Identical in every respect except `SMILIFY_SMAL_FILE`. Both use `D1_PROD.yaml` **unchanged**
(`w_limit: 0.006` already shipped), corpus `diagnostics/moonshot/bench50_clean` (50 real workers,
37 genera — the corpus §6.10 and X1 were measured on), seed 0.

- **A** `OmniAnt_25PCs_joint_limited.pkl` — as shipped.
- **B** `OmniAnt_25PCs_anterior_limited.pkl` — A plus symmetric bands on the 12 free non-leg
  joints (head ±60°, petiole/postpetiole ±45°, gaster ±40/45°, antennae ±90/75°). Built and
  readback-verified by `z2_build_limited_model.py`; every other axis asserted byte-identical.

Legs are **deliberately out of scope**: 18 leg axes remain unauthored in both arms. The leg axis
is the one this project has already closed (W1–W5, C14p), and including it would make a positive
result unattributable.

## 3. Endpoint — MECHANISM

**§6.10's head-carried ratio: mean `|deform_verts|` on head vertices ÷ same on thorax vertices**,
paired across the 50 specimens. Arm A is expected to reproduce 0.70–0.79. The corrected value is
**1.0** (the head should deform like the thorax).

- **PASS** — |ratio − 1.0| falls by **≥ 0.05** with sign-test p < 0.05.
- **PARTIAL** — moves toward 1.0 significantly but by less than 0.05.
- **FAIL** — no significant move toward 1.0.

## 4. VOIDING checks — a failure on any of these voids the endpoint

1. **The limits bind.** In arm B, max |rotation| per constrained axis must be ≤ its authored band
   (+1° tolerance) on every specimen, and arm A must exceed it on at least one. If B never
   touches its bands the intervention did nothing and the endpoint is meaningless.
2. **Shape space open in both arms.** `betas` mean |z| vs the model prior > 0.5 in A and B. This
   is the Z1 lesson: a result measured against a frozen shape channel is not evidence.
3. **Thorax denominator non-degenerate.** Mean thorax `|deform_verts|` within 0.019–0.043 (§6.10's
   range) in both arms, so a near-zero denominator cannot inflate ratios. *(Note: the open-shape
   arms measured in Z1b sit at 0.0041, an order of magnitude below §6.10's closed-shape range. If
   both arms land there consistently the check is reported as a scale shift, not a void — but the
   ratio must then be read as within-arm paired, which it is.)*
4. **Fit quality reported.** `chamfer_l1` and `fscore@0.01` from `metrics.csv`, both arms. A
   mechanism move bought by a collapsed fit is not a pass.

## 5. PROXY — reported, explicitly NOT decisive

`gen@20/spread`, per-part `within_tau`, and per-joint scale statistics (head max ×, `b_a_5` p99
`|log s|`). **Per this project's rule (X1, seven prior instances), a proxy move without the
mechanism move in §3 is a FAIL, not a pass.** These are recorded to characterise the intervention,
not to promote it.

## 6. What each outcome licenses

- **PASS** → author the limits properly per-axis in Blender (`docs/joint_limits_user_guide.md`);
  the symmetric bands here are a test instrument, not a shipping candidate.
- **PARTIAL** → report the fraction; do not ship.
- **FAIL** → the anterior representation gap is **closed as a hypothesis**. The head defect is
  sustained by something other than rotation freedom, and the next candidate must come from
  elsewhere. Do not re-run with different band values — that is the scale-cap-variant pattern X1
  closed.

## 7. Out of scope, stated in advance

Scan quality, correspondence, pose initialisation, leg axes, and the `b_a_5`/low-skinmass scale
runaway (a joint owning 5.56 skin mass scaling 28× affects almost nothing; it is cosmetic, and
saying so in advance stops it being promoted post hoc if it happens to move).
