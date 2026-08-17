# Task 2 + Task 3 summary — D1 recipe confirmation

Both tasks ran the full D1 recipe (`fitter_3d.optimise_hierarchical` -> `fitter_3d.optimise_moonshot`)
at full iteration budget (1000+1000) on 1x H100, per the exact preflight/harness established in
`diagnostics/d1_evidence/` (Task 1, commit 9ebe508). Full evidence, per-specimen tables, exact
commands, and logs are in `diagnostics/rest_edge_evidence/` and `diagnostics/offset_sweep_evidence/`
respectively.

## Task 2 — rest-edge regularizer confirmation

**Verdict: REJECT `rest`.** Tested on the same 10-specimen `bench50_clean` set as Task 1, with the
hierarchical placement stage shared between arms (edge_mode is not read there) and only the
moonshot handoff stage varied. `edge_mode: rest` cuts `edge_logratio_absmean` ~18x on 10/10
specimens, but this is a uniform trade-off, not a clean win: `folded_face_frac` regresses on
10/10 specimens, `dihedral_p99` nearly doubles in aggregate, and `fscore@0.01`/`chamfer_l2` both
collapse well outside the pre-registered gate (fscore -0.194, chamfer +147%). Per the
pre-registered decision rule, a majority-of-specimens regression on an integrity metric is a
REJECT, not the mixed/opt-in branch — the pattern here is metric-vs-metric consistent across
every specimen, not specimen-level heterogeneity. `edge_mode: shrink` stays the D1 default.

## Task 3 — offset-penalty strength sweep

**Verdict: CONFIRMED, 5.0/2.0 is already near-optimal (closed non-finding).** Swept
`w_offset` in {2.5/1.0, 5.0/2.0 control, 10.0/4.0, 20.0/8.0} through the full D1 recipe on the
synthetic ground-truth round-trip corpus (`synth_clean` + `synth_noisy`, 12 specimens each),
scored by `correct_frac` (direct correspondence correctness). No setting beat the 5.0/2.0
control's `correct_frac` by more than the pre-registered ~3 percentage-point noise floor on
either corpus — the largest gap was -1.28pp (w20 vs control, synth_noisy). `correct_frac` trends
monotonically down as `w_offset` increases past the control on both corpora, which is worth
carrying forward as a soft prior against increasing the weight further, but the trend never
clears the noise floor even at 4x the control, so it is reported as a confirmed non-finding, not
adopted as a change. `w_offset` stays at 5.0/2.0 in both stages.

## Final recommended `D1_low.yaml` values

**Unchanged from the version committed at 9ebe508** (Task 1): `edge_mode: shrink` in both
`Stage_2_deform_coarse` and `Stage_3_deform_fine`; `w_offset: 5.0` (coarse) / `2.0` (fine). Both
open recipe questions from `diagnostics/FINAL_REPORT.md` §6.2 are now closed with evidence in
favor of the status quo — no config edit is needed.

## Ready for N=50 / multi-seed validation: **yes**

Both Task 2 and Task 3 produced settled verdicts against the pre-registered decision rules, and
neither changes the D1 recipe. `config.py` was not touched by either task (verified: neither
`optimise_hierarchical.py` nor `optimise_moonshot.py` reads `config.PLOT_RESULTS`, so the flag
that caused Task 1's stall is not a factor on this chain — confirmed via `git diff config.py`
showing no diff at any point in this work).
