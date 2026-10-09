# Reconciliation Report — synthesis doc vs. new doc contradictions

Date: 2026-09-07. Compiled from repo evidence (git log, diagnostics/ scripts and reports) by four
parallel investigations. Every claim below cites a file path, line number, or commit hash — treat
anything without a citation as unconfirmed.

---

## Tier 1 — direct contradictions

### 1. scale_cap — shipped fix AND null mechanism result. Not a contradiction; a sequencing gap.

**Verdict: (a) same intervention, measured two different ways. X1 came 17 days *after* the N=838
ship, and asked a question the N=838 validation never asked.**

- Adoption commit: `3ca5799c` (2026-08-13), "ship scale_cap, full-corpus A/B decision" — promotes
  `w_scale=0.052` into `D1_PROD.yaml`. What it measured: the **proxy** —
  `exp(log_beta_scales).max()/min()`, a scale-outlier tail statistic, reduced 58–70%.
- X1 commit: `731724f3` (2026-08-31), `diagnostics/anterior_mechanism/` (`PREREGISTRATION_X1_scalecap_mechanism.md`,
  `RESULTS_X1_scalecap_mechanism_20260830.md`, `x1_anterior_mechanism.py`, configs `A_nocap.yaml` /
  `B_scalecap.yaml`, differing only in `w_scale: 0.0` vs `0.052`). What it measured: the **mechanism**
  — the "head-carried deform ratio" (head 0.793→0.802, mandible 1.783→1.856, antenna 1.668→1.704),
  none moving significantly toward 1.0 (sign p = 0.32, 0.32, 1.00). X1's own proxy arm reproduced the
  shipped tail-reduction, confirming the harness is faithful — the mechanism metric alone is flat.
- X1's commit message states directly: "Nothing had ever measured whether the shipped term moved the
  head ratio... X1 adds a fourth measurement that was never taken." The N=838 validation was already
  shipped before anyone checked whether it fixed the anatomy it was meant to fix.
- Current production value: `diagnostics/moonshot/cfg/D1_PROD.yaml` lines 45 and 65,
  `w_scale: 0.052`, header comment lines 9–20 documents "PROMOTED 2026-08-13." **Still present, not
  reverted**, despite X1's null mechanism result.
- Recipe sweep (Tier 3 Q6): `w_scale: 0.052` also present in `diagnostics/anterior_mechanism/B_scalecap.yaml`,
  `diagnostics/deform_nature/B_dsym2.yaml`, `C_dsym10.yaml`, `diagnostics/joint_blendshapes/C_couple_jresid.yaml`
  (all inherit the D1_PROD default). `A_nocap.yaml` is the deliberate X1 control (`w_scale: 0.0`).
  `diagnostics/atta_reference/cfg_arm_a_master.yaml` has **no** `w_scale`/`scale_cap` key at all — it's
  an unrelated pre-existing master config (see Q5 below).

**Bottom line for slides: scale_cap fixed the statistic it was designed against (proxy), was shipped
on that basis, and 17 days later was shown NOT to move the anatomical symptom (mechanism). Both
things are true; they are not competing claims about the same measurement.**

---

### 2. Genus classification 5.6–6.1× — which pipeline, and does it survive?

**Verdict: the positive numbers are from morphometric-measurement Mosimann log-shape-ratios (not
betas), and this feature set is confirmed distinct from anything using raw beta-space PCA/UMAP.
However, the specific "§41" document described could not be located in this repository — the
YES/NO reconciliation the user proposed cannot be verified against §41 itself.**

- Source of 6.1×/3.8×: `diagnostics/morphometrics/REPORT_MORPHOMETRICS.md` (line 244: genus 13.0%
  vs null 2.1±0.7%, p<0.0001, **6.1×**; line 39: cross-corpus **3.8×**), produced by
  `diagnostics/morphometrics/analyse.py` (LOO 1-NN + permutation null), driven by
  `run_m1_fit_all.sh` + `measure.py`, on a 616-worker + 81-`ALL_ANTS_CLEAN` corpus, **without**
  scale_cap.
- Source of 5.4×/5.3×: `diagnostics/khaoula_v2/REPORT_MORPHOMETRICS.md` §1 — the **same** `analyse.py`
  pipeline, run on the combined 838-specimen corpus (757 `worker_ALT` + 81 `ALL_ANTS_CLEAN`), as an
  explicit scale_cap A/B: arm A (no cap) = **5.4×**, arm B (cap) = **5.3×**.
- Feature vector (confirmed from `analyse.py`): `features()` (line 239) computes Mosimann
  log-shape-ratios of mesh-measured lengths (`ms.log_shape_ratios(X)`) restricted to
  `measure.CORE_BLOCKS`. This is **not** raw betas — a separate `features_with_betas()` (line 118)
  exists but belongs to the later Z12 experiment (traits+betas combined, 0.179 top-1), not this one.
- **§41 as described (raw beta-space PCA/UMAP failing, PC1 correlating with fit quality) was not
  found anywhere in the repo** — zero hits for "§41", and the closest candidates
  (`diagnostics/pose_causality/y1_shape_space_analysis.py`, `diagnostics/absolute_scale/out_validation/pc1_tautology_check.log`,
  `diagnostics/FINAL_REPORT.md`'s UMAP/HDBSCAN section) either measure something else entirely or —
  in `FINAL_REPORT.md`'s case — use the *same* morphometric-measurement feature space as the positive
  result, not raw betas, and its negative clustering finding (ARI ≤0.004) is framed there as
  consistent with, not contradicting, the positive 1-NN classification.
- **Answer to the user's proposed reconciliation: cannot confirm YES/NO — §41 isn't in this repo.**
  If it exists as an external document using raw beta PCA/UMAP, the "measurement-averaging survives
  correspondence noise, raw beta-space doesn't" hypothesis is structurally plausible given
  `y1_shape_space_analysis.py`'s finding that beta-space structure degrades under pose noise — but
  this needs the actual §41 source to verify.

### Tier 3 Q7 — 6.1×/3.8× vs 5.4×/5.3×: already reconciled, and it touches scale_cap directly.

Explicitly reconciled in `diagnostics/khaoula_v2/REPORT_MORPHOMETRICS.md` lines 9–16 and
`diagnostics/EXECUTION_PLAN.md` §5 (~lines 205–215): **not a regression, a different (larger,
combined) test population.** 6.1×/3.8× = Fabian's original 616-worker run, tested separately from
81 `ALL_ANTS_CLEAN`, no scale_cap. 5.4×/5.3× = the *same* combined 838-specimen corpus's scale_cap
A/B pair (5.4× no-cap, 5.3× capped) — judged "statistically flat" (0.3pp gap, within the null's
±0.7% spread). This is consistent with the X1 finding above: scale_cap doesn't move genus
classification either way, matching "fixes a statistic, not anatomy."

---

### 3. GNC — marginal-and-rejected, or genuinely improved?

**Verdict: the two numbers are not the same metric — they cannot be compared as an
improvement-over-time. `gnc_legonly_topofree` fixed a topology-index crash, not the antenna
regression per se, and GNC remains unshipped/experimental as of the latest evidence.**

- Earlier number (+0.03–0.05 leg_distal, "leg-scoped opt-in at most"): a genuine **Pearson
  correlation** from `calib_features.py` (`np.corrcoef`) — leg_distal R 0.144 → 0.179–0.194, N=12,
  with every other feature block (antenna, gaster, head, mandible, ALL-feature) *dropping*. Source
  file (`diagnostics/correspondence_accuracy/GNC_R_METRIC_CORRECTION.md`) no longer exists on disk —
  survives only via memory `project_correspondence_accuracy_metric.md`.
- Later number (drop30 0.602→0.797, drop60 0.411→0.607): from `diagnostics/cycle2_20260819/CYCLE2_REPORT.md`
  (recoverable via `git show c462a95c:diagnostics/cycle2_20260819/CYCLE2_REPORT.md`, not in working
  tree), sections B6/B7. These numbers come from `audit_run` in `probe_d2b_correspondence_audit.py`,
  which returns per-specimen **mismatch fractions**, not a correlation — confirmed no `corrcoef` call
  anywhere in that file. It was informally mislabeled "R" throughout Track B purely because the
  numeric range (0.4–0.96) coincidentally resembles a correlation's range.
- **Explicit answer: not the same metric, not the same statistical family.** §7's number is a real
  Pearson R on morphometric traits; §13's number is a mismatch-fraction quantity from a
  differently-named, non-correlation function. "+0.03 → 0.797" is not a valid before/after comparison.
- `gnc_legonly_topofree` is a later, targeted fix — but for a **crash**, not the antenna regression:
  `gnc_legonly` failed outright on the `drop30`/`drop60` damage corpora because those corpora
  renumber faces/vertices, breaking the face-index-based leg/non-leg split
  (`CYCLE2_REPORT.md` lines 427–491). `robust_chamfer_leg_split_topofree`
  (`fitter_3d/trainer_moonshot.py`) removed that topology-index dependency. "Avoiding the antenna
  regression" (§13's framing) is a side-observation in the same report, not the fix's design target.
- **Caveat**: no evidence `bootstrap_damage_b6b7.py` — written explicitly to re-verify B6/B7's claim
  with an unambiguous metric (`leg_acc`/`seg_acc` from `confusion.py`) — was ever actually run (no
  output under `diagnostics/correspondence_accuracy/out/`). So the §13 damage-corpus numbers remain
  unverified with a correctly-defined metric, unlike the clean-data `gnc_legonly` claim, which *was*
  re-verified and **failed to replicate** (every metric crossed zero under paired bootstrap, per
  memory `project_correspondence_accuracy_metric.md`).
- **Current status: unshipped.** `D1_PROD.yaml` and `SHIPPED_RECIPE.md` have zero GNC references.
  Memory `project_fabian_synthesis_20260819.md`: "still experimental/opt-in, never merged into
  D1_PROD.yaml, never tested combined with scale_cap." No later document promotes it.

---

## Tier 2

### 4. Objective-decomposition experiment (production vs. oracle, term by term)

**Verdict: already run — twice. First run (G4) was retracted as an artifact; the corrected version
(G6) is load-bearing and gives an answer. The new doc not reporting a result is out of sync with
`diagnostics/groundtruth/`.**

- `diagnostics/groundtruth/g4_objective_split.py` (2026-09-01) does exactly what §40/§52 propose:
  evaluates the full production loss (chamfer, edge, normal, laplacian, offset, sym, mid, scale,
  limit) at both production-fitted and oracle parameters. Result: production joint_resid 38.86% vs
  oracle 7.36%, but oracle total loss 500× worse (0.437 vs 0.000867).
- **Retracted**: `RESULTS_G6_20260901.md` lines 76–82 — the unconstrained oracle drove betas to
  `|z|~19–26.65` (vs production's `|z|~0.81–1.0`), an anatomically implausible shape. G3's 3.7% and
  G4's verdict were both withdrawn as artifacts of an unconstrained refit.
- **Corrected version, `g6_plausible_oracle.py`**: bounds `|z|≤k`, sweeps k=1,2,3,∞. At `|z|≤1`
  (matching production's own magnitude): joint residual 11.01% vs production's 38.86% (3.5× better),
  at 34× worse chamfer cost. **Conclusion: the objective isn't mis-weighted — it's under-determined
  w.r.t. skeleton position; no term constrains joint position at all.** Proposed next step (Pareto
  sweep of chamfer + λ·joint_target) not yet run.
- Oracle parameter vectors are **not saved to disk** — no save/pickle/torch.save call in either
  script, only scalar loss-term JSON (`g4_results.json`, `g6_results.json`) persists. The oracle is
  refit in-memory via Adam each run; **not reproducible without re-running the optimization**, though
  the annotations it refits against (`annotation/gt_batch1/*_joints.json`) are on disk.

### 5. Atta / physical-measurement replication

- Complete and reported: `diagnostics/atta_reference/REPORT_ATTA_HEADWIDTH.md` (status "Complete —
  2026-09-06"). 20 specimens, fit via stock `origin/master@b5bf9565`
  `fitter_3d/optimise.py`. Whole-skeleton median error 1.05% (55 joints). New `b_h_l`/`b_h_r`
  head-width bones: median 1.77%, max 4.85%, full per-specimen table present (e.g. specimen 10: ref
  2.310mm vs replicated 2.273mm, −1.59%). Allometric exponents match reference within CI. Two addon
  bugs documented in `BUG_REPORT.md` (one fixed locally, tracked as issue #92 item A1).
- **The Atta fits predate scale_cap entirely.** Config used (`diagnostics/atta_reference/cfg_arm_a_master.yaml`)
  has zero `w_scale`/`scale_cap` keys. Fit commit `b5bf9565` is 2026-07-16; scale_cap shipped
  2026-08-13 (`3ca5799c`) and X1 landed 2026-08-31 — both roughly a month **after**. The report
  write-up itself is recent (2026-09-06, commit `fbc4cbf6`), but the underlying fit used a
  pre-scale_cap master config, so scale_cap's mechanism status does not bear on these 20 specimens.

---

## Tier 3

**6.** Covered under Q1 above — `w_scale: 0.052` present and current in `D1_PROD.yaml` and every
recent recipe derived from it; absent from the pre-existing Atta config and the deliberate X1 control.

**7.** Covered under Q2 above — reconciled: different corpus sizes, and the pair is literally the
scale_cap A/B toggle on the larger combined corpus, judged statistically flat.

**8. Functional maps: confirmed never implemented.** `grep -rn "functional_map\|laplace_beltrami\|fmap"`
across the whole repo returns only `diagnostics/khaoula_v2/t10_diff3f_sanity.py` lines 162–218, where
`fmap` is a local variable name for a CNN feature map — unrelated to functional-map correspondence.

**9. D1 recipe HOLD status: still HOLD, but the reason has moved twice.** Original HOLD (`27d6410f`)
→ revised to PROMOTE (`138510d0`) → **reverted to HOLD** (`80b48104`, 2026-08-18): "the stratified-10
reference was carved out of the same 50-specimen pool... the alphabetical-first-10 is shown biased."
→ same day, a genuinely independent 80-specimen non-alphabetical holdout was built and rerun
(`46d05fe2`): bars 1–2 (no catastrophic failures, tight cross-seed CV) now pass, but **bars 3–4 fail
again** — this time because the new holdout corpus is intrinsically easier (fscore@0.01 0.896 vs
0.500), not because of a recipe regression. **Final verdict: "HOLD, unchanged — but the reason has
moved,"** with a recommendation to retire bars 3–4 in favor of corpus-relative checks. (These files
were removed from the working tree by merge `f0b7322f` but are recoverable via
`git show 46d05fe2:diagnostics/d1_n50_evidence/scorecard.md`.)

**10. W-series vertex counts (4,096 vs 10,235): stable, confirmed, but "dead-joints exclusion list"
by that name could not be located.** No `dead_joint`/`DEAD_JOINT`/`excluded_joint` symbol exists
anywhere in the repo — if this list exists it goes by a different name; flag this as unconfirmed
rather than assumed-present. Vertex counts themselves are solid:
- 10,235 (full mesh): `PREREGISTRATION_W3_contextual_embedding.md` lines 20, 54; corroborated in
  `diagnostics/moonshot/sdf_stratify.py` lines 13/83/88 (`verts`/`deform_verts` shape
  `(n_specimens, 10235, 3)`).
- 4,096 (density-matched sample): same preregistration line 58; `RESULTS_W3_contextual_embedding_20260830.md`
  line 11; default `--n_points` in `w3_contextual_embedding.py` line 288.
- W3's p=7.8e-26 result: `RESULTS_W3_contextual_embedding_20260830.md` line 26 (coxa row: +16.4% rel.,
  640/960, sign p=7.8e-26, PASS; other segments tr/fe/ti at p=9.1e-27/1.1e-43/1.3e-105, all "no").
  Both constants are used consistently within this report and reused unchanged in the later
  `sdf_stratify.py` (part of the Aug 13 ship commit) — no later script redefines either number.

---

## Summary table

| # | Question | Verdict |
|---|---|---|
| 1 | scale_cap shipped vs null | Both true — proxy fixed & shipped (Aug 13), mechanism checked & failed 17 days later (X1, Aug 31). Still shipped (`w_scale=0.052` in D1_PROD.yaml). |
| 2 | Genus 5.6–6.1× pipeline | Morphometric Mosimann-ratio features, not betas. §41 (raw beta PCA/UMAP) not found in repo — cannot verify user's proposed reconciliation. |
| 3 | GNC marginal vs improved | Different metrics entirely (Pearson R vs mismatch-fraction mislabeled "R") — not comparable. GNC unshipped/experimental. |
| 4 | Objective decomposition | Already run (G4, retracted as artifact; G6, corrected, load-bearing): objective is under-determined on joint position, not mis-weighted. Oracle vectors not saved to disk. |
| 5 | Atta replication | Complete, reported, median 1.77% head-width error. Predates scale_cap by ~1 month — irrelevant to this fit. |
| 6 | scale_cap in recent recipes | Present in D1_PROD-derived configs; absent from pre-existing Atta config. |
| 7 | 6.1×/3.8× vs 5.4×/5.3× | Reconciled: corpus-size difference; the smaller pair IS the scale_cap A/B, statistically flat. |
| 8 | Functional maps | Never implemented — confirmed, zero real hits. |
| 9 | D1 HOLD status | Still HOLD — re-selected once to an independent corpus, which turned out too easy; recommend retiring bars 3-4. |
| 10 | Dead-joints / 4096 vs 10235 | Vertex counts stable and confirmed. "Dead-joints list" by that name not found — flag as unconfirmed. |

---

*Compiled by four parallel research agents from git history and diagnostics/ reports. All claims
above are traceable to file paths, line numbers, or commit hashes cited inline; anything the agents
could not locate is stated explicitly rather than inferred.*
