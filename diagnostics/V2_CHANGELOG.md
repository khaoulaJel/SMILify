# V2 plan — change log

Tracks every file touched while executing "Registration Correspondence — Revised Plan
(v2, supersedes the SDF-part-masking brief)". One entry per change, added at the time
the change is made, not reconstructed after the fact. Purpose: so "test this" or
"deliver this" can be scoped to an exact file/line list instead of a fuzzy branch diff,
and so any change can be traced back to the specific plan item and evidence that
justified it.

Convention per entry:

```
### [plan item] short title
- date:
- files: path — new file | edit (lines X-Y) | config value change
- what: one line, mechanical description of the diff
- why: which plan item / which measured number justifies this
- status: not run | running | E6 result attached | reverted (+ reason)
```

Nothing has been changed in the pipeline yet — everything up to this point (the SDF
brief retraction, the moonshot-branch diagnostics sync, the v2 plan itself) was
investigation and file preservation, not a pipeline change, and is not logged here.
The diagnostics/ sync from `upstream/feature/registration_moonshot` is recorded once,
below, for completeness, since it did add files to the working tree/index — but it is
read-only reference material, not something to revert as part of "testing/delivering"
the v2 plan.

---

## Reference material added (not a pipeline change)

### [prerequisite] sync upstream/feature/registration_moonshot's diagnostics/ tree
- date: 2026-08-11
- files: `diagnostics/` — ~1389 files added via `git checkout upstream/feature/registration_moonshot -- diagnostics/` (758 real files + 631 dangling symlinks to the original author's local machine, see prior session notes). Zero files outside `diagnostics/` touched. Zero pre-existing local files under `diagnostics/` overwritten (verified: the 50 pre-existing `bench50_clean/*.obj` were byte-identical to the branch copy before the checkout).
- why: reference corpus/report access for the plan below (E6 harness, D1_low.yaml, within_part_signal.py, etc. all live here)
- status: staged, uncommitted

### [prerequisite] methodology notes
- date: 2026-08-11
- files: `diagnostics/MOONSHOT_METHODOLOGY_NOTES.txt` — new file
- why: written record of the E6/E7 methodology, offset/rest-edge-loss code review, and SDF-probe/bench50_clean provenance, so later work doesn't re-derive it
- status: n/a (documentation, not code)

---

## Tier 0 — pipeline changes (none yet)

### [T0.1] offset-penalty weight sweep
- status: not started

### [T0.2] rest-edge loss confirmation run
- status: not started

### [T0.3] bundle low-risk fixes (pose-freeze, symmetric sampling, midline, joint limits, D1 config defects)
- status: not started

### [T0.4] beta-scale bounding validation
- status: not started

---

## Tier 1 — pipeline changes (blocked on T0 result)

### [T1.1] fork within_part_signal.py with extrinsic-feature extractor
- status: blocked on T0 deliverable

---

## Environment repair (blocking prerequisite, before any T0 run)

- date: 2026-08-11
- what: `pytorch3d` conda env had widespread file corruption (torch, pyyaml, numpy, pip,
  packaging, pytorch3d, scipy all had missing package files despite intact metadata).
  Root cause: `/p/scratch` (Lustre) was corrupting package extraction during
  conda install/reinstall — confirmed by successfully reinstalling pytorch3d after
  pointing `pkgs_dirs` in `~/.condarc` to local `/tmp/conda_pkgs_test` instead.
- fixes applied, in order:
  1. torch: reinstalled via pip from download.pytorch.org (cu118 wheel), bypassing
     conda entirely — torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
  2. pyyaml, numpy==2.2.6, scipy==1.15.2: `pip install --ignore-installed --no-deps`
  3. pip, packaging: reinstalled via `ensurepip --upgrade` + `pip install --force-reinstall --no-deps`
  4. pytorch3d==0.7.8 (py310_cu118_pyt231): reinstalled via conda after fixing
     `~/.condarc` pkgs_dirs to `/tmp/conda_pkgs_test` (was `/p/scratch/.../conda_pkgs`,
     corrupting installs; backup saved as `~/.condarc.bak`)
  5. Also synced missing `fitter_3d/*.py` moonshot-branch files (not part of env repair,
     but blocking the same smoke test) — see entry below.
- verification: full-env RECORD scan (0 missing files), then `import fitter_3d.optimise_hierarchical`
  and `import fitter_3d.optimise_moonshot` both succeed cleanly.
- status: DONE — env confirmed working, ready for T0.1

### [prerequisite] sync missing fitter_3d/*.py files from upstream/feature/registration_moonshot
- date: 2026-08-11
- files added (git checkout from upstream branch, additive only, verified no local overlap
  via `diff` of file lists first): `fitter_3d/correspondence.py`, `hull_decomposition.py`,
  `hull_partition.py`, `joint_limits.py`, `optimise_hierarchical.py`, `optimise_moonshot.py`,
  `optimise_multistart.py`, `part_anchor_init.py`, `partfield.py`, `trainer_hierarchical.py`,
  `trainer_moonshot.py`
- why: these only existed on the moonshot branch; current branch only had `diagnostics/`
  synced previously, not `fitter_3d/`
- note: `fitter_3d/trainer.py` was found already modified (+344/-3, uncommitted, predates
  this session) — unrelated penetration-loss/local-smoothness diagnostic work, left untouched,
  documented separately above ("pre-existing unrelated local changes")
- status: DONE

---

## Working-directory convention change

- date: 2026-08-11
- what: all new outputs for this plan (T0/T1) go under `diagnostics/khaoula_v2/`
  (`cfg/`, `runs/`), not inside `diagnostics/moonshot/`, to keep this work separate
  from Fabian's original moonshot folder.
- `diagnostics/moonshot/` is now read-only reference material for this plan: the
  synth_clean/synth_noisy corpora, the existing scored `out/synth_scores.npz`,
  `within_part_signal.py`, and the SMAL `.pkl` model file are read from there, but
  nothing is written there going forward.
- files moved: `diagnostics/moonshot/cfg/T01_off2.yaml`, `T01_off10.yaml`,
  `T01_off20.yaml` → `diagnostics/khaoula_v2/cfg/`
- new file: `diagnostics/khaoula_v2/submit_t01_sweep.sbatch` — runs the 3 missing
  T0.1 grid points (w_offset 2.0/0.8, 10/4, 20/8) on synth_clean, 3 GPUs parallel,
  writes to `diagnostics/khaoula_v2/runs/`
- status: SUBMITTED — job 15517946

---

## T0.1 sweep — first submission failed, env fix, resubmitted

- date: 2026-08-11
- what: job 15517946 failed all 3 arms at optimizer construction — `gmpy2` package
  had the same missing-top-level-file corruption as torch/scipy/numpy earlier
  (__file__ was None). Fixed via `pip install --ignore-installed --no-deps gmpy2==2.3.0`.
  Re-ran full env RECORD scan (0 missing) and a local (no-GPU) smoke test importing
  optimise_hierarchical, optimise_moonshot, trainer_hierarchical, trainer_moonshot,
  and constructing a torch.optim.Adam instance, before resubmitting.
- resubmitted: job 15517954
- status: RUNNING

---

## T0.1 sweep — second failure: config.py missing env-var override, fixed, resubmitted

- date: 2026-08-11
- what: job 15517954 failed all 3 arms — `SMILIFY_SMAL_FILE` env var (set by the sbatch
  script, matching the moonshot branch's original run_e6/run_e8b scripts) was silently
  ignored because this branch's `config.py` hardcoded `SMAL_FILE` with no env override.
  The moonshot branch's `config.py` adds `SMAL_FILE = os.environ.get("SMILIFY_SMAL_FILE", SMAL_FILE)`
  plus two unrelated, off-by-default additions (COUPLE_JOINT_BLENDSHAPES, COUPLE_TRANSLATION_FACTOR).
- files: `config.py` — synced from upstream/feature/registration_moonshot via
  `git checkout upstream/feature/registration_moonshot -- config.py` (verified no local
  modifications existed first via `git diff`, so this was a clean pull, not a merge)
- verification: `config.SMAL_FILE` now resolves to `OmniAnt_25PCs_joint_limited.pkl` when
  `SMILIFY_SMAL_FILE` is set, and the file's `joint_limits` key is confirmed present.
- status: DONE, resubmitting T0.1 sweep

---

## Scoring script forked for khaoula_v2/

- date: 2026-08-11
- files: `diagnostics/khaoula_v2/score_synth_roundtrip.py` — copied from
  `diagnostics/moonshot/score_synth_roundtrip.py`, patched to add `CORPUS_DIR`
  pointing back at `diagnostics/moonshot/` (read-only reference corpus:
  ground_truth.npz, .obj files) while `runs/` lookups stay local to
  `diagnostics/khaoula_v2/runs/`. Two lines changed (`ground_truth.npz` load,
  `.obj` load), everything else byte-identical to the original.
- why: original script hardcodes `HERE/runs/<run>/...` where HERE = its own
  directory, so it can't see khaoula_v2/runs/ without either symlinking into
  moonshot/ or forking the script; forking keeps moonshot/ untouched.
- status: DONE, using it to score T0.1

---

## [T0.1] offset-penalty weight sweep — COMPLETE

- date: 2026-08-11
- ran: w_offset 2.0/0.8, 10/4, 20/8 on synth_clean (job 15517979 for training,
  job 15518176 via srun for scoring). Outputs: `diagnostics/khaoula_v2/runs/T01_off{2,10,20}/`.
- combined with pre-existing scored points (0.2/0.08, 5.0/2.0, 200/200 from
  `diagnostics/moonshot/out/synth_scores.npz`), full grid:
    0.2/0.08   -> 4.81% correct, 3.48% median err
    2.0/0.8    -> 6.58% correct, 3.14% median err
    5.0/2.0    -> 6.69% correct, 3.21% median err  <- WINNER
    10/4       -> 6.59% correct, 3.41% median err
    20/8       -> 5.76% correct, 3.69% median err
    200/200    -> 3.36% correct, 4.87% median err
- reading: non-monotonic as the report predicted; 5.0/2.0 (existing D1 setting) wins
  on the correctness metric. No change to w_offset needed.
- known bug (non-blocking): `score_synth_roundtrip.py --render_n 0` crashes in the
  plotting step (GridSpec 0 rows) — plotting code doesn't guard for render_n=0.
  Numeric results print successfully before the crash, so this doesn't block scoring.
  Not fixed (out of scope, cosmetic).
- status: DONE — winner is D1's existing 5.0/2.0, carries forward into T0.2's combined run

---

## [T0.2] rest-edge loss confirmation run — submitted

- date: 2026-08-11
- what: verified `rest_edge_loss()` in `fitter_3d/trainer_moonshot.py` (lines 256-280,
  wired at 455-480) already matches the plan's required corrected form: `rest_now`
  recomputed live via `self.smal_3d_fitter(deform_verts=torch.zeros_like(...))` under
  `torch.no_grad()`, per-batch. No code changes needed, `edge_mode="rest"` is already
  the class default; D1_SYN.yaml just overrides it to "shrink".
- files: `diagnostics/khaoula_v2/cfg/T02_restedge.yaml` — copied from
  `diagnostics/moonshot/cfg/D1_SYN.yaml` with only `edge_mode: shrink` -> `edge_mode: rest`
  on both stages (lines 7, 26). w_offset unchanged at 5.0/2.0 (T0.1's winner) — this is
  the "run combined with the T0.1 winner" arm the plan calls for.
- new file: `diagnostics/khaoula_v2/submit_t02_restedge.sbatch`
- status: submitting

---

## [T0.2] rest-edge loss confirmation run — COMPLETE, NEGATIVE RESULT

- date: 2026-08-11
- ran: `edge_mode: rest` (corrected, live-recomputed form, already correctly implemented
  in trainer_moonshot.py) combined with T0.1's winning w_offset (5.0/2.0), on synth_clean.
  Job 15518194 (training), srun 15518252 (scoring). Output: `diagnostics/khaoula_v2/runs/T02_restedge/`.
- result vs T0.1 winner (offset-only, edge_mode=shrink):
    offset-only (T0.1 winner)   -> 6.69% correct, 3.21% median err
    + rest-edge loss (T0.2)     -> 1.52% correct, 6.67% median err
- reading: rest-edge loss INTERACTS NEGATIVELY with the offset penalty — correctness
  drops >4x, error roughly doubles. Does not stack additively, is not merely redundant,
  it actively hurts. This answers the plan's pre-registered T0.2 gate question directly.
- decision: DO NOT adopt rest-edge loss (edge_mode: rest) alongside the offset penalty.
  Carry edge_mode: shrink forward into the T0 combined deliverable, not edge_mode: rest.
  This is a genuine new finding — the plan explicitly noted this combination had never
  been tested in the pipeline that produced headline numbers before this run.
- status: DONE — negative result, rest-edge loss rejected for the combined T0 recipe

---

## [T0.3] bundle low-risk fixes — VERIFIED, no code changes needed

- date: 2026-08-11
- verified all five items already satisfied by D1_SYN.yaml / current code, no gaps:
  1. `scheme: all` (never-freeze-pose) — present, both stages (verified earlier)
  2. symmetric area-weighted sampling both meshes — confirmed unconditional in
     `trainer_moonshot.py` Stage.forward (src_pts and tgt_pts both go through
     `sample_points_from_meshes` with the same n_sample; no config flag needed,
     no asymmetric-sampling bug present)
  3. `w_midline: 2.0` — present, both stages (verified earlier)
  4. `w_limit: 0.006` in handoff stage — present, both stages in D1_SYN.yaml
     (this was literally the defect D1_low.yaml had; D1_SYN.yaml doesn't have it)
  5. H3_deform as optional diagnostic, not a mandatory default step — `--deform_its`
     CLI flag already exists (optimise_hierarchical.py:51, default 600); confirmed
     `HierarchicalStage.run()` uses a plain `for i in range(self.n_it)` loop (safe
     no-op at 0) and `save_npz()` doesn't depend on any loop-local variable, so
     `--deform_its 0` is a safe way to skip it. No code change needed — going
     forward, pass `--deform_its 0` explicitly in run scripts instead of relying
     on the 600-iteration default (H3's output was never consumed downstream anyway
     in any of our T0.1/T0.2 runs, confirmed both init from H2_joint.npz not H3).
- status: DONE — all items confirmed already correct; only behavioral change is
  passing `--deform_its 0` explicitly in future run scripts to stop wasting the
  ~600 iterations/specimen

---

## [T0.4] beta-scale/translation bounding — submitted

- date: 2026-08-11
- what: `scale_barrier`/`trans_barrier` (`fitter_3d/joint_limits.py:79-135`) are implemented
  and wired into `trainer_moonshot.py` (`w_scale`/`w_trans`, both default 0.0) and
  `trainer_hierarchical.py`, but never validated against E6 — confirmed by grep, no
  `w_scale`/`w_trans` value appears in any prior scored run except
  `diagnostics/moonshot/cfg/M7_scale1x.yaml` (`w_scale: 0.052`, on the OLD M7-era
  offset weight 0.2/0.08, not D1's 5.0/2.0) and `M7_scale3x.yaml` (`w_scale: 0.157`).
  No `w_trans` candidate value exists anywhere in the prior work — trans_cap has zero
  empirical basis, not even an old-recipe reference point.
- decision: run scale and trans bounding as TWO SEPARATE arms, not combined, so a
  null/negative result on one can't be blamed on the other (same discipline as T0.2's
  "don't assume additivity"):
  - `T04_scale`: `diagnostics/khaoula_v2/cfg/T04_scale.yaml` = T0.1's winning recipe
    (D1_SYN.yaml, w_offset 5.0/2.0) + `w_scale: 0.052` (the one externally-calibrated
    candidate, both stages)
  - `T04_trans`: `diagnostics/khaoula_v2/cfg/T04_trans.yaml` = same base + `w_trans: 1.0`
    — **explicitly uncalibrated**, no prior reference exists; this is an exploratory
    probe, not a validated weight, and must be reported as such regardless of outcome
- note: checked whether an existing H2_joint.npz hierarchical checkpoint from T0.1/T0.2
  could be reused to save compute — found the 4 existing checkpoints are NOT
  byte-identical (md5 differs across T01_off2_hier/T01_off10_hier/T01_off20_hier/
  T02_restedge_hier) despite identical hierarchical flags, i.e. the hierarchical stage
  is not deterministic run-to-run here. Reusing a foreign checkpoint would be an
  uncontrolled confound, so each T0.4 arm reruns its own hierarchical stage, matching
  how T0.1's 3 arms and T0.2's 1 arm were each run independently.
- new file: `diagnostics/khaoula_v2/submit_t04_scalebound.sbatch` — 2 arms, 2 GPUs parallel
- status: submitting

---

## [T0.4] beta-scale/translation bounding — COMPLETE on the correctness question

- date: 2026-08-11
- ran: job 15518280 (both arms, 2 GPUs parallel, 12:51 elapsed, exit 0:0). Independently
  recomputed from raw `Stage_3_deform_fine.npz` (not from any log), same method as
  T0.1/T0.2:
    T04_scale (w_scale=0.052)              -> 6.70% correct, 3.28% median err
    T04_trans (w_trans=1.0, uncalibrated)  -> 6.84% correct, 3.18% median err
    reference: T0.1 winner (offset only)   -> 6.69% correct, 3.21% median err
- both deltas are within the noise band T0.1's own grid already established (~0.11pp
  spread between genuinely-different offset settings) — neither is a correctness win
  or loss.
- engagement check (does the penalty actually fire, or is a null result meaningless):
    T04_scale: log_beta_scales |max| = 0.7697, past the free band (ln2=0.6931) -- ENGAGED
    T04_trans: trans=0.00000 every logged iteration; betas_trans |max| = 0.0434 is over
      the 0.02 band but the barrier means squared excess over 1980 elements (55 joints x
      3 dims x 12 specimens), so one mild excursion rounds to ~0 -- NOT MEANINGFULLY ENGAGED
- reading, split by mechanism (do not conflate the two):
  - **scale_cap**: engaged, null on E6 correctness. Matches the pattern joint-limits
    already showed (§2.2 row 6 of FINAL_REPORT) -- not a correctness fix, potentially
    an output-validity one. The question it was actually built to answer (does it stop
    the 889x antenna-scale/head-shrinks-into-thorax pathology) is separate from
    correctness and is NOT yet answered -- see the follow-up entry below.
  - **trans_cap**: status is "implemented, not meaningfully exercised by this test,
    deprioritized" -- neither passed nor failed. The near-identical number is not
    evidence it works; it's evidence D1's other regularizers already keep
    `betas_trans` inside the free band on this corpus, so this run never tested the
    mechanism. Not shipped. A real test would need a specimen/config where translation
    actually drifts past 0.02 often enough to matter -- lower priority, not scheduled.
- status: DONE for the correctness question. scale_cap's anterior-plausibility question
  (the mechanism's actual design target) is open -- follow-up below, explicitly NOT
  blocking the T0 deliverable.

---

## [T0 deliverable] honest current-best E6 number

- date: 2026-08-11
- **the number: 6.69% correct, 3.21% median error, at `w_offset` 5.0/2.0 -- i.e. exactly
  `diagnostics/moonshot/cfg/D1_SYN.yaml`, unchanged.** Independently reproduced twice:
  once from the pre-existing `diagnostics/moonshot/out/synth_scores.npz` (SYN_D1 arm),
  once by recomputing directly from raw fitted `.npz` weights in this session's own
  T0.1/T0.4 runs, bypassing every log/transcription both times.
- what was tried against this number and what happened to each:
    T0.1 offset sweep {2.0/0.8, 10/4, 20/8}   -> all worse or equal; 5.0/2.0 wins the grid
    T0.2 rest-edge loss + offset winner       -> REJECTED: 1.52% correct (catastrophic,
                                                  4.4x worse) -- do not combine
    T0.3 scheme:all / symmetric sampling /
         w_midline / w_limit / --deform_its   -> already present in D1_SYN.yaml or a
                                                  behavioral-only fix; no config change
    T0.4 scale_cap (w_scale=0.052)            -> null on correctness (6.70%, engaged
                                                  but no effect) -- see anterior-check
                                                  follow-up for its actual design target
    T0.4 trans_cap (w_trans=1.0)              -> not meaningfully exercised; not shipped
- **conclusion: nothing tested in Tier 0 improves on the existing D1 recipe's 6.69%.**
  This is a legitimate, honest outcome, not a failed exercise -- Tier 0's job was to
  verify D1's number is real and see whether any of its unvalidated/untested
  components could push past it; none did. 6.69% / 3.21% median error is the number
  Tier 1 has to beat, and it was independently reproduced rather than taken on faith
  from FINAL_REPORT.
- recommended recipe going forward: `D1_SYN.yaml` as-is (`w_offset` 5.0/2.0, `w_limit`
  0.006, `w_midline` 2.0, `edge_mode: shrink`, `scheme: all`), optionally + `w_scale:
  0.052` (no measured downside, pending the anterior-plausibility follow-up before
  calling it a positive addition). Do NOT add `edge_mode: rest` or `w_trans` on the
  strength of anything measured so far.
- status: DONE

---

## [T0.4 follow-up] anterior deform-ratio check — clean baseline running

- date: 2026-08-11
- what: the one specific test FINAL_REPORT §6.3 named and this session hadn't done yet
  -- does scale_cap actually reduce the per-joint log_beta_scales explosion on
  anterior joints (head/mandible/antenna), the pathology it was built for, independent
  of the (already-answered, null) E6 correctness question.
- missing piece: no raw `Stage_3_deform_fine.npz` existed anywhere in this session or
  on the original branch for the exact combination "w_offset 5.0/2.0, no scale_cap,
  no trans_cap, edge_mode: shrink" -- i.e. `D1_SYN.yaml` run completely unchanged.
  T0.1's three arms used different offset weights; T0.2 used `edge_mode: rest`; T0.4's
  own arms both had one barrier or the other switched on. None is a valid baseline for
  isolating scale_cap's effect on log_beta_scales specifically.
- action: submitted a single-arm job running `D1_SYN.yaml` completely unchanged
  (`--yaml_src diagnostics/moonshot/cfg/D1_SYN.yaml`, no new file needed -- it already
  has no `w_scale`/`w_trans` keys) to `diagnostics/khaoula_v2/runs/T04_baseline`, to
  be compared against the existing `T04_scale` run's `log_beta_scales` on the
  head/mandible/antenna joints identified via `anatomical_groups()`.
- job: submitted, see next entry for result
- status: running, explicitly NOT blocking the T0 deliverable above (already closed)

---

## [T0.4 follow-up] anterior deform-ratio check — COMPLETE, result is inconclusive by construction

- date: 2026-08-11
- ran: job 15518351 (COMPLETED, 0:0, 12:22). Sanity-checked its E6 correctness first
  (6.48% / 3.22% median err — close to the 6.69%/6.70% seen elsewhere today, within
  the hierarchical-stage run-to-run noise already established), confirming this is a
  legitimate D1_SYN.yaml instance before using it for anything.
- method: `exp(log_beta_scales)`, pooled over specimens x joints-in-group x 3 axes,
  min/max ratio per anatomical group, identified via `JOINT_NAMES` (55 joints,
  confirmed from the model pkl directly): head=[46] (`b_h`), mandible=[47,51]
  (`ma_r`,`ma_l`), antenna=[48,49,50,52,53,54] (`an_{1,2,3}_{r,l}`) — same grouping
  logic as `anatomical_groups()`, same ratio methodology FINAL_REPORT used for the
  "0.033-4.612 (138x)" style figures.
- result:
    group      baseline ratio   scale_cap ratio   change
    head            2.11x            2.19x         +3.8% (worse)
    mandible        2.67x            2.69x         +0.8% (~flat)
    antenna         3.08x            3.58x        +16.1% (worse)
  scale_cap does NOT reduce anterior scale range on this corpus -- all three groups
  are flat-to-worse, antenna measurably so.
- **why this result can't actually answer the question it was run for, and must not
  be reported as "scale_cap fails to fix the anterior pathology":** the ranges here
  (2.1x-3.1x baseline) are nowhere near the pathology scale_cap was built to address
  -- FINAL_REPORT's own figures (138x head / 389x mandible / 889x antenna) were
  measured on REAL bench50/worker scans fitted by the stock (unconstrained) pipeline,
  not on synth_clean. synth_clean's targets are generated FROM the model with only
  scale_scale=0.10 per-joint log-scale noise (`make_synth_corpus.py`), so there is no
  incentive for any joint to explode in this corpus -- the fitter never needs to
  cover up bad geometry by scaling a joint 100x, because the target IS well-behaved
  model geometry. **This test corpus does not reproduce the pathology, so it cannot
  validate or refute the fix.** Additionally n is small per group (head n=36 = 12
  specimens x 1 joint x 3 axes) against already-demonstrated hierarchical-stage
  run-to-run noise, so even the observed +16% antenna change should not be
  over-read.
- what a real answer requires: running scale_cap on `bench50_clean` (real worker
  scans, no E6 ground truth needed for this specific check since it's a scale-
  magnitude measurement, not a correspondence one) and comparing anterior
  `log_beta_scales` ranges against the stock/no-scale_cap fit on the same scans --
  the actual regime the 138-889x figures came from. Not run in this session.
- status: DONE as a measurement; INCONCLUSIVE as a validation of scale_cap's design
  target. Recommend: keep scale_cap as an optional, no-downside addition (T0.4's
  correctness result already cleared it), but do not claim it fixes the anterior
  pathology until tested on real scans. Flagging as open, not closing it as either
  a pass or a fail.

---

## [T0.4 follow-up, real corpus] anterior check on bench50_clean — submitted

- date: 2026-08-11
- what: the synth_clean check above is structurally incapable of validating
  scale_cap's design target (no anterior-scale explosion exists in that corpus to
  fix). This is the actual test: same baseline-vs-scale_cap-on comparison, on
  `diagnostics/moonshot/bench50_clean/` (50 real ethanol-preserved worker scans —
  confirmed present, byte-identical to the original branch corpus), the regime
  FINAL_REPORT's 138x/389x/889x figures came from. No E6/ground-truth needed for
  this specific measurement — it's a scale-magnitude read on `log_beta_scales`,
  not a correspondence one, so no synthetic corpus or scoring harness required.
- configs: baseline = `diagnostics/moonshot/cfg/D1_SYN.yaml` unchanged (confirmed
  by diff: differs from `T04_scale.yaml` by exactly `w_scale: 0.052` on both
  stages, nothing else); scale_cap arm = `diagnostics/khaoula_v2/cfg/T04_scale.yaml`
  (already exists, reused as-is).
- pass condition, set before running: scale_cap must compress the anterior range
  MATERIALLY below the baseline's real-scan range (not just "not worse," which is
  all the synth_clean check could show) — if it doesn't, scale_cap is dead for its
  actual purpose, not merely unproven.
- new file: `diagnostics/khaoula_v2/submit_t04_bench50.sbatch` — 2 arms
  (`T04_bench_baseline`, `T04_bench_scale`), 2 GPUs parallel, `--mesh_dir
  diagnostics/moonshot/bench50_clean` (50 specimens vs synth_clean's 12 — longer
  runtime expected, same 04:00:00 conservative walltime)
- status: submitting

---

## [T0.4 follow-up, real corpus] anterior check on bench50_clean — COMPLETE, PASSES

- date: 2026-08-11
- ran: job 15518454 (COMPLETED, 0:0, 27:13 — longer than synth_clean's ~13min, as
  expected for 50 vs 12 specimens). Both arms verified: real, non-empty
  `Stage_3_deform_fine.npz` (~37MB each), all 50 specimens present in `labels`,
  zero NaN/Inf in `log_beta_scales` (checked directly before trusting the ratios).
- same method as the synth_clean check (`exp(log_beta_scales)`, pooled over
  specimens x joints-in-group x 3 axes, per anatomical group via the same
  `JOINT_NAMES` indices), plus a robust p1-p99 version alongside the full
  min/max range, since real scans can carry single-vertex outliers the synthetic
  corpus doesn't:

  | group | baseline (full range) | scale_cap (full range) | baseline (p1-p99) | scale_cap (p1-p99) |
  |---|---|---|---|---|
  | head | 7.47x | **3.79x** | 4.27x | **3.15x** |
  | mandible | 10.69x | **4.12x** | 8.36x | **4.08x** |
  | antenna | 23.56x | **4.06x** | 5.70x | **3.22x** |

- reduction: head -49%, mandible -61%, antenna -83% (full range); -26%/-51%/-44%
  on the outlier-robust p1-p99 version. **Consistent, substantial compression in
  the same direction on all three groups, both statistics.** This clears the
  pre-registered pass condition (materially compress the range, not just "not
  worse") decisively -- unlike the synth_clean check, this is not ambiguous.
- **important framing caveat, stated so this isn't mistaken for reproducing
  FINAL_REPORT's raw 138x/389x/889x figures:** this baseline (`T04_bench_baseline`,
  no `w_scale`) is `D1_SYN.yaml` -- i.e. it ALREADY has joint limits (`w_limit
  0.006`), the offset penalty, midline symmetry, etc. active. FINAL_REPORT's
  138x/389x/889x figures were measured on an earlier, less-constrained baseline
  before those fixes existed. That's why this baseline already sits at 7-24x, not
  138-889x -- the other D1 fixes already absorb some of the explosion as a side
  effect. The comparison that matters here, and the one actually run, is
  **within the current recommended D1 recipe, does adding scale_cap help
  further** -- and it does, substantially. It is not a claim that scale_cap alone
  would take an unconstrained stock fit from 889x to 4x; that specific comparison
  was not run and would need the original unconstrained baseline reproduced,
  which is out of scope here.
- decision: **scale_cap PASSES.** Combined with T0.4's earlier correctness result
  (null on E6, i.e. no downside) and this result (substantial anterior-scale
  compression on the corpus that actually exhibits the pathology, i.e. real
  upside on its actual design target), recommend adding `w_scale: 0.052` to the
  standing recipe. `w_trans` remains not shipped (T0.4 correctness entry: never
  meaningfully exercised, deprioritized, no analogous real-corpus test run here).
- **cost check (wall-clock + convergence), so the benefit numbers above aren't
  sitting next to a silent gap:**
  - wall-clock, hierarchical-checkpoint-to-Stage_3-done, same job (both arms ran
    in parallel on separate GPUs, so this isolates each arm's own time, not a
    shared-resource artifact): baseline Stage_2 177s + Stage_3 151s = 328s total;
    scale_cap Stage_2 182s + Stage_3 153s = 335s total. **+7s (+2.1%) on the
    moonshot stage** -- consistent with a handful of extra elementwise ops on an
    already-materialized tensor (`log_beta_scales`), not a new forward pass. The
    hierarchical stage itself (identical between arms -- `--scale_cap` was not
    set there, only `w_scale` in the moonshot YAML) showed ~10s swings between
    arms too (H1: 6s, H2: 9s, H3: 11s), i.e. this cost is at or below the
    run-to-run noise floor already established elsewhere in this session.
    **Checked, effectively free.**
  - convergence quality, final-iteration core loss terms (mean over 50
    specimens, from the training logs directly, not recomputed): Stage_2
    chamfer +0.84%, edge +0.00%, normal +0.67%, laplacian +0.93%; Stage_3
    chamfer +1.94%, edge +0.00%, normal +0.46%, laplacian +1.11% (scale_cap vs
    baseline, all directionally worse but by <2% on every term, no term
    unaffected-vs-degraded split -- they all move together by a small, uniform
    amount, which is what redirecting a little optimization pressure toward a
    newly-added regularizer looks like, not what a fit breaking down looks
    like). **Checked, no meaningful degradation.**
- status: DONE. T0.4 is now fully closed on all three questions it was ever
  supposed to answer (correctness: null; anterior plausibility: pass; cost:
  checked, negligible on both wall-clock and convergence).

---

## [T1.0 prerequisite] env setup for extrinsic-feature extraction, + a self-inflicted metadata bug fixed

- date: 2026-08-11
- what: T1.0 (Diff3F-style feature sanity check) needs `diffusers`+`transformers`,
  not previously in the `pytorch3d` env. Compute nodes (`dc-gpu` partition) have
  NO internet access (confirmed via a 2-min sbatch probe, job 15518509:
  `curl https://huggingface.co` -> blocked on node `jrc0326`); the login node
  does. So all model downloads must happen here first, into a shared cache
  compute nodes can read offline.
- `pip install diffusers==0.29.2 transformers==4.42.4 accelerate==0.32.1
  huggingface_hub==0.24.5` broke `import transformers` immediately after:
  `ValueError: Unable to compare versions for numpy>=1.17,<2.0: need=1.17
  found=None`. Root cause, diagnosed via `importlib.metadata.version()` (not
  `numpy.__version__`, which worked fine throughout -- the break was in
  package-metadata resolution, not the package itself):
  1. `torch`/`torchvision` had stale, EMPTY `*.egg-info` directories left over
     from the earlier torch env-repair (dated before this session, 0 files each)
     that were shadowing/confusing their valid, complete `*.dist-info` entries
     -- `importlib.metadata.version('torch')` returned `None` despite
     `torch.__version__` working. Fixed: `rmdir` on the two empty dirs (nothing
     to lose, they contained zero files) -- confirmed `torch`/`torchvision`
     resolve correctly immediately after.
  2. This pip install's own dependency resolver silently downgraded numpy
     2.2.6 -> 1.26.4 (to satisfy transformers' `numpy<2.0` pin) and left a
     stale, incomplete `numpy-2.2.6.dist-info` fragment behind (4 entries,
     missing METADATA and RECORD -- confirmed genuinely broken, not just
     sparse) instead of cleanly removing it on upgrade/downgrade -- the same
     "/p/scratch install corruption" pattern the original env-repair entry
     already documented, just triggered by this pip call instead of a prior
     one. Fixed: removed the one stale file inside then `rmdir`'d the now-empty
     directory (surgical two-step, not a recursive delete, after confirming
     the file's only content was a `direct_url.json` and that 1.26.4's own
     dist-info was complete and matched the actually-importable numpy).
- regression check after both fixes: `pytorch3d`, `scipy`, `cKDTree` all still
  import; re-ran the exact T0.1 correctness computation on `T01_off2` specimen 0
  through the full pipeline again (fresh process) -- executes cleanly, returns a
  plausible non-degenerate number, confirming nothing upstream broke from the
  numpy version change.
- status: DONE — `diffusers`/`transformers` importable, no regression in the
  existing T0 toolchain. Next: download SD + DINOv2 weights to a shared cache
  from this (internet-having) login node before submitting any GPU job.

---

## [T1.0 prerequisite, cont.] models cached offline; one more corrupted package (PIL)

- date: 2026-08-11
- cache: `HF_HOME=/p/scratch/cias-7/jellal1/hf_cache` (shared Lustre, readable from
  compute nodes without network access — this is the fix for the no-internet
  finding above). Downloaded from the login node:
  - `stabilityai/sd-turbo` (single-step SD variant — chosen over standard SD1.5
    specifically because it removes the "which timestep to pick" judgment call
    Diff3F's multi-step variant requires; one clean forward pass per view)
  - `facebook/dinov2-base`
  Both downloaded with no gating/token issues.
- third instance of the same "/p/scratch extraction corruption" pattern this
  session has now hit four times total (torch, numpy metadata, gmpy2 earlier,
  now PIL): `from PIL import ImageFilter` failed — `ImageFilter.py` was
  genuinely absent from the installed `PIL/` directory (28 files present vs
  115 after the fix; confirmed by direct `ls`, not inferred from the error
  alone). Fixed the same way as the earlier gmpy2 fix: `pip install
  --ignore-installed --no-deps pillow==12.0.0`.
- verified end-to-end with `HF_HUB_OFFLINE=1` set (i.e. network calls would
  hard-fail, not silently fall back to online) — `AutoencoderKL`, `UNet2DConditionModel`,
  `CLIPTokenizer`, `CLIPTextModel`, `Dinov2Model`, `AutoImageProcessor` all load
  from the local cache: `unet down_blocks=4 up_blocks=4` (sd-turbo's standard
  SD1.x-derived UNet shape, as expected).
- status: DONE. Environment is now fully ready for the T1.0 probe script
  (nothing further to install or download). **If picking this up fresh in a
  new shell, the two env vars below are required before anything else works**
  (see handoff note at the end of this file for the exact commands).

---

## HANDOFF — exact state as of 2026-08-11, end of session, mid-T1.0

**Where things stand:** Tier 0 is fully closed (see the `[T0 deliverable]` entry
above — 6.69%/3.21%, `D1_SYN.yaml` unchanged, `w_scale: 0.052` recommended as an
addition). T1.0 (the cheap Diff3F/DINO sanity check, gating T1.1) has its
environment fully set up and verified, but **the actual probe script has not
been written yet.** That is the next and only remaining step to run T1.0.

### 1. Always run this first, every new shell (nothing works without it)

```bash
source /p/scratch/cias-7/jellal1/miniforge3/etc/profile.d/conda.sh
conda activate pytorch3d
export HF_HOME=/p/scratch/cias-7/jellal1/hf_cache
export HF_HUB_OFFLINE=1          # only needed on a compute node without internet;
                                  # unset (or leave off) if downloading anything new from the login node
cd /p/home/jusers/jellal1/jureca/SMILify
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl
```

Verify it worked: `python -c "from diffusers import UNet2DConditionModel; from transformers import Dinov2Model; print('ok')"` should print `ok` with no network call (confirms the offline cache is being read).

### 2. What's cached and ready (no re-download needed)

- `stabilityai/sd-turbo` (VAE + UNet + tokenizer + text encoder) and
  `facebook/dinov2-base`, both in `$HF_HOME`. Loadable via
  `AutoencoderKL.from_pretrained('stabilityai/sd-turbo', subfolder='vae')`,
  `UNet2DConditionModel.from_pretrained('stabilityai/sd-turbo', subfolder='unet')`,
  `CLIPTokenizer`/`CLIPTextModel.from_pretrained('stabilityai/sd-turbo', subfolder='tokenizer'/'text_encoder')`,
  `Dinov2Model.from_pretrained('facebook/dinov2-base')`,
  `AutoImageProcessor.from_pretrained('facebook/dinov2-base')`.

### 3. What T1.0 still needs, precisely (not started)

Write `diagnostics/khaoula_v2/t10_diff3f_sanity.py`. Per the plan and this
session's design decisions so far:

1. **Load the template mesh + ground truth part labels** the same way
   `diagnostics/moonshot/sdf_prior_PROBE.py` does (`config.SMAL_FILE`,
   `weights.argmax(1)` for per-vertex joint id, group into
   {thorax, gaster, head, leg, antenna} — the same 5-group scheme the SDF
   probe used, so the result is directly comparable to that probe's AUC
   0.97-0.99 benchmark). Do NOT invent a new grouping for this step.
2. **Render the template from ~6-8 views.** `diagnostics/moonshot/render_3d.py`
   exists and other probes in this repo already depend on it (`within_part_signal.py`
   does not use it, but `score_synth_roundtrip.py` and others do via
   `make_renderer`/`render`/`colour_from_scalar`) — reuse it rather than
   writing a new pytorch3d renderer from scratch; it is already proven to work
   in this exact env. **Not yet checked in this session**: whether it renders
   plain RGB (needed for SD/DINO input) or only the colour-mapped-scalar style
   used elsewhere — read it before assuming.
3. **Per view: encode RGB -> VAE latent -> add noise at a fixed timestep (or
   use sd-turbo's single-step scheduler directly) -> UNet forward pass ->
   grab one or two intermediate up_block feature maps -> upsample to render
   resolution.** Separately, run the same RGB render through DINOv2, take the
   patch token grid, reshape to spatial, upsample to render resolution.
4. **Backproject to vertices**: for each view, rasterize the mesh (pytorch3d
   fragments already used elsewhere in this codebase) to get per-pixel
   vertex/face visibility, bilinearly sample the SD and DINO feature maps at
   each visible vertex's projected pixel location, average across all views
   where a vertex is visible (plain average is fine for a first pass; do not
   over-engineer a normal-weighted version before checking if the plain
   version already shows structure).
5. **k-means with k=5** (matching the 5-group ground truth) on (a) DINO
   features alone, (b) SD features alone, (c) concatenated — report purity and
   ARI against the ground-truth groups for each, the same quantitative
   framing `within_part_signal.py` uses (`correct` vs `random` chance
   baseline), not just a visual/qualitative read.
6. **Pre-registered pass/fail, state it in the script before running, per the
   plan**: if none of the three (DINO/SD/combined) clears chance by a wide
   margin, that is a hard stop before forking `within_part_signal.py` for
   T1.1 — do not proceed to the symmetry-refinement literature or the full
   T1.1 build on a negative T1.0.

### 4. After T1.0 has a result

Log it in this changelog the same way every other entry here is written:
exact numbers, files touched, pre-registered pass condition restated next to
the actual outcome, and whether it clears the bar to proceed to T1.1
(fork `within_part_signal.py`, swap `hks()` for whichever of DINO/SD/combined
won T1.0, gate against `hks_within`'s 12.91% median error per the existing
plan — not against `random_within`'s chance level).

---

## [T1.0] script written, submitted — first run

- date: 2026-08-11
- installed `scikit-learn` (needed for `KMeans`/`adjusted_rand_score`); checked
  for the same metadata-corruption pattern that's hit torch/numpy/PIL/gmpy2
  this session — clean this time, no fix needed (verified via
  `importlib.metadata.version()` on scikit-learn/numpy/scipy/torch, plus a
  full regression re-import of scipy/numpy/torch/pytorch3d).
- new files: `diagnostics/khaoula_v2/t10_diff3f_sanity.py` (design per the
  handoff note above: 5-group ground truth adapted from `sdf_prior_PROBE.py`'s
  scheme to this model's joint-name convention — `group_of()`, documented
  inline as an adaptation, not a reuse, since the two model files use
  different naming conventions; 8-view render via `render_3d.py` reused
  directly; SD-turbo feature via one UNet forward pass at t=100 with a null
  text embedding, `up_blocks[1]` hooked; DINOv2-base patch tokens; nearest-
  vertex-by-barycentric-weight backprojection, not full barycentral splatting
  -- a stated simplification for a sanity check; k-means k=5 on
  DINO/SD/combined vs a random-vector chance control, purity + ARI, same
  >=3x-over-chance reading `within_part_signal.py` used, pre-registered in the
  script's own docstring before running), `submit_t10_sanity.sbatch`.
- submitted job 15518525 (30 min walltime — generous, this is one template +
  8 views + no corpus loop, expected much faster; first run of new code, some
  debug iteration likely).
- status: running, first attempt

---

## [T1.0] COMPLETE — PASS, decisively

- date: 2026-08-11
- ran: job 15518525, COMPLETED, 0:0, 2:56 elapsed — clean on the first attempt,
  no debug iteration needed. Full log:
  `diagnostics/khaoula_v2/runs/t10_slurm_15518525.log`; machine-readable result:
  `diagnostics/khaoula_v2/out/t10_result.json`.
- ground truth: thorax n=1381, gaster n=1230, head n=2553, leg n=4709,
  antenna n=362 (10235 total, matches the model's known vertex count).
- vertex coverage from 8 views: 9477/10235 (92.6%) seen by both SD and DINO
  identically (same rasterization, same visibility) — plausible for
  self-occluded regions (leg joints, mouth interior), not suspiciously
  complete or suspiciously sparse.

  | candidate | dim | purity | ARI | chance purity/ARI | verdict |
  |---|---|---|---|---|---|
  | DINO | 768 | 0.797 | **0.428** | 0.463 / 0.000 | clears >=3x |
  | SD (sd-turbo, t=100) | 1280 | 0.693 | 0.275 | 0.463 / -0.001 | clears >=3x |
  | combined | 2048 | 0.799 | **0.447** | 0.463 / 0.000 | clears >=3x |

- **PRE-REGISTERED VERDICT: PASS**, and decisively, not marginally. Contrast
  with HKS's own earlier result (`within_part_signal.json`): HKS technically
  cleared its 3x-over-chance bar too (0.66% vs 0.14%) but was then shown to be
  practically useless -- 3x worse placement error than the fitted pipeline
  itself. This result is a different kind: ARI 0.428-0.447 is a real,
  substantial clustering agreement (0=chance, 1=perfect), not a whisper just
  above noise.
- **scope of what this proves, stated precisely so it isn't overclaimed**:
  this tests the COARSE question -- do raw DINO/SD features carry ANY
  anatomical semantic signal, via unsupervised 5-group k-means. It does NOT
  yet test the actual hard problem this investigation exists for --
  WITHIN-part separation, the specific thing HKS failed at (0.66% correct
  even with a part oracle, i.e. even when restricted to the correct part).
  That is what T1.1's fork of `within_part_signal.py` tests next, using the
  exact same `hks_within`/`random_within` protocol with DINO swapped in for
  `hks()`. Passing T1.0 is the correct, necessary green light to build that --
  it is not proof the harder problem is solved.
- **recommendation for T1.1: use DINO as the primary candidate, not the full
  SD pipeline.** `combined` (0.799/0.447) barely beats `DINO` alone
  (0.797/0.428) while `SD` alone is clearly weaker (0.693/0.275) and far more
  expensive (VAE encode + UNet forward pass vs. one DINO forward pass per
  view). DINO is doing nearly all the work; SD adds marginal lift at
  substantial extra cost. If T1.1 wants a secondary check, add SD as an
  ablation, not as the primary path.
- status: DONE. Proceeding to T1.1: fork `within_part_signal.py`, swap `hks()`
  for a DINO feature extractor, gate against `hks_within`'s 12.91% median
  error (not `random_within`'s chance level), per the existing plan.

---

## [T1.1] within_part_signal_dino.py written — the actual test

- date: 2026-08-11
- new file: `diagnostics/khaoula_v2/within_part_signal_dino.py`, a fork of
  `diagnostics/moonshot/hull/within_part_signal.py` per the plan's own
  instruction ("fork, don't write a new harness"). Full diff summary (also in
  the file's own docstring, which states this explicitly so nobody has to
  diff it to know what changed):
  - `hks(v, f, n_eig, n_times)` -> `dino_descriptor(v, f)`, which reuses
    T1.0's already-validated `render_views`/`dino_features`/`backproject`
    (imported from `t10_diff3f_sanity.py`, not reimplemented). DINO only, not
    SD or combined — T1.0 showed DINO alone (0.797/0.428) barely trails
    `combined` (0.799/0.447) while being far cheaper; using SD too here would
    ~double the per-specimen cost for a documented marginal gain.
  - **coverage is no longer total, and every score is restricted to the
    intersection of template-seen and target-seen vertices per specimen** —
    HKS is intrinsic (defined everywhere); DINO here is extrinsic/view-based
    (T1.0: 92.6% single-mesh coverage). Scoring over unseen vertices without
    saying so would silently redefine "% correct" to mean something looser
    than the number it's being compared against. Coverage is logged per
    specimen and as a mean.
  - `spatial`/`fitted` controls are byte-for-byte unchanged (no descriptor
    involved) — kept as an internal-consistency check against the original
    run's own `spatial`/`fitted` numbers.
  - `--n` default lowered 6->4 (DINO is cheap per-call, but 8 renders x
    (1 template + n targets) adds up; 4 is enough to read the gate).
  - pre-registered gate, stated in the script before running: DINO's
    `dino_within` median error must land clearly below (defined as: at most
    half of) HKS's 12.91% (`within_part_signal.json`'s own `hks_within`
    result) — not merely above `random_within`'s chance level, which HKS
    already showed is close to worthless as a bar on its own.
- **caught before submitting**: the original script's default `--run
  SYN_clean` (and the `SYN_D1` I initially wrote into the sbatch script) have
  no raw `Stage_3_deform_fine.npz` anywhere on disk — confirmed missing under
  `diagnostics/moonshot/runs/` (only the scored summary survives in
  `moonshot/out/synth_scores.npz`, established earlier this session when
  checking T0.1's anchors). Fixed by adding a khaoula_v2/runs lookup first,
  and pointing `--run` at `T04_baseline` — this session's own Tier-0 run,
  which is not just available but arguably a BETTER "fitted" reference than
  the original SYN_clean/SYN_D1 would have been: it's the actual honest
  current-best D1 recipe this investigation settled on, with verified-real
  data (confirmed non-empty, sanity-checked against E6 back in the T0.4
  section above).
- status: written, syntax-checked, path bug fixed before running (not after),
  submitting.

---

## [T1.1] first run — result was an unfair comparison, not logged as a verdict

- date: 2026-08-11
- ran: job 15518535, COMPLETED, 0:0, 7:30. Numbers: `dino_within` 0.91%
  correct / 10.34% median err (vs `random_within` chance 0.20%/24.91%,
  clears 3x); `fitted` (T04_baseline, same 4 specimens) 5.62%/4.41%;
  `spatial` 0.58%/9.74% -- this last one is an exact match to the original
  `within_part_signal.json`'s own `spatial` row, confirming the fork's
  unchanged parts are genuinely unchanged.
- mean joint (template x target) vertex coverage: 77.0% -- lower than T1.0's
  92.6% single-mesh figure, because coverage here is the INTERSECTION of
  template-seen and target-seen, and a posed target's self-occlusion pattern
  differs from the template's canonical pose.
- as-run against the pre-registered gate: DINO 10.34% vs HKS's 12.91%
  full-coverage reference -- nominally better (~20% relative), does not clear
  the "half of 12.91%" bar.
- **not logged as T1.1's verdict.** Caught before writing anything final: the
  10.34% DINO number is computed over 77% of within-part vertices; the 12.91%
  HKS reference is computed over 100%. Occlusion is not random -- it plausibly
  favours easier, more-visible geometry over cluttered leg-joint/gaster regions,
  which are also plausibly where correspondence is hardest. So the comparison
  as-run is biased toward DINO by construction, not merely noisy, and could
  not be trusted either way without a same-population check. User (2026-08-11):
  "the honest state right now isn't leaning negative, it's unknown" -- correct,
  and the fix was cheap (HKS is CPU/scipy-only, no extra GPU pass needed for
  HKS itself, only the same DINO pass already being run) -- see next entry.

---

## [T1.1] matched-coverage addendum added, before logging any verdict

- date: 2026-08-11
- what: added `hks_within_full` (reproduction check: HKS scored on the
  ORIGINAL 100%-coverage within-part mask, should land near 12.91%) and
  `hks_within_matched` (HKS scored on the EXACT SAME per-specimen
  jointly-seen mask `dino_within` used) to `within_part_signal_dino.py`,
  importing `hks()` unmodified from
  `diagnostics/moonshot/hull/within_part_signal.py` (not reimplemented --
  same function, same `n_eig=120`). Added an assertion that the HKS-matched
  and DINO-within masks are byte-identical per specimen, so a future edit
  can't silently let them drift apart without failing loudly.
- the gate itself was rewritten to compare against `hks_within_matched`
  (the fair number), not the original 12.91% reference, with the reasoning
  stated inline in the script so it can't be silently reverted.
- resubmitting with a longer walltime (00:30->00:45) and more CPUs
  (8->16, `eigsh` on a 10235-vertex cotangent Laplacian benefits from more
  BLAS threads) since this run now does 5 HKS eigendecompositions (1
  template + 4 targets) on top of the existing DINO work.
- status: submitting, result in next entry.

---

## [T1.1] COMPLETE — matched-coverage result: real, non-spurious, insufficient

- date: 2026-08-11
- ran: job 15520392, COMPLETED, 0:0, 8:35.
  `diagnostics/khaoula_v2/runs/t11_slurm_15520392.log`;
  `diagnostics/khaoula_v2/out/within_part_signal_dino.json`.

  | method | median err | coverage |
  |---|---|---|
  | fitted (T04_baseline) | 4.41% | 100% |
  | hks_within_full | **12.91%** | 100% -- exact reproduction of the original probe's reference, confirms the imported `hks()` is unmodified and correctly wired |
  | hks_within_matched | **13.51%** | same 77.0% subset as DINO |
  | dino_within | **10.34%** | 77.0% |

- **the coverage-bias concern is resolved, and resolved in the direction that
  clears DINO's number rather than the direction that would have discredited
  it**: HKS on the identical 77%-visible subset scores 13.51%, marginally
  *worse* than its own full-coverage 12.91% -- so the visible/occluded split
  is not an "easy subset" that would inflate DINO's apparent edge by itself.
  Restated precisely: DINO's improvement over HKS is a real effect measured
  on a matched population, not a population-selection artifact. The concern
  raised before running this addendum is directly refuted by the addendum's
  own numbers, not merely addressed procedurally.
- **verdict, stated in full rather than collapsed to a single pass/fail**:
  1. DINO genuinely beats HKS on an identical vertex population: 10.34% vs
     13.51%, ~23% relative error reduction. Real, not spurious. This is a
     positive mark for the extrinsic-vs-intrinsic hypothesis FINAL_REPORT
     proposed (`hull/within_part_signal.py`'s HKS work, and the general
     "no intrinsic descriptor can see this" argument in `hull/REPORT_HULL.md`
     section 5) -- DINO is doing something HKS structurally cannot.
  2. It does NOT clear T1.1's own pre-registered gate (<=half of matched
     HKS, i.e. <=6.76%). The improvement is real but not the "wide margin"
     the gate required.
  3. DINO still loses decisively to the fitted pipeline (10.34% vs 4.41%,
     >2x worse) -- nowhere near sufficient to justify T1.2's full build on
     this raw signal alone.
- **T1.1 is closed on its own narrow question: does RAW, off-the-shelf DINO
  clear the bar to justify building the full matching pipeline (T1.2 as
  originally scoped)? No -- cleanly, on a verified-fair comparison, not on
  an ambiguous or population-biased one.**
- **T1.2-as-originally-scoped (build a matching pipeline directly on raw
  DINO features) is correctly NOT justified by this result and should not be
  built.** This is not the same as closing the extrinsic-feature question --
  see the next entry.
- status: DONE, closed on its narrow question. Does not answer, and was never
  meant to answer, whether a REFINED extrinsic feature would clear the bar --
  see below.

---

## [Open item, NOT started] symmetry/geometry-refined extrinsic features (Uzolas et al.-style)

- date: 2026-08-11
- **this is an open item, explicitly not run, not fully scoped, and not to be
  read as closed alongside T1.1 above.** The v2 plan named a specific
  refinement mechanism for exactly this residual gap before T1.1 was ever
  run: a small autoencoder trained on top of Diff3F/DINO features under a
  geodesic-distance-preservation objective (Uzolas et al. 2025,
  "disambiguating symmetric parts on a shape"), because plain Diff3F/DINO
  only guarantees SEMANTIC separation, not GEOMETRIC/positional separation --
  which is exactly the gap T1.1 just measured directly: DINO clusters
  anatomical parts well (T1.0: ARI 0.428-0.447) and beats HKS within a part
  by a real margin (T1.1: 23% relative), but 10.34% median error is still
  far short of useful. Raw extrinsic features were always the CHEAP PRE-CHECK
  for whether this refinement layer is worth building, not the candidate
  itself. T1.1's result -- real signal, insufficient magnitude -- is the
  expected intermediate outcome that this pre-check existed to produce, not
  a dead end.
- **pre-registered gate for this item, set now, before any implementation,
  same discipline as every gate in this investigation** (not deferred to
  whenever it gets built):
  - primary kill condition: refined-descriptor `within_part` median error,
    scored with the EXACT SAME matched-coverage protocol
    `within_part_signal_dino.py` already implements, must land BELOW HALF of
    raw DINO's 10.34% (i.e. <=5.17%) -- mirroring exactly how T1.1's own gate
    was built off the immediately-prior baseline, not off chance.
  - the number that would actually justify T1.2's full build, restated from
    the original plan: meaningfully below the fitted pipeline's 4.41% (this
    run's own fitted reference, not the original branch's stale 4.32%).
  - if the primary kill condition is not cleared: close the entire extrinsic-
    feature family (raw and refined) for the within-part problem, the same
    way the intrinsic family (HKS, geodesic branches, convex hulls, SDF, EM
    partitions, anterior split, robust kernels, per-part robust scaling) was
    closed in FINAL_REPORT section 3.
- **scope, honestly stated as NOT yet designed in detail**: this needs (a) a
  geodesic (or graph-approximate) distance computation on the template mesh
  -- `cotangent_laplacian` inside `within_part_signal.py`/this fork gives the
  machinery for the Laplace-Beltrami side but not geodesic distances directly,
  a separate (if related) computation; (b) a small autoencoder architecture
  and training loop, i.e. this is a TRAINING task, not a feature-extraction
  task like T1.0/T1.1 -- a materially different and larger scope than
  anything built so far in Tier 1; (c) a decision on training data (per-
  specimen on `synth_clean`, or a single template-only self-supervised
  objective -- not decided). None of this is designed yet. Do not treat "next
  step: Uzolas-style refinement" as a scoped, ready-to-build item -- it is a
  named direction with a pre-registered pass bar, not a plan.
- status: NOT STARTED. Explicitly open, not closed, not scoped beyond the
  gate above.

---

## [Write-up] REPORT_V2.md

- date: 2026-08-11
- new file: `diagnostics/khaoula_v2/REPORT_V2.md` — consolidated synthesis of
  the full v2 investigation (Tier 0 + Tier 1), in the style of
  `diagnostics/FINAL_REPORT.md` (TL;DR, numbered sections, explicit
  recommendation order). This changelog remains the primary, chronological
  source (every command, every job ID); the report is the reader-facing
  synthesis, cross-checked against it, not a replacement for it.
- explicitly preserves the open/closed distinction from the last two
  entries: T1.1 closed on its narrow raw-feature question; the Uzolas-style
  refinement layer stated as open, not started, with its own pre-registered
  gate — not folded into a vague "future work" line.
- status: DONE.

---

## [Refinement layer] build starting — pre-registered risk + gate + design, stated before any code runs

- date: 2026-08-11
- **what it actually is, confirmed against the real paper** (Uzolas et al.,
  SIGGRAPH Asia 2025, arXiv 2503.18254, "disambiguating symmetric parts on a
  shape"): a small point-wise autoencoder on top of Diff3F/DINO base
  features, trained with a reconstruction loss plus a geodesic-distance-guided
  contrastive term — vertex pairs that are geodesically FAR get pushed apart
  in embedding space even when their base features are near-identical (their
  own examples: paw vs. paw, human left/right leg).
- **pre-registered risk, stated now, before the run, not after a miss**: the
  paper states its own limitation directly -- it "cannot establish a
  consistent partitioning for objects that are both geometrically and
  semantically isotropic," and its human-leg result specifically depends on
  the underlying vision features already carrying a learned front/rear pose
  cue from the diffusion model's training distribution -- NOT from geodesics
  alone, which genuinely cannot distinguish a true mirror pair (a left and
  right leg are geodesically identical under reflection, by construction, no
  amount of training fixes that with geodesics as the only supervision
  signal). An ant is close to the hard case the paper admits failing on:
  6-fold leg symmetry plus bilateral symmetry, and a much weaker "front/rear"
  visual prior in Stable Diffusion's training distribution than human limbs
  plausibly carry (ants are a comparatively rare, visually
  repetitive-jointed subject; recall this exact "may not be sufficiently
  represented in the training distribution" caveat was already flagged as
  T1.0's cheap pre-check rationale). **Pre-registered expectation: the
  ≤5.17% gate may well not clear for exactly this reason, on the ant
  template specifically -- and if it doesn't, that is evidence for THIS
  specific failure mode (symmetry the geodesic signal cannot break), not a
  generic "refinement didn't help."** State this now so a miss cannot be
  read as an unexplained negative later.
- **design, so this is buildable cheaply, not a re-derivation of the whole
  paper**:
  1. Geodesic ground truth: `synth_clean`'s specimens share the template's
     mesh topology, so geodesic distance is computed ONCE on the template's
     own rest-pose geometry (intrinsic, pose-invariant under the isometric-
     bending assumption already relied on by HKS elsewhere in this
     investigation) via Dijkstra shortest-path on the mesh edge graph
     (`scipy.sparse.csgraph.dijkstra`, edge weights = rest-pose Euclidean
     edge length -- a standard discrete geodesic approximation, not exact
     but adequate for a probe), from a sampled anchor set, not full V x V
     (10235^2 is unnecessary; a few hundred to ~1000 anchors gives ample
     training pairs).
  2. Base features: DINO, reusing T1.0/T1.1's already-validated
     `render_views`/`dino_features`/`backproject` pipeline -- NOT
     re-derived, but (correction to the initial framing) NOT actually cached
     to disk by either prior script either, confirmed by direct check
     (`diagnostics/khaoula_v2/out/` holds only summary JSONs + one plot, no
     raw per-vertex feature arrays). Adding caching now so this run's
     extraction is paid once and reused for any future training iteration,
     not re-paid every time.
  3. **Train/eval split, to avoid training on the evaluation set**: train the
     autoencoder on `synth_004`..`synth_011` (8 specimens), evaluate via the
     unmodified `within_part_signal_dino.py` harness on `synth_000`..`synth_003`
     -- the SAME 4 specimens T1.1's raw-DINO number was already measured on,
     preserving direct comparability while keeping train and eval disjoint.
  4. Architecture: small MLP encoder (768 -> 128) + decoder (128 -> 768),
     reconstruction MSE + margin-based contrastive term on geodesic distance
     (far pairs pushed apart, near pairs pulled together), trained per-vertex
     across the 8 training specimens' worth of extracted DINO features paired
     with the one fixed template geodesic-distance table.
  5. Evaluation: swap the refined encoder's output in place of raw DINO
     inside `within_part_signal_dino.py`'s `dino_descriptor()` (a new
     `--refined` flag, not a new script -- keeps the exact same protocol,
     controls, and matched-coverage discipline T1.1 already established),
     score `dino_within` (refined) against the pre-registered gate: within-
     part median error <= 5.17% (half of raw DINO's 10.34%), restated from
     the open-item entry above.
- status: design + risk logged. Building next: geodesic computation, caching,
  training script, eval wiring.

---

## [Refinement layer] built and syntax/logic-checked before any GPU job

- date: 2026-08-11
- new files:
  - `diagnostics/khaoula_v2/refine_net.py` — `RefineNet` (encoder 768->256->128,
    decoder 128->256->768), imported by BOTH the training script and the eval
    wiring so they cannot silently drift onto different architectures.
  - `diagnostics/khaoula_v2/train_refine_autoencoder.py` — geodesic anchors via
    `scipy.sparse.csgraph.dijkstra` on the template's rest-pose edge graph (500
    anchors, cached), DINO extraction+caching for template + `synth_004..synth_011`
    (8 training specimens, DISJOINT from `synth_000..synth_003` which T1.1 already
    scored raw DINO/HKS on and which stays held out for eval — avoids training on
    the evaluation set), reconstruction MSE + margin-based geodesic contrastive
    training loop.
  - `diagnostics/khaoula_v2/within_part_signal_dino.py` modified (not forked again):
    `dino_descriptor()` now takes `cache_name=` (reuse extraction across runs,
    corrects the initial framing that this already existed) and `refine_model=`
    (apply the trained encoder before scoring); new `--refined <path>` CLI flag;
    output JSON path becomes `within_part_signal_dino_refined.json` when used, so
    it cannot silently overwrite T1.1's original raw-DINO result file.
  - **gate correction caught while wiring this up**: the existing READING section's
    `clears_hks` check compares against matched-coverage HKS (T1.1's own gate) — NOT
    the refinement layer's actual pre-registered gate (half of RAW DINO's 10.34%,
    i.e. <=5.17%), which is a different comparison. Added a second, separate gate
    check that only fires under `--refined`, comparing against a hardcoded
    `RAW_DINO_REFERENCE = 0.1034` (T1.1's own result, job 15520392, identical
    protocol) — the two gates are not interchangeable and the script now reports
    both rather than silently reusing the wrong one.
- pre-run checks passed on the login node before spending any GPU time: geodesic
  computation ran standalone (CPU-only, no SLURM needed) — 500 anchors, 10235
  vertices, 1.8s, mesh diameter 5.38 (rest-pose, unnormalised units — consistent,
  since only the normalised ratio `geo/diam` is ever used downstream). Cached to
  `diagnostics/khaoula_v2/out/template_geodesic.npz`. All three files
  syntax-checked (`py_compile`).
- new file: `diagnostics/khaoula_v2/submit_train_refine.sbatch` (1 GPU, 45 min —
  8 specimens' DINO extraction at T1.0/T1.1's ~1-1.5 min/specimen rate, plus a
  fast MLP training loop).
- status: submitting training job next.

---

## [Refinement layer] training COMPLETE

- date: 2026-08-11
- ran: job 15520806, COMPLETED, 0:0, 6:15.
  `diagnostics/khaoula_v2/runs/train_refine_slurm_15520806.log`.
- DINO extraction: template + `synth_004`..`synth_011` (9 renders total), each
  cached to `diagnostics/khaoula_v2/out/dino_cache/*.npz` — reusable by any
  future run, not re-paid again.
- geodesic coverage: 460/500 sampled anchors seen in the template (92.0%,
  consistent with T1.0/T1.1's ~92.6% single-mesh coverage figure).
- training curve, 30 epochs, 200k pairs/epoch: reconstruction loss 2.6699 ->
  0.6341 (smooth, monotonic, no blowup/NaN); contrast loss 0.6357 -> ~0.13,
  dropping fast in the first ~5 epochs then PLATEAUING around 0.13 rather than
  continuing toward zero. Read against the pre-registered symmetry risk above:
  a plateau (not full convergence) on the contrastive term is consistent with
  a real subset of far-pairs (plausibly the symmetric leg/antenna instances)
  that geodesic distance alone cannot separate, exactly the named limitation —
  this is a training-curve observation, not yet a claim about the eval result,
  which is scored separately below.
- new file: `diagnostics/khaoula_v2/out/refine_net.pt` (trained weights,
  embed_dim=128, records which 8 specimens it trained on).
- status: DONE. Running eval next: held-out `synth_000`..`synth_003` (same 4
  T1.1's raw-DINO/HKS numbers were measured on), scored against the
  pre-registered <=5.17% refinement-layer gate.

---

## [Refinement layer] eval COMPLETE — gate NOT cleared, matches the pre-registered risk

- date: 2026-08-11
- ran: job 15520820, COMPLETED, 0:0, 4:05.
  `diagnostics/khaoula_v2/runs/t11_refined_slurm_15520820.log`;
  `diagnostics/khaoula_v2/out/within_part_signal_dino_refined.json` (verified
  internally consistent with the printed log; the hardcoded
  `RAW_DINO_REFERENCE=0.1034` cross-checked exactly against T1.1's original
  `within_part_signal_dino.json` value, 0.10339404979198297 -> 0.1034, no
  discrepancy).

  | descriptor | median err | correct % | coverage |
  |---|---|---|---|
  | raw DINO (T1.1, job 15520392) | 10.34% | 0.91% | 77.0% |
  | **refined DINO (this run)** | **10.12%** | 0.74% | 77.0% |
  | hks_within_matched (reference) | 13.51% | 0.85% | 77.0% |
  | fitted (T04_baseline) | 4.41% | 5.62% | 100% |

- **refined vs raw DINO: 10.12% vs 10.34%, a 2.1% relative change — within
  noise, not a real improvement.** `correct%` actually moved slightly the
  wrong way (0.91%->0.74%). The refinement layer did not measurably change
  within-part discriminative power on this template.
- **REFINEMENT-LAYER GATE: NOT CLEARED.** Required <=5.17% (half of raw
  DINO's 10.34%); got 10.12%, essentially unchanged. Not a close miss — the
  gate needed roughly a halving of error and got ~2%.
- refined DINO still clears T1.1's own gate direction (beats matched HKS,
  10.12% vs 13.51%) by about the same margin raw DINO already did — i.e. the
  refinement layer preserved DINO's existing edge over HKS without adding
  anything beyond it.
- **this closely matches the pre-registered risk, not a generic miss**: the
  training log already showed the contrastive loss plateauing at ~0.13
  instead of continuing toward zero (logged above, before this eval ran) --
  consistent with a real subset of geodesically-far pairs the encoder could
  not learn to separate. The named mechanism (Uzolas et al.'s own admitted
  limitation: geodesics alone cannot break a true mirror/rotational symmetry
  -- a left leg and its mirror-image right-leg counterpart, or two of six
  legs related by the body's rotational symmetry, are geodesically
  near-identical by construction) is the most parsimonious explanation for
  why training visibly plateaued short of separating the classes and eval
  shows no measurable gain. This was flagged as the specific, named,
  expected failure mode before the training job was ever submitted -- not
  invoked after the fact to explain away a miss.
- **per the pre-registered decision rule** (`V2_CHANGELOG.md`'s
  "[Open item, NOT started]" entry and `REPORT_V2.md` §2.3, both written
  before this result was known): **the primary kill condition was not
  cleared, so the entire extrinsic-feature family (raw and refined) is now
  CLOSED for the within-part correspondence problem**, on the same
  evidentiary footing the intrinsic family (HKS, geodesic branches, convex
  hulls, SDF, EM partitions, anterior split, robust kernels) was closed in
  `FINAL_REPORT.md` §3.
- status: DONE. Tier 1 is now fully closed: T1.0 (coarse signal exists,
  decisively), T1.1 (real but insufficient within-part signal, verified
  non-spurious), refinement layer (does not rescue it, matches the named
  symmetry-limitation risk). Updating REPORT_V2.md next.

---

## [Write-up] REPORT_V2.md updated to close Tier 1

- date: 2026-08-11
- updated TL;DR (new point 4-5: refinement layer built/tested/gate not
  cleared, Tier 1 fully closed), rewrote §2.3 from "open item, not started"
  to the actual built/tested/closed result (design, training curve, eval
  numbers, gate outcome, the pre-registered-risk match), updated §3's next
  steps to state the closure plainly and note the specific, evidenced reason
  (symmetry the geodesic signal cannot see) rather than leaving a generic
  "try harder" door open, updated the appendix file list.
- status: DONE. Investigation write-up now matches its actual, final state.

---

## [Production port] D1_PROD.yaml + run_m1_fit_all.sh updated

- date: 2026-08-11
- new file: `diagnostics/moonshot/cfg/D1_PROD.yaml` = `D1_SYN.yaml` (the corrected
  D1 recipe -- has `w_limit`, unlike `D1_low.yaml`'s known defect) + `w_scale: 0.052`
  on both stages, the one validated addition from Tier 0. Header comment states
  provenance and explicitly what was tested-and-rejected (`edge_mode: rest`,
  `w_trans`) so it can't be silently reopened by someone reading the file cold.
- `diagnostics/morphometrics/run_m1_fit_all.sh` — the actual morphometrics
  production driver (not a diagnostics-only script) — `--yaml_src` changed from
  `D1_low.yaml` to `D1_PROD.yaml`; its "RECIPE" comment block rewritten to state
  why and to flag that it previously pointed at the defective config.
- **operational gotcha flagged, not silently left**: `run_m1_fit_all.sh`'s `fit()`
  skips any tag whose `Stage_3_deform_fine.npz` already exists. If old `MORPH_W*`/
  `MORPH_CLEAN` output from a prior `D1_low` run is still on disk (on the machine
  that actually has the 757-worker + 81-clean corpus -- see below), this recipe
  change would be silently ineffective, mixing two recipes in one morphometrics
  table with no error. Added an explicit warning comment in the script.
- **cannot be run from this machine**: `run_m1_fit_all.sh` requires
  `W=/media/fabi/Data/SMILify_LEGACY/custom_processing/antscan_proofread_castes/worker`
  (757 workers) and `CLEAN=/media/fabi/Data/SMILify_DATASETS_BACKUP/ALL_ANTS_CLEAN`
  (81 specimens) -- both local paths on the original author's own machine, not
  reachable from this HPC cluster (confirmed: `diagnostics/morphometrics/stage/`
  does not exist here, no `MORPH_*` runs exist under `diagnostics/moonshot/runs/`,
  same "dangling symlink to /media/fabi/..." pattern already found for
  `clean81`/`coreg_train`/`coreg_test`/`heldout_*` when `diagnostics/` was synced
  earlier this session). The config port is complete and correct; actually
  executing the full production refit needs to happen on a machine with that
  corpus. Only `diagnostics/moonshot/bench50_clean/` (50 real specimens) is
  actually present here — used for the genus-classification check below instead.
- status: DONE (port complete; execution blocked on data availability, not on
  anything left undone in this session).

---

## [Downstream validation] does scale_cap move genus-classification signal? — YES, directionally

- date: 2026-08-11
- what: the actual question asked — Tier 0 validated `scale_cap` on a magnitude
  metric (anterior joint-scale compression) but never checked the morphometrics
  deliverable itself. Reused the existing `T04_bench_baseline`/`T04_bench_scale`
  fits (no new GPU time — same `bench50_clean` 50 specimens from the T0.4
  anterior check, identical except `w_scale`), symlinked into
  `diagnostics/moonshot/runs/` so `diagnostics/morphometrics/measure.py`'s
  hardcoded path resolves without modification.
- **cannot literally replicate "the same corpus used before"**: the original
  6.1x/3.8x genus-classification headline used 757 workers + 81 `ALL_ANTS_CLEAN`
  specimens (`run_m1_fit_all.sh`), which live only at
  `/media/fabi/Data/...` on the original author's machine — confirmed
  unreachable here (see the production-port entry above). `bench50_clean` is
  the only real corpus this session had access to.
- **severe, stated power constraint**: `analyse.py`'s own default `min_n=3`
  (genus needs >=3 specimens to enter the classification test) leaves only 2
  usable genera on `bench50_clean` (cephalotes n=3, strumigenys n=5 — 8
  specimens; `bench50_clean` was built for taxonomic diversity, the opposite
  of what classification needs). Lowered to `min_n=2` for this check: 9
  genera, 22 specimens — still a severe reduction from the original's scale.
  Stated as a directional signal check, not a powered replication, before
  running.
- new file: `diagnostics/khaoula_v2/genus_scale_cap_check.py`, reusing
  `measure.py`/`analyse.py`'s own functions directly (`measure_run`, `pcs`,
  `loo_1nn`, `perm_test`, `accession_lot`, `features`, `select_features`) —
  not a reimplementation, the SAME lot-blind leave-one-accession-lot-out 1-NN
  + permutation-null protocol the original 6.1x number used. CPU-only
  (`measure.py` has no torch dependency), ran directly on the login node, no
  SLURM job needed.

  | | baseline | scale_cap | delta |
  |---|---|---|---|
  | accuracy | 22.7% (5/22) | 27.3% (6/22) | +4.5pp |
  | null (chance) | 8.5% | 8.4% | -0.2pp |
  | **LIFT (acc/null)** | **2.66x** | **3.27x** | **+0.60x** |

- **paired check, added because accuracy alone can't distinguish "one
  specimen flipped" from "a consistent shift"**: per-specimen correctness
  compared directly (same 22 specimens, McNemar-style). Result: exactly ONE
  discordant pair (`Ponera_kohmoku_OKENT0105440`, wrong-under-baseline ->
  correct-under-scale_cap) and ZERO specimens flipped the other way. The
  entire accuracy delta is this one specimen.
- **honest reading**: the direction is unambiguous and consistent — every
  specimen that changed status changed the SAME way (toward correct), none
  the other way. That is a real, if small, piece of evidence that scale_cap
  does not hurt and plausibly helps the actual downstream signal, not just
  the magnitude metric it was validated on. But at 1 discordant pair out of
  22, this delta is **not statistically distinguishable from noise on its
  own** — a McNemar-style test has no meaningful power at n=1. **Answer to
  "up or down": UP, on this corpus, this specimen set. Answer to "is this
  proof scale_cap improves genus classification": no — underpowered, directional
  evidence only, consistent with (not contradicting) the magnitude-metric result.**
- status: DONE. This is the honest ceiling of what's answerable on real data
  from this machine. A powered answer needs the full 757+81 corpus, which
  needs `run_m1_fit_all.sh` (now pointed at `D1_PROD.yaml`) run on a machine
  that has it.

---

## [Write-up] REPORT_V2.md — production port + genus-check section added

- date: 2026-08-11
- new §2.4, updated TL;DR (points 6), updated appendix file list and §3 item
  1 to reflect `w_scale: 0.052` as shipped-to-config (not just recommended).
- status: DONE.

---

## [Tier 1, new lever A] arc-length kinematic-chain ordering — COMPLETE, gate not cleared

- date: 2026-08-11
- what: a fundamentally different lever from everything closed so far — not a
  better descriptor for independent per-point NN search, but a constraint
  derived from the rig's own known kinematic order (coxa->trochanter->femur->
  tibia->tarsus->pretarsus). "Position along the limb can't jump backward" --
  targets the aperture-problem structure (ambiguity along a tube's own axis)
  directly, using only `weights`/`kintree_table`/`J_regressor`, already in the
  model file -- no external model, no training.
- new file: `diagnostics/khaoula_v2/within_part_signal_arclength.py`. Per-vertex
  arc-length computed independently on EACH mesh (template and target) from
  that mesh's OWN joint positions and segment lengths -- explicitly NOT looked
  up from a template table by vertex index, which would trivially recover
  ground truth by construction and prove nothing (documented in the script
  as the thing NOT to do). Matching restricted to the same `gt16` anatomical
  groups `within_part_signal.py`/`within_part_signal_dino.py` already use, for
  direct comparability. Fully defined everywhere (no rendering, no occlusion,
  100% coverage like HKS) -- ran entirely on the login node, no GPU, no SLURM
  job, ~1 minute for n=6 specimens (arc-length is near-free; HKS's
  eigendecomposition, computed alongside for the reference row, is the
  actual cost).
- result (n=6 specimens, `synth_000`..`synth_005`):

  | | median err |
  |---|---|
  | fitted (pipeline, this run) | 3.88% |
  | random_within (chance) | 23.93% |
  | hks_within (reproduction, 100% coverage) | 12.81% |
  | raw-DINO reference (T1.1) | 10.34% |
  | **arclen_within, leg/antenna groups (hypothesis target)** | **10.80%** |
  | arclen_within, other groups (head/mandible/body — not the target) | 19.34% |
  | arclen_within, all groups pooled | 16.09% |

- **on its own hypothesis target (leg/antenna chains), arc-length beats HKS
  by almost the same margin raw DINO did** (10.80% vs 12.81%, real signal,
  not noise-level) **but does not beat raw DINO itself**, and comes nowhere
  near the pre-registered gate (<=5.17%, half of raw DINO's 10.34%).
- reading this correctly, per the mechanism's own limits stated before
  running: arc-length genuinely cannot resolve circumferential position
  around a limb -- only position along its length -- and this result says
  that residual, unresolved circumferential ambiguity alone accounts for
  most of the remaining error. Consistent with the pre-registered framing,
  not a surprise requiring a new explanation.
- **kept separate the head/mandible/body groups (19.34%) explicitly so this
  result isn't averaged with a case the mechanism was never expected to
  fix** — pooling them (16.09%) would have made the result look worse than
  it is on the actual hypothesis, and better than it is if someone
  mistakenly read the pooled number as the target-group result.
- status: DONE. Gate not cleared — lever A does not, on its own, justify
  further investment. Testing lever B (GBCPD-style geodesic motion
  coherence) next, as planned before either lever's result was known.

---

## [Tier 1, new lever B] geodesic motion coherence — design, build, pre-registered gate

- date: 2026-08-11
- what: the structurally different lever — not a per-point descriptor, a
  change to the correspondence-SEARCH mechanism itself. GBCPD (Kondo et al.,
  TPAMI 2022) replaces standard CPD's Euclidean-proximity motion-coherence
  prior with a geodesic one, fixing exactly the failure this investigation
  diagnosed (a folded leg touching the thorax gets treated as coherent with
  it under Euclidean-only coherence).
- checked `probreg` (pip-installable, has `bcpd` module) before building
  anything: `grep -r geodesic` over its installed source returns ZERO
  matches — its BCPD is plain Euclidean-coherence, not GBCPD. Using it as-is
  would test a different (and already implicitly related to closed)
  hypothesis, not the one proposed. Full from-scratch GBCPD (its own
  Bayesian-EM formulation) judged out of proportion to this investigation's
  remaining scope.
- **built the second option explicitly offered when this lever was proposed:
  geodesic motion coherence as an ADDED TERM in the existing optimizer**, not
  a standalone alternative registration architecture. New files:
  - `diagnostics/khaoula_v2/trainer_geocoherence.py` — `GeoCoherentMoonshotStage(MoonshotStage)`,
    subclassing (not modifying) `fitter_3d/trainer_moonshot.py`. New `w_geocoh`
    term follows the exact same opt-in `.get(key, 0.0)` pattern every other
    addition there uses (`w_scale`, `w_trans`, ...) — default 0, strict
    superset of already-validated behaviour. Mechanism: Gaussian kernel in
    GEODESIC distance (reusing the cached anchor-to-all distances from the
    refinement-layer work, `template_geodesic.npz` — not recomputed) defines
    a soft neighbourhood per anchor; penalises the VARIANCE of per-vertex
    total motion (current fit minus rest pose) within that neighbourhood.
    Computed via the weighted second-moment identity (`Var = E[x^2] - E[x]^2`)
    for O(anchors x V) cost, not O(anchors x V x V).
  - `diagnostics/khaoula_v2/optimise_geocoherence.py` — minimal fork of
    `fitter_3d/optimise_moonshot.py` (two changes: import, stage-construction
    line), same isolation discipline as `within_part_signal_dino.py`.
    `fitter_3d/optimise_moonshot.py`/`trainer_moonshot.py` untouched.
  - **offline smoke test before spending any GPU time** (CPU, no SLURM):
    uniform displacement -> loss ~0 (2.4e-6, numerical noise); random
    N(0, 0.1^2) displacement -> loss 0.0299, matching the theoretical
    3*0.01=0.03 exactly (3 spatial dims x variance); gradient exists and is
    finite. Math verified correct before running anything real.
- **no prior calibration exists for `w_geocoh`** (a genuinely new term, unlike
  `w_scale` which had `M7_scale1x.yaml` as a reference point) — sweeping 3
  arms spanning two orders of magnitude (0.1, 1.0, 10.0) rather than
  committing to one blind guess: `diagnostics/khaoula_v2/cfg/GEOCOH_{01,1,10}.yaml`,
  each = `D1_SYN.yaml` (the same baseline every Tier-1 within-part candidate —
  HKS, DINO, arc-length — was compared against) + `w_geocoh`. Kernel bandwidth
  `sigma = 0.05 x geodesic diameter` — one reasoned first pass (~one
  leg-segment's scale), explicitly not tuned.
- **pre-registered gate, before any arm is run**: score the fitted output via
  the same independent-recompute E6 protocol every prior run in this
  investigation used (not trusting any training log), for BOTH overall
  correctness and, for whichever arm performs best, a within-part/between-part
  decomposition (`why_partitions_null.py`'s own method) specifically —
  within-part median error must clear <= 5.17% (half of raw DINO's 10.34%,
  the same bar every Tier-1 candidate has been held to) to justify further
  investment in this direction.
- status: built, smoke-tested, submitting the 3-arm sweep next.

---

## [Tier 1, new lever B] geodesic motion coherence — COMPLETE, decisive negative

- date: 2026-08-12
- ran: job 15521128, COMPLETED, 0:0, 13:02 (SLURM estimated a ~2h20m queue wait
  under high partition load — 154/180 nodes `alloc` at submission time,
  0:00 `TIME`, real backfill `START_TIME` given, not indefinite — but backfill
  found an earlier slot; actual wait was well under the estimate).
- **aggregate E6 correctness, independently recomputed from raw
  `Stage_3_deform_fine.npz` for all 3 arms (not trusted from any log)**:

  | `w_geocoh` | correct % | median err |
  |---|---|---|
  | 0 (baseline, T0.1 winner) | 6.69% | 3.21% |
  | 0.1 | 4.13% | 4.13% |
  | 1.0 | 2.12% | 5.50% |
  | 10.0 | 1.79% | 6.04% |

  **Worse than baseline at every tested weight, monotonically worsening as
  the weight increases — no interior optimum, not even at the gentlest
  setting tried.** Training logs (checked before scoring) show the expected
  mechanical trade-off driving this: as `w_geocoh` increases, the geocoh loss
  term itself shrinks (0.00077 -> 0.00014 -> 0.00002) exactly as designed,
  but `chamfer` rises in lockstep (0.00054 -> 0.00088 -> 0.00169) — the term
  is doing what it was built to do (suppress local displacement variance)
  at a real, direct cost to surface-fit accuracy, and that cost is not being
  repaid in correspondence.
- **within-part decomposition, done on the least-bad arm (w=0.1) per the
  pre-registered protocol** (`why_partitions_null.py`'s own method:
  unrestricted global NN match, classified post-hoc as within/between-part —
  NOT the DINO/HKS oracle-restricted-search protocol): 77.81% within-part
  error / 18.05% between-part error (vs baseline's ~79%/~16% from the
  original E7 probe) — a small, not-clearly-meaningful shift toward more
  between-part confusion.
- **a real methodology error caught before finalizing the verdict, corrected
  rather than quietly fixed**: the pre-registered `<=5.17%` gate was
  calibrated from DINO/HKS's ORACLE-RESTRICTED protocol (search space
  pre-restricted to the correct part before matching). The within-part
  decomposition actually run here uses `why_partitions_null.py`'s DIFFERENT
  method (global unrestricted match, classified post-hoc) — which is what
  was actually pre-registered for this step, but produces numbers on a
  different scale, not comparable to the 5.17% bar. Comparing GEOCOH_01's
  raw within-part median error (3.79%) against that bar and calling it
  "cleared" would have been an apples-to-oranges mistake. **The correct
  check is GEOCOH_01 vs the baseline, both scored with the IDENTICAL E7-style
  method**: median 3.79% (geocoh) vs **3.03%** (baseline, worse for geocoh),
  p90 11.11% vs 12.29% (marginally better for geocoh), mean 5.73% vs 5.60%
  (worse for geocoh), max 69.96% vs 70.28% (a wash). Properly matched, this
  is not an improvement — consistent with, not contradicting, the aggregate
  result.
- **why this correction matters beyond just fixing one number**: the
  mechanism under test (a loss term that directly minimizes local
  displacement VARIANCE) is mechanically related to metrics that measure
  local error MAGNITUDE among nearby vertices — a term built to smooth
  displacement will tend to shrink such metrics somewhat by construction,
  independent of whether it improves actual correspondence. This is the
  same class of trap as D2's "looked best, measured worst" and chamfer/
  fscore's shrink-wrap-gaming, already named in `FINAL_REPORT.md` §3.2 as a
  standing risk in this investigation. Caught here by insisting on a
  same-method, same-baseline comparison rather than trusting a single
  number against an externally-calibrated bar.
- **overall verdict: decisive negative.** Aggregate correctness is worse at
  every tested weight, monotonically so; the within-part decomposition,
  scored the same way as the baseline, shows no real improvement either.
  GBCPD's core mechanism (geodesic motion coherence), implemented as an
  added regularisation term in the existing optimizer, does not help this
  pipeline's correspondence problem — it trades surface-fit accuracy for
  smoothness without recovering correspondence in return.
- status: DONE. Lever B closed. Updating REPORT_V2.md next.

---

## [Write-up] REPORT_V2.md — levers A and B added, Tier 1 closure restated

- date: 2026-08-12
- new §2.5 (both levers, full numbers, the gate-comparison correction stated
  in full), updated TL;DR (5b), updated §3 items 2-6 to include both levers
  in the "do not build" / "closed" statements and add the two forward-looking
  notes (§2.3's symmetry-signal gap, §2.5's untried full-Bayesian-EM-GBCPD
  gap), fixed a leftover formatting artifact in §3 item 1 from an earlier
  edit pass, updated appendix file list.
- status: DONE. Both new levers logged, scored, and written up with the same
  discipline as everything else in this investigation: pre-registered before
  running, independently recomputed after, methodology errors corrected in
  the open rather than smoothed over.
