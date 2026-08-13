# SMILify Registration/Morphometrics — Execution Plan for Claude Code

**Read this whole document before running anything.** It supersedes prior ad hoc probes. Every phase has a precondition check, a pre-registered decision rule, and a required output set. Do not skip a precondition check to get to the "interesting" part of a phase — a phase that starts on an unmet precondition produces numbers that look like results and aren't.

---

## 0. Goals — do not drift from these

- Build a usable parametric SMIL (OmniAnt) model and registration chain so morphometric measurements are comparable across many specimens.
- **Dense per-vertex correspondence is not the success metric and this is now settled, not provisional.** On the synthetic ceiling test (E6), the best validated recipe reaches ~6.69% exact-correct vertices; ~83% of remaining error is within the correct anatomical part. Six independent mechanism families — geometric partitioning (SDF and six siblings), intrinsic descriptors (HKS), extrinsic descriptors (raw and symmetry-refined DINO/Diff3F), known-order constraints, and coherence-based correspondence architecture (geodesic motion coherence) — were tested against this ceiling and none moved it. Do not re-open dense correspondence as a research question without a genuinely new mechanism family; see §6 Phase D for the explicit do-not-retry list.
- The metric that matters is morphometric signal: genus classification currently sits at ~6.1× chance (lot-blind), replicated at ~3.8× on an independent corpus. This is the number every remaining decision should be judged against.
- Ship the D1 recipe fixes that improve integrity/validity, then judge success on genus signal and measurement stability — not on E6 in isolation.

**Your role, in order:** (1) verify preconditions, (2) port what's already validated into the production path, (3) run one unbiased, pre-registered A/B on the full corpus, (4) conclude from that A/B, (5) only then consider anything new — and anything new must clear the bar in §6 Phase D before it gets built, not just proposed.

---

## 1. How the current production chain works — verify this before touching anything

Production chain for morphometrics is **not** stock `fitter_3d/optimise.py`. Pinned recipe (D1):

```bash
export SMILIFY_SMAL_FILE=3D_model_prep/OmniAnt_25PCs_joint_limited.pkl

python -u -m fitter_3d.optimise_hierarchical --mesh_dir $M \
  --midline 2.0 --beta_prior 0.0 --limit 0.273 --offset 30.0 \
  --results_dir $R/${t}_hier

python -u -m fitter_3d.optimise_moonshot --mesh_dir $M \
  --yaml_src diagnostics/moonshot/cfg/D1_low.yaml \
  --init_from $R/${t}_hier/H2_joint.npz \
  --results_dir $R/$t
```

Driver: `diagnostics/morphometrics/run_m1_fit_all.sh`. Stages: `H0_body → H1_legs → H2_joint` (skeleton placement), then moonshot `scheme: all` deform stages. Defining setting: `w_offset: 5.0 / 2.0` in `D1_low.yaml`. Final artefact: `Stage_3_deform_fine.npz`. Model: `OmniAnt_25PCs_joint_limited.pkl`.

Morphometrics (same directory, run after fits): `measure.py`, `embed.py`, `genus_table.py`, `analyse.py`, `taxonomy.py`, `filter_quality.py`, `run_all.sh` → `REPORT_MORPHOMETRICS.md`.

**Any "shipped" claim must travel with integrity diagnostics**, not chamfer/fscore alone: `diagnostics/moonshot/metrics.py` (`edge_logratio`, `deform_mag`, `tri_quality`, dihedral, folded-face fraction, `midline_dev`), the E6 round-trip protocol, and neighborhood-preservation probes.

**Known, already-flagged defects — confirm current state, do not assume fixed:**
- `D1_low.yaml` may ship joint limits off in the handoff stage while SYN configs use `w_limit: 0.006`.
- `H3_deform` is largely wasted — its npz is not consumed downstream, and `--offset 30` on the hierarchical stage is inert for handoff.
- Stock `master` still has pose freeze (`scheme: deform`), sampling asymmetry, shrink edge loss, and unbounded joint scales — the moonshot path partially fixes these; confirm which fixes have and haven't been merged to `master` since the branch forked.

**Precondition check before proceeding to §2:** run `git log --oneline master..feature/registration_moonshot -- fitter_3d/trainer.py` (or equivalent) to confirm what's diverged since the fork point, and diff `D1_low.yaml` against the current `master` defaults. Report this diff before writing `D1_PROD.yaml` — don't assume the branch's understanding of "stock" still matches current `master`.

---

## 2. What gets shipped

"Ship" means one production recipe plus one evaluation path, documented and runnable, with diagnostics attached — not a PR dump of every probe run in this investigation.

### 2.1 Code/config changeset

| Priority | Item | Where | Action |
|---|---|---|---|
| P0 | Recipe D1 chain as default for morph fits | `run_m1_fit_all.sh` + `D1_low.yaml` | Keep hierarchical → moonshot; pin model env |
| P0 | `w_offset` 5.0 / 2.0 | `D1_low.yaml` | Confirm present in both moonshot stages |
| P0 | `scheme: all` (pose never frozen in deform) | YAML | Confirm; do not revert to `deform` |
| P0 | Symmetric area sampling | already in `trainer_moonshot` | Keep; do not reintroduce vertex-vs-area asymmetry |
| P0 | Midline penalty | hier + YAML | Keep measured weights |
| P0 | Joint limits at measured weights | hier 0.273 / handoff 0.006 | Fix `D1_low.yaml` if handoff limits are off |
| P0 | Beta prior OFF | all | Stays 0.0 |
| P1 | `scale_cap` / `w_scale` | `trainer_moonshot` + production YAML | **Port only after the full-corpus A/B in §4 — do not silently merge into "shipped D1" before that A/B completes and passes its decision rule.** |
| P1 | `edge_mode` | D1 still `shrink` in places | One confirmation run, `rest` vs `shrink`, on full `synth_clean` plus a real-scan sample; if null, ship `shrink` as-is |
| P2 | Skip-logic in `run_m1_fit_all.sh` | `[ -f Stage_3... ] && skip` | Force-refit when the recipe changes — this has already caused one silent no-op in this investigation; treat it as a standing landmine, not a one-time bug |
| — | `geocoh`, `refine_net`, DINO-within as default losses | `khaoula_v2` | **Diagnostics only. Do not ship. Gates failed, closed.** |
| — | SDF-derived part labels | `SDF_batch.py` / new stratification script | **Diagnostic/stratification input only — see §2.3. Not a correspondence mechanism. Do not attempt to revive as a data-term fix; this is closed on evidence, not on lack of effort.** |

### 2.2 What the shipped package looks like on disk

```
diagnostics/moonshot/cfg/D1_PROD.yaml          # single source of truth (fixed D1_low + explicit limits + optional scale_cap flag)
diagnostics/morphometrics/run_m1_fit_all.sh    # points at D1_PROD; no silent skip on recipe change
diagnostics/SHIPPED_RECIPE.md                  # 1 page: command, weights table, model file, what was NOT included and why
diagnostics/moonshot/metrics.py                # integrity suite used for any claim
diagnostics/moonshot/sdf_stratify.py           # new — see 2.3
```

Renders/plots required with any ship package:
- Overlay renders: template vs. fit for 12 stratified specimens (easy / hard / multi-genus).
- Integrity dashboard: `deform_mag`, `edge_logratio`, `midline_dev`, folded-face-fraction histograms, baseline vs. `D1_PROD`.
- Trade-off plot (chamfer vs. integrity), same family as the existing `summary_tradeoff.png`.
- **Do not claim a correspondence improvement unless E6 is re-run under an identical protocol.**

### 2.3 SDF's actual role in the shipped package — diagnostic, not mechanism

SDF-derived part labels are cheap, reproducible (AUC 0.97–0.99 on the template, ≥87.8% G1 agreement on real scans, independently confirmed against source) and anatomically sensible. That work is not wasted — it's just closed as a *correspondence-restriction* mechanism, and open as a *stratification* tool. Concretely, before anything in §3 or §4 is scored:

1. Write `diagnostics/moonshot/sdf_stratify.py`: given a fitted mesh, compute per-vertex SDF (reuse `SDF_batch.py` unchanged) and tag each vertex with its anatomical region.
2. Add region as a **grouping column**, not a filter, to the integrity dashboard (§2.2) and to the failure-case gallery in §4.3 — e.g. is `deform_mag` blowing up specifically on legs vs. gaster, per specimen, per arm.
3. Use it to triage the "top-20 worst `deform_mag`" gallery by anatomical region before manual inspection, so failures get explained by body region rather than eyeballed one mesh at a time.

This is a real, shippable use of the SDF investment. It is explicitly not a data term, not a correspondence mask, and should not reappear in any loss-weight table.

---

## 3. Clean everything — worked / failed / biased, and de-bias before reusing anything

### 3.1 Inventory table — maintain in `diagnostics/INVENTORY_V2.md`

| ID | Experiment | Corpus | n specimens/genera | Outcome | Status | Bias risk |
|---|---|---|---|---|---|---|
| T0.1 | `w_offset` sweep | `synth_clean` | 12 poses | 6.69%/3.21% winner | KEEP | synth-only — acceptable as a ceiling test |
| T0.4 | `scale_cap` | synth + `bench50_clean` | limited real | anterior scale compressed | CONDITIONAL | must pass full-corpus A/B, §4 |
| T1.0 | DINO coarse sanity | synth | small | pass, ARI ~0.43 | CLOSED (diagnostic only) | — |
| T1.1 | within-part DINO vs. HKS | synth 000–003 | real, gate failed | CLOSED | — |
| Refine | Uzolas AE | train 8 / eval 4 | 12 | gate failed | CLOSED | — |
| Lever A | arc-length ordering | synth | small | weak | CLOSED | — |
| Lever B | geocoh 0.1/1/10 | synth | 12 | worse than baseline, monotonically | CLOSED | — |
| M1 morph | D1 fits | 757 workers + CLEAN | multi-genus | genus 6.1×/3.8× | BASELINE | lot confound still open, §5/§6 Phase C |
| scale_cap morph | — | — | — | **NOT DONE** | REQUIRED | this is the actual pending question |

### 3.2 Rules for "potential only on a few species"

Anything that (a) was scored only on `synth_000–003`, `bench50`, or fewer than 3 genera, or (b) is a candidate to change production defaults (`scale_cap`, `edge_mode: rest`, any joint experiment under `scheme: all`) **must be re-run** with:
- Full `synth_clean` (every pose in `ground_truth.npz`) for E6 and integrity.
- The full morphometrics corpora — all staged worker chunks plus `ALL_ANTS_CLEAN`, under one recipe (no recipe × corpus confound).
- Genus evaluation via the same `genus_table.py` / lot-blind protocol as `REPORT_MORPHOMETRICS.md`.
- Pre-registered primary metric: genus accuracy/enrichment vs. chance. Secondary: integrity metrics. Tertiary: E6, if a synthetic analogue exists.

### 3.3 Cleaning actions

- Freeze `khaoula_v2` closed items in the changelog as CLOSED — do not reopen without a genuinely new mechanism, per §6 Phase D.
- Move failed weights/checkpoints to `diagnostics/khaoula_v2/archive_closed/`.
- Keep only reusable assets: geodesic cache, DINO cache (disk permitting), `refine_net.pt` as a negative control, `geocoh` run directories for audit.
- One canonical baseline tag: `MORPH_BASELINE_D1` = current M1 recipe without `scale_cap`.
- One candidate tag: `MORPH_CAND_SCALE_CAP` = D1 + validated `w_scale`.
- Delete or quarantine any script that overwrites `within_part_signal_dino.json` without a date/version suffix — this class of bug (silently overwriting the one file a comparison depends on) has already caused one near-miss in this investigation.

---

## 4. Unbiased re-run protocol — this is the one experiment that actually answers whether any of this mattered

### 4.0 Precondition — check this first, it is currently unmet

**This entire phase requires the 757-worker + 81-`ALL_ANTS_CLEAN` corpus at `/media/fabi/Data/...`, confirmed unreachable from this environment as of the last check.** Before doing anything else in §4:
1. Verify current access. If still unreachable, stop and produce a precise, minimal handoff runbook (exact commands, exact config, exact expected outputs) for whoever has access — do not attempt to substitute `bench50_clean` or any smaller corpus and call it the §4 result. A result from the wrong corpus is not a smaller version of this experiment, it's a different, already-run, already-known-to-be-underpowered experiment (see the 22-specimen `scale_cap` result: one specimen flip, no statistical power).
2. Once access is confirmed, run the power check in §4.4a **before** the fits, not after — this investigation has already produced one result whose only finding was "underpowered," and that should be predictable in advance, not discovered again.

### 4.1 Experiment schema — `diagnostics/morphometrics/ab_scale_cap/SCHEMA.md`

```yaml
name: AB_scale_cap_vs_D1
primary_endpoint: genus classification enrichment vs chance (lot-blind), same protocol as REPORT_MORPHOMETRICS
secondary: integrity metrics (deform_mag, edge_logratio, midline_dev, folded_face_frac), stratified by SDF anatomical region (2.3)
tertiary: E6 correct% / median_err on full synth_clean only
arms:
  A: D1_PROD without scale_cap
  B: D1_PROD with scale_cap (w_scale = 0.052, or whatever value T0.4 locked)
corpora:
  - all worker chunks (757)
  - ALL_ANTS_CLEAN (81)
model: OmniAnt_25PCs_joint_limited.pkl
fit_chain: optimise_hierarchical -> optimise_moonshot
force_refit: true   # disable skip-if-exists — see 2.1 P2
seeds: N/A for deterministic optimization; record torch/CUDA versions regardless
exclusion: none a priori; quality gate is a column, not a filter (per existing M1 design)
```

### 4.2 Required diagnostic outputs

| Output | Path pattern | Content |
|---|---|---|
| Fit npz | `runs/MORPH_{A|B}_*/Stage_3_deform_fine.npz` | verts, labels, params |
| Integrity CSV | `ab_scale_cap/integrity_{A|B}.csv` | per-specimen metrics from `metrics.py`, **plus SDF region column (2.3)** |
| Genus table | `ab_scale_cap/genus_table_{A|B}.csv` | same columns as the existing genus table |
| E6 summary | `ab_scale_cap/e6_synth_clean.json` | correct%, median, p90, within/between split |
| Config snapshot | `ab_scale_cap/D1_PROD_A.yaml`, `_B.yaml` | exact weights |
| Run log | SLURM logs + git commit hash | reproducibility |

### 4.3 Required renders/plots

- Fit-quality gallery: 4 views × 20 stratified specimens × arm A/B, side by side.
- Anterior joint-scale distribution: histogram/violin, A vs. B, on the full corpus (the effect seen on `bench50_clean` re-measured at real power).
- Genus confusion matrices, A vs. B, plus an enrichment bar (chance / A / B).
- Integrity-vs-chamfer scatter, colored by arm — this is the specific plot that would have caught the "looks better, measures worse" gate mismatch flagged earlier in this investigation, before it became a written result.
- Lot-confound check: genus signal within-lot vs. across-lot (the open item from §6 Phase C).
- Failure cases: top-20 worst `deform_mag`, with mesh overlays, **stratified by SDF anatomical region (2.3)**.

### 4.4 Pre-registered decision rule — write this before running, not after

Ship `scale_cap` into the default recipe only if:
- Primary metric (genus enrichment) is non-inferior to baseline, **and**
- Integrity is not significantly worse on the pre-specified tests, **and**
- Anterior scale pathology is measurably reduced on the full corpus (not just `bench50_clean`).

If genus drops: keep D1 without `scale_cap`; document the magnitude-vs-taxonomy mismatch explicitly rather than treating it as a wash.
If genus rises: adopt B as `D1_PROD`; update `SHIPPED_RECIPE.md`.

### 4.4a Power check — do this before running, using the real corpus size

Once corpus access is confirmed, compute the minimum detectable effect for the lot-blind classification test at the actual number of genera/lots in the 757+81 corpus, before running either arm. State the number explicitly in `SCHEMA.md` alongside the decision rule. If the full corpus still can't reliably detect an effect the size of what `bench50_clean` suggested (one specimen flip out of 22), say so before the run, not as a post-hoc excuse if the result comes back flat.

### 4.5 Landmark-restricted scoring — cheap addition, answers a real open question

Near-zero marginal cost given the existing harness and existing landmark annotations: restrict `within_part_signal.py`'s scoring to only the specific vertices morphometric measurements actually depend on (leg tips, joint centers, antenna tips), instead of all ~10,000 vertices. This is a materially different, smaller question than the dense-correspondence problem that's now closed — it's not obviously subject to the same ceiling, since tips and joints are exactly the kind of locally distinctive point the investigation's own data (leg-proximal region scoring best of all parts) suggests should be easier. Run this once, on `synth_clean`, before finalizing which measurements are safe to keep as-is vs. which need auditing in §5.

---

## 5. Conclusion block — filled in 2026-08-13 after the §4 A/B ran on the full corpus

```
Corpus access: confirmed (757 worker_ALT + 81 ALL_ANTS_CLEAN, synced via rclone from the
  shared UM6P_2026 Google Drive, 838 specimens, 0 sync errors)
Power check (4.4a): min detectable lift at full corpus ~1.28-1.33x (vs bench50_clean's 2.80x);
  observed delta (below) is a wash, within this design's detection floor either way
Genus enrichment (PRIMARY, lot-blind LOO-1NN): A (baseline) 11.7%, LIFT 5.4x vs A (scale_cap)
  11.4%, LIFT 5.3x — 0.3pp gap, smaller than the null's own +-0.7% spread. NON-INFERIOR, not a
  drop. (Subfamily-level and specimen-level LOO tell the same story; cross-corpus transfer
  moved in scale_cap's favour, 4.3%->6.5%, but neither arm reached significance there.)
Integrity (SECONDARY, full real corpus): all metrics.py terms moved <1% (edge_logratio -0.3%,
  deform_mag +0.3-0.6%, folded_face -0.7%, tri_quality +0.2%). NOT WORSE.
Anterior joint-scale pathology (THIRD CONDITION, full real corpus, 838 specimens): mean ratio
  down 4.9% (head) / 12.0% (mandible) / 14.2% (antenna); the outlier TAIL scale_cap was built
  for drops hard — max ratio head 8.91x->3.77x (-58%), mandible 10.56x->4.09x (-61%), antenna
  13.57x->4.08x (-70%). MEASURABLY REDUCED, replicating bench50_clean's original 22-specimen
  directional finding at full power.
Landmark-restricted correspondence (4.5): run in parallel, see its own out/ JSON when the
  landmark_scoring job lands — not a gate for this decision, informational only.

TWO SEPARATE CONCLUSIONS, per the decision rule's own conjunctive design (genus alone cannot
answer either — it is a wash, and the rule exists precisely so a wash does not get treated as
a de facto reject):

(1) Does scale_cap ship as the DEFAULT D1_PROD recipe? — ADOPT. All three §4.4 conditions
    (non-inferior genus, integrity not worse, pathology measurably reduced) are satisfied.
    PROMOTED into diagnostics/moonshot/cfg/D1_PROD.yaml 2026-08-13 (w_scale: 0.052, both
    stages); D1_PROD_SCALECAP.yaml kept only as the audit-trail record of arm B's config, no
    longer a distinct candidate.

(2) Does scale_cap earn keeping as an optional flag for anterior mesh-validity independent of
    genus classification? — YES, independently confirmed. This was always scale_cap's actual
    design target (T0.4, the 889x anterior-scale-explosion pathology), and the full-corpus
    pathology numbers above replicate the original bench50_clean signal at ~38x the specimen
    count. This conclusion holds regardless of (1) — even if genus had gone the other way,
    the mesh-validity case would stand on its own.

Since both conclusions point the same direction here, decision (1) already subsumes (2) in
practice — but they were evaluated as separate questions per the plan's own instruction, and
would not necessarily have agreed (e.g. if genus integrity had dropped, (1) would reject while
(2) could still hold as a non-default flag).

Decision: ADOPT scale_cap for production (both as default and as the mesh-validity mechanism)
Closed research lines: geometric partitioning (SDF + siblings), intrinsic descriptors, extrinsic descriptors (raw + refined), known-order constraints, coherence-based correspondence architecture
SDF's shipped role: diagnostic stratification only (2.3) — not reopened as a mechanism
Informational only (jobs 15526016, 15526370 — logged 2026-08-13, NOT part of §4.4's gate,
  does not reopen the ADOPT decision above):
  E6 on full synth_clean, final D1_PROD (w/ vs w/o scale_cap): correct 6.65% (A, no scale_cap)
    vs 6.50% (B, scale_cap), median err 3.28% both arms — flat, consistent with T0.4's original
    "null on E6 correctness" finding (scale_cap was never a correspondence fix).
  Landmark-restricted scoring (execution plan §4.5), synth_clean, fitted pipeline vs dense
    whole-mesh baseline (6.5-6.65% correct / 3.28% median, from the E6 run above):
      joint centers (53 verts):        3.77% correct / 3.65% median err
      leg-tip + antenna-tip (14 verts): 3.57% correct / 2.92% median err
    Lower exact-match% than the dense population (small-sample noise on 14-53 vertices vs
    ~10,235) but lower median placement error on leg/antenna tips specifically — mixed,
    inconclusive on its own, not a basis for any action per this task's own framing.

PROVENANCE (so this decision is reproducible, not just asserted):
  commit: 0f441b9cab541cc34dc486c9bf4b7ca33e628be5 (branch feature/investigation)
  primary A/B fit jobs: 15525281 (arm A, ab_a_baseline, COMPLETED 03:17:48) /
    15525282 (arm B, ab_b_scalecap, COMPLETED 03:16:52) — both 13/13 chunks, 0 failures
  genus classification: diagnostics/morphometrics/ab_scale_cap/analyse_A.log,
    analyse_B.log (+ out_A/analysis.json, out_B/analysis.json)
  integrity + anterior-scale pathology: diagnostics/morphometrics/ab_scale_cap/
    integrity_and_anterior_check.py -> integrity_anterior_full_corpus.json
  informational-only, non-gating jobs: 15526016 (ab_e6_tertiary), 15526019 (landmark_scoring)
```

---

## 6. Next-action proposals, ordered by actual project goals

### Phase A — Production hygiene (do first; no new science)
1. Fix the `D1_low.yaml` joint-limit inconsistency vs. SYN configs.
2. Document `H3` inertness; add an optional `--skip_h3` flag rather than leaving it silently wasteful.
3. Fix skip-if-exists so it doesn't silently no-op when the recipe changes (§2.1 P2).
4. Write `D1_PROD.yaml` + `SHIPPED_RECIPE.md`.
5. Reconcile against current `master` (§1 precondition check) before any long fit run.
6. Implement `sdf_stratify.py` (§2.3) and wire it into the integrity dashboard.

**Diagnostics required:** config diff table, one smoke fit on 3 meshes, integrity CSV including the new SDF region column.

### Phase B — Unbiased impact of the one real positive (`scale_cap`)
Full A/B per §4, on the full corpus, once access is confirmed. **This is the only experiment that answers whether Tier 0 mattered for the actual project goal.** Nothing in Phase C or D should start before this either completes or is formally blocked-and-handed-off.

### Phase C — Remaining production fixes
1. Publish the `w_offset` sweep curve on full `synth_clean` (already run at smaller scale; confirm it holds at full corpus).
2. Penetration × `scheme: all` joint experiment — cheap, pre-register "shallower but not less frequent" vs. "resolved via joint rotation" as the two candidate outcomes before running.
3. Validate the `log_beta_scales`/`betas_trans` barriers on the full corpus, not only `bench50_clean` — the same logic that made the `scale_cap` bench50 result provisional applies here.
4. Morphometrics data issues: absolute scale recovery and the lot confound (§5) — these are metadata/collection problems, not another correspondence loss to chase.

### Phase D — Explicitly do not prioritize until A–C are done, and do not reopen without a new mechanism

| Idea | Why it waits / why it's closed |
|---|---|
| Full Bayesian-EM GBCPD reimplementation | Different algorithm from the coherence-term bolt-on already tested; the bolt-on failed monotonically with no interior optimum. A full reimplementation is a genuinely different mechanism and is the one legitimate remaining gap — but only worth it if Phase B shows morphometrics is still limb-swap-limited after the shipped fixes. Not before. |
| More DINO/Diff3F/refinement variants | Gates failed twice (raw and symmetry-refined); the failure mode (true bilateral/radial symmetry, geodesics can't see it) is diagnosed and documented, not a tuning gap. |
| `geocoh` weights, any value | Monotonically worse than baseline across three orders of magnitude tested; there is no interior optimum to search for. |
| SDF or any geometric partitioning as a correspondence mechanism | Structurally capped at the within-part ceiling regardless of label quality — closed on the same evidence as every partition sibling, independently verified against source. Its only remaining role is §2.3. |
| Dense correspondence as the project KPI | Explicitly abandoned as primary in `FINAL_REPORT.md`'s own TL;DR; nothing since has overturned that. |

### Phase E — Only if genus signal is confirmed stable on the full corpus and the recipe is frozen
1. Scale up morphometrics reporting — more genera, ecology correlates, using full Antscan coverage.
2. Keep quality gate as a column, not a hard filter, per the existing M1 design.
3. Consider weak landmarks/part labels **only** if a specific measurement class fails an integrity audit — and if so, use the SDF stratification and §4.5 landmark-restricted scoring already built, rather than opening a new investigation.
