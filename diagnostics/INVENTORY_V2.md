# Inventory — every experiment in the v2 registration/morphometrics investigation

Per the execution plan §3.1: what worked, what failed, what's biased, before anything gets
reused. Primary source for every row: `V2_CHANGELOG.md` (chronological, every command/job ID)
and `khaoula_v2/REPORT_V2.md` (synthesis). `FINAL_REPORT.md` covers the original moonshot-
branch investigation this one builds on.

| ID | Experiment | Corpus | n specimens/genera | Outcome | Status | Bias risk |
|---|---|---|---|---|---|---|
| T0.1 | `w_offset` weight sweep (0.2/0.08, 2/0.8, 5/2, 10/4, 20/8, 200/200) | `synth_clean` | 12 poses | 5.0/2.0 wins the full grid (6.69% correct / 3.21% median err), non-monotonic as predicted | KEEP | synth-only — acceptable as a ceiling test, already the existing D1 default |
| T0.2 | `edge_mode: rest` combined with T0.1's winner | `synth_clean` | 12 poses | REJECTED — correctness collapses 6.69%→1.52%, interacts negatively with the offset penalty | CLOSED | synth-only, but this is a negative result (rejecting an addition), so understating harm is the only risk and none was found |
| T0.3 | bundle: `scheme:all`, symmetric sampling, `w_midline`, `w_limit`, `--deform_its` | code review + `D1_SYN.yaml` | n/a (code audit) | all five already present/correct in the existing recipe; only behavioral change is passing `--deform_its 0` (now `--skip_h3`) explicitly | DONE, no pipeline change needed | none — this was verification, not a scored experiment |
| T0.4a | `scale_cap` (`w_scale=0.052`) — E6 correctness | `synth_clean` | 12 poses | null (6.70%/3.28% vs 6.69%/3.21% baseline, within noise); confirmed engaged (`log_beta_scales` max 0.77) | KEEP (no downside) | synth-only for correctness; see T0.4c for the real-scan magnitude test |
| T0.4b | `trans_cap` (`w_trans=1.0`, uncalibrated) — E6 correctness | `synth_clean` | 12 poses | near-identical to baseline (6.84%/3.18%) but `betas_trans` barrier ≈0 throughout — NOT meaningfully engaged, so this is not evidence the mechanism works | DEPRIORITIZED, not shipped | no prior calibration reference exists anywhere in the branch; near-identical number is not a validation |
| T0.4c | `scale_cap` anterior joint-scale check | `synth_clean` | 12 poses | INCONCLUSIVE BY CONSTRUCTION — synth_clean's own anterior ranges (2.1–3.1x) never approach the 138–889x real-scan pathology; this corpus cannot exhibit the failure mode being tested | INCONCLUSIVE, not a validation either way | synth_clean structurally cannot answer this question — see T0.4d |
| T0.4d | `scale_cap` anterior joint-scale check, real scans | `bench50_clean` | 50 specimens, 37 genera (diversity-built, not classification-built) | PASSES decisively: head −49%, mandible −61%, antenna −83% (full range), −26/−51/−44% (p1–p99 robust) | CONDITIONAL — real magnitude evidence, but see T0.4e for whether it moves the metric that matters | real corpus, single-purpose check (magnitude only), no correctness/classification claim made from this row alone |
| T0.4e | `scale_cap` genus-classification lift | `bench50_clean` (reused T0.4d fits, no new compute) | 22 specimens, 9 genera (`min_n=2`, lowered from default 3) | directional: 2.66x→3.27x lift, but the ENTIRE delta is 1 specimen (`Ponera_kohmoku`) flipping correct, 0 the other way | **NOT DONE at power** — directional evidence only, explicitly not a replication of the 6.1x/3.8x headline | **severe bias risk**: `bench50_clean` was built for taxonomic diversity, not classification power; 1 discordant pair has no statistical power. Must not be cited as "scale_cap improves genus signal" without the §4 full-corpus A/B |
| T1.0 | coarse extrinsic-signal sanity: raw DINOv2/SD-turbo features, k=5 k-means vs ground truth | template only, 8 rendered views | 1 template | PASS decisively: DINO ARI 0.428, SD 0.275, combined 0.447 (chance ARI ≈0) | CLOSED (diagnostic only — answers "does any signal exist", not "is it enough") | template-only, coarse question by design; correctly used only as a gate to proceed to T1.1, not as a standalone result |
| T1.1 | within-part correspondence: DINO vs HKS, matched coverage | `synth_clean` (`T04_baseline` fits) | 4 held-out specimens (`synth_000`–`003`) | real, non-spurious, insufficient: DINO 10.34% vs HKS 13.51% (matched 77% coverage) — ~23% relative improvement, but gate required ≤6.76% (half of matched HKS) and DINO still loses >2x to the fitted pipeline's 4.41% | CLOSED | first run was an unfair comparison (77% vs 100% coverage) — caught before logging any verdict, addendum re-ran with matched coverage; final result not biased by this |
| T1.1-refine | Uzolas-et-al.-style autoencoder refining DINO features against geodesic distance | `synth_clean`, train 8 (`synth_004`–`011`) / eval 4 (`synth_000`–`003`) | 4 held-out specimens | gate NOT cleared: 10.12% vs raw DINO's 10.34% (2% relative, within noise); required ≤5.17% (half of raw DINO) | CLOSED | disjoint train/eval split by design; contrastive-loss plateau observed before eval ran (pre-registered risk, not a post-hoc excuse) |
| Lever A | kinematic-chain arc-length ordering (known-order constraint, not a descriptor) | `synth_clean` | 4 held-out specimens | 10.80% vs HKS's 12.81% (real signal on its own hypothesis target) but short of the ≤5.17% gate and worse than DINO (10.34%) | CLOSED | arc-length computed independently per-mesh from that mesh's own joint positions, not looked up from a template table by index — ruled out trivially recovering ground truth by construction |
| Lever B | GBCPD-style geodesic motion coherence, added as an optimizer term | `synth_clean`, `w_geocoh` ∈ {0.1, 1.0, 10.0} | 12 poses per arm | worse than baseline at every weight, monotonically, no interior optimum | CLOSED | a genuine gate-comparison error (oracle-restricted bar vs. global-unrestricted method) was caught and corrected before finalizing — not smoothed over; corrected same-method comparison confirms the negative |
| M1 morph (baseline) | D1 fits, morphometrics pipeline | 757 workers + 81 `ALL_ANTS_CLEAN` | multi-genus | genus classification 6.1x chance (original), 3.8x (independent replication corpus) | BASELINE | lot confound still open (execution plan §5/§6 Phase C) — not re-litigated here |
| scale_cap morph (full corpus) | full-corpus A/B, `D1_PROD.yaml` vs `D1_PROD_SCALECAP.yaml` | 757 workers + 81 `ALL_ANTS_CLEAN` | 838 specimens fit, 678 in the `min_n=3` genus test | **DONE, 2026-08-13**: genus non-inferior (11.4% vs 11.7% LOO-1NN, within null noise), integrity not worse (<1% shift), anterior pathology tail reduced 58-70% | ADOPTED — promoted into `D1_PROD.yaml` | full write-up: `diagnostics/EXECUTION_PLAN.md` §5. Both A/B arms COMPLETED all 13 chunks each, no failures, ~3h17m/arm |

## Cleaning actions taken (execution plan §3.3)

- `khaoula_v2` closed items (T1.1-refine, Lever A, Lever B, T0.2) are frozen as CLOSED above —
  not reopened without a genuinely new mechanism family, per execution plan §6 Phase D's
  explicit do-not-retry list.
- Canonical baseline/candidate tags, matching the actual config files on disk:
  - `MORPH_BASELINE_D1` = `diagnostics/moonshot/cfg/D1_PROD.yaml` (current M1 recipe, **no**
    `scale_cap` — corrected 2026-08-13, see below).
  - `MORPH_CAND_SCALE_CAP` = `diagnostics/moonshot/cfg/D1_PROD_SCALECAP.yaml` (`D1_PROD` +
    `w_scale: 0.052`).
- **Correction made 2026-08-13, flagged explicitly, not silent**: `D1_PROD.yaml` had
  `w_scale: 0.052` baked in directly (from the 2026-08-11 session), which is exactly the
  "silently merge into shipped D1 before the A/B completes" the execution plan's §2.1 P1
  explicitly prohibits — the underlying validation (T0.4d/T0.4e above) was real but
  underpowered/magnitude-only, not the pre-registered full-corpus A/B. Split back into
  `D1_PROD.yaml` (no scale_cap, the actual default `run_m1_fit_all.sh` now points at) and
  `D1_PROD_SCALECAP.yaml` (the candidate, arm B of the pending A/B). See `SHIPPED_RECIPE.md`
  for the corrected weights table and `V2_CHANGELOG.md`'s original entry for what was there
  before.

## Reusable assets kept (not deleted)

- `diagnostics/khaoula_v2/out/*.json` — machine-readable results for every closed item above.
- `diagnostics/khaoula_v2/refine_net.pt`, `template_geodesic.npz`, `dino_cache/*.npz` — kept
  as a negative-control reference and to avoid re-extraction if the extrinsic-feature family
  is ever revisited under a genuinely new supervision signal (execution plan §6 Phase D notes
  what that would need to be — not a retry of the same geodesic-only approach).
- `diagnostics/khaoula_v2/runs/GEOCOH_*`, `T0*`, `T1*` — raw fitted output for every arm,
  kept for audit.
