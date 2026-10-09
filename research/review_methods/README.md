# Review methods programme (started 2026-10-06)

**Paper question (revised 2026-10-07, REVIEW2_RESPONSE.md).** Why does anatomical correspondence
degrade between a learned correspondence signal and the final fitted model under real articulation,
and what information or constraint is required to preserve identity during optimisation?

Diagnosis first; the diagnosis decides which interventions get built. Governing rules are in
`PROTOCOL.md` (frozen before any new method touches real data). Evidence that may enter the paper
is listed in `EVIDENCE_LEDGER.md`; invalidated results are excluded there, not argued with.

## Roles of the data

- **Synthetic degraded corpus** = the controlled laboratory: mechanism diagnosis, oracle
  information budget, morphology / pose / partiality / noise / fragmentation shifts.
- **Real JAB (n = 11, expert joints)** = the phenomenon and the external check. It is never a
  development set. Its use is fixed in `PROTOCOL.md` §3.

## Question structure and status

| Q | question | folder | status |
|---|---|---|---|
| P0 | Real-scan **fitted-pose library** (not ground truth): 657 worker scans, all JAB genera excluded, genus-disjoint pose_train / pose_eval | `P0_real_pose_corpus/` | running |
| 3c | Does anatomical identity erode during optimisation while geometry improves? (all five stages, synthetic true labels + real proxy) | `Q3c_identity_trajectory/` | running |
| 0 | Phase I: reproduce corpus (+ joints, params), freeze evaluation, consolidate evidence | `common/`, `PROTOCOL.md`, `EVIDENCE_LEDGER.md` | in progress |
| 1 | Is the problem actually correspondence? (existing valid evidence only) | `EVIDENCE_LEDGER.md` §Q1 | in progress |
| 2 | What information is missing? Oracle budget on real-articulation synthetic | `Q2_oracle_budget/` | pilot queued; confirmatory after P0; **decides Q4** |
| 3 | Why does the real system fail differently across specimens? JAB case study -> hypotheses -> controlled synthetic shift | `Q3a_cse_transfer_case_study/` (done, RESULTS.md), `Q3b_controlled_shift/` (pilot done; confirmatory on P0 pose_eval) | in progress |
| 4 | Interventions, each tied to a mechanism Q2/Q3 identified | see below | **gated on Q2/Q3** |
| B | Classical baselines CPD / NICP, fixed tuning protocol | `M08_classical_baselines/` | planned |

Q4 candidates (built only if Q2/Q3 point at their mechanism):

| candidate | mechanism it targets | folder |
|---|---|---|
| skeleton-conditioned correspondence (surface / predicted / oracle skeleton / anatomical coords) | missing articulation information at matching time | `M01_skeleton_conditioned/` |
| articulation canonicalisation (raw / predicted unposed / oracle unposed) | articulation as nuisance variation | `M05_canonicalization/` |
| iterative correspondence-registration refinement | correspondence and pose errors are coupled | `M04_corr_fit_loop/` |
| uncertainty-aware correspondence in the fitting objective | hard identity forced on ambiguous points | `M02_soft_uncertainty_fitter/` |

Not committed (only if Q4 leaves a specific open question): cross-attention (`M03_cross_attention/`),
curriculum and uncertainty head as separate arms (`M06_curriculum_uncertainty/`).
Cleanup: deformation dose curve (`M07_deformation_dose/`). Ideas not re-run, with the evidence that
closes each: `M09_not_rerun/` (a note, not an experiment).

## Running record

`LAB_NOTEBOOK.md` (every idea and finding, dated), `RULES_AUDIT.md` (inherited rules checked
against the literature), `DEVIATIONS.md` (every protocol change, dated, with what had been seen).

## Folder contract

Every experiment folder holds `PREREGISTRATION.md` (bar fixed before the run), code, `*.sbatch`
(account `rwth2151`, partition `c25g`), `out/` (outputs and probes, kept), and `RESULTS.md`.
Large artefacts live under `/hpcwork/nao48500/review_methods/<folder>/`.
