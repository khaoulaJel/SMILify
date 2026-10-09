# Pre-registration — M3b: real-corpus extraction transfer audit

**Written and committed BEFORE any result is inspected.** Bars fixed here are not revised
after seeing a number.

Date: 2026-09-09. Folder: `diagnostics/morphometric_validation/`.

---

## 1. What this is, and — emphatically — what it is not

The phenotype panel is **frozen** (`PHENOTYPE_PANEL.md`, M3, 2026-09-09): HW + HL primary, no
secondary, WL/TBL covariates, SL conditional, FL/ML/GL/PetL excluded. That panel was selected on
the conjunction of measurement accuracy × reproducibility × biological validity, and it was
validated on the 20-specimen *Atta* set (B1–B4) and on in-model synthetic data (M2).

This audit asks **one** question:

> Does the extraction pipeline that produced those validated traits behave sanely when pointed at
> the heterogeneous 172-genus real corpus, rather than at 20 conspecific *Atta* workers?

It is a **transfer audit**. Its only two possible consequences are:

- **PASS** → M4 proceeds on the frozen panel, unchanged.
- **FAIL** → an *extraction bug* is fixed, the audit is re-run, and M4 proceeds on the frozen
  panel, unchanged.

**This audit may NOT redefine the phenotype panel.** It cannot promote an excluded trait, demote a
primary one, or add a trait. If a trait outside the panel looks well-behaved here, that is not
evidence — the panel's selection axes (pose invariance, expert-landmark adjudication, allometric
validity) are not measured by this audit at all. Recording that temptation here, in advance, is
the mechanism that stops it.

## 2. Sample — fixed, and fixed before fitting

40 specimens drawn from the 757-specimen `worker_ALT` corpus by
`random.Random(20260909).shuffle` then first 40, sorted. Seed and the resulting specimen list are
committed at `audit40_specimens.txt` **before** the fit was submitted. 35 of 172 genera are
represented. Unstratified by design: the question is how the pipeline behaves on a typical draw,
and stratifying would over-represent the taxonomic extremes the audit is not powered to speak about.

n=40 is a screen, not an estimator. It detects a failure mode present in ≳7% of specimens with
~95% probability; it says nothing useful about one present in <2%.

## 3. Provenance note — these fits did not previously exist on this cluster

`run_m1_fit_all.sh`'s 757-corpus run was executed on JUWELS
(`/p/scratch/cias-7/jellal1/...`) and the resulting `Stage_3_deform_fine.npz` files were never
synced back to RWTH. Only the meshes are present here. This audit therefore **fits its own 40
specimens** under the unmodified recipe (`D1_PROD.yaml`, `HIER_FLAGS` as in `run_m1_fit_all.sh`,
`SMILIFY_SMAL_FILE=OmniAnt_25PCs_joint_limited.pkl`), via
`submit_audit40.sbatch`, which calls `run_m1_fit_all.sh` with overridden paths and **no recipe
edits**. Consequence to state plainly: M4 will also require a fresh 757-specimen fitting run
(~80 GPU-minutes by extrapolation), which was not in the earlier plan.

## 4. Checks and their pass bars — all fixed now

Each check is scored on the 40 specimens. `WL` (Weber's length) is the normaliser throughout, so
all shape checks are **dimensionless ratios**, which is what survives `load_meshes`'
per-specimen normalisation (the frame bug that invalidated the first B3 pass — see
`project-morphometric-validation-framework`). No physical-unit claim is made anywhere in this
audit, because no `BL_mm` reference exists for `worker_ALT`.

| # | Check | Bar to PASS |
|---|---|---|
| C1 | **Extraction completeness.** HW, HL, WL finite and strictly positive for every specimen. | 40/40 |
| C2 | **Ratio plausibility.** HW/WL and HL/WL within [0.15, 2.0] — a deliberately wide box that no real ant violates but a collapsed or exploded head does. | ≥ 38/40 (≤2 flagged) |
| C3 | **Anterior–posterior ordering.** Along the fitted body axis, `clypeal_margin_ant_mid` anterior to `cephalic_margin_post_mid` anterior to `petiole_ant_mid` anterior to `gaster_apex_mid`. This is the check that can actually fail: landmark template indices are fixed, so a violation means the *fit* has folded the specimen, not that the landmark is mis-indexed. | ≥ 36/40 |
| C4 | **HW reporter bilateral sanity.** The `b_h_l`/`b_h_r` reporter offsets straddle the midline with \|left\|/\|right\| ratio in [0.5, 2.0]. | ≥ 38/40 |
| C5 | **Distributional transfer.** Corpus HW/WL and HL/WL are unimodal — no separated second mode holding a degenerate value. Scored by visual inspection of the histogram **plus** a dip-test-free rule: no gap ≥ 20% of the observed range separating ≥10% of specimens from the rest. | no such gap |
| C6 | **Failure attribution.** Every specimen flagged by C1–C4 is inspected by rendering its fitted mesh, and classified as (a) obvious scan/mesh pathology, (b) fit failure, or (c) extraction bug. | reported, not gated |

**Overall PASS** = C1–C5 all meet their bars. **FAIL** = any one misses, and the C6 attribution
then determines whether an extraction fix is required before M4.

C6 is the check with real teeth and it is deliberately **not** gated on a number: the user's
instruction was to *manually inspect*, and an automated flag count would substitute for that. Its
output is renders plus a written classification per flagged specimen.

## 5. What this audit cannot establish

- Nothing about accuracy. There is no ground truth for `worker_ALT` — no expert landmarks, no
  physical measurements, no mass. Every check above is an internal-consistency or plausibility
  check. A trait can pass all six and still be biologically wrong.
- Nothing about pose invariance (B2) or allometric validity (B3) on this corpus. B3 in
  particular is **unavailable** here: it needs a physical size reference to restore scale, and
  regressing normalised traits against normalised WL would reproduce exactly the frame artefact
  already withdrawn once.
- Nothing that licenses a panel change. See §1.
