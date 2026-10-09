# Pre-registration — Z3: does connecting the shape space's joint channels make shape transfer?

**Written and committed BEFORE the runs.** Date 2026-08-31. Series `Z`.
Folder `diagnostics/joint_blendshapes/`.

## 1. The claim under test

`OmniAnt_25PCs_joint_limited.pkl` ships a 25-D shape space with **three** channels indexed by the
same betas: `shapedirs` (10235×3×25, vertices), `scaledirs` (25×55×3, per-joint scale),
`transdirs` (25×55×3, per-joint translation). `SMAL.__init__` loaded **only `shapedirs`**. So on
the shipped pipeline a specimen's shape was 25 modelled numbers plus 330 unmodelled free ones
plus 30,705 free offsets. Measured on Z2 arm A (`D1_PROD`, shape space open, betas |z| 1.106):

| channel | dims | modelled | gen@20/spread | spread |
|---|---:|---|---:|---:|
| `betas` | 25 | yes | **0.2641** | 0.38983 |
| `log_beta_scales` | 165 | `scaledirs` unused | 0.7403 | **0.21446** |
| `betas_trans` | 165 | `transdirs` unused | 0.7971 | 0.06600 |
| `deform_verts` | 30,705 | no | 0.9881 | 0.00577 |

`log_beta_scales` carries **55% as much inter-specimen variation as betas** and generalises at
0.74. Regressing the fitted scales on the model's own prediction gives **R² = 0.002,
corr −0.047** — what the fitter chose is unrelated to what the model says.

> **H: joint scale and translation are shape, and modelling them with the channels the model
> already ships moves that variation out of unmodelled per-specimen parameters and into the
> betas — where it transfers across specimens.**

§6.6 identified this root cause. §6.7 recorded it as refuted, but every arm in that refutation had
**betas sd 0.00042–0.00050** — the frozen `optimise_hierarchical` path. Coupling scale to a betas
vector pinned at sd 0.0005 is coupling it to a constant; that test could not return a positive.
The hypothesis is untested on an open-shape pipeline, and the implementation it used is no longer
in the repo.

## 2. Arms

`bench50_clean` (50 real workers, 37 genera), `D1_PROD.yaml`, seed 0, three arms:

- **A** — as shipped. `COUPLE_JOINT_BLENDSHAPES=0`. (Z2 arm A is this configuration; re-run here
  so all three arms share one job and one environment.)
- **B** — `COUPLE_JOINT_BLENDSHAPES=1`. Free `log_beta_scales`/`betas_trans` kept as an
  **unpenalised** residual. Isolates coupling alone.
- **C** — B plus `w_jresid: 5.0`, penalising the free residual so the betas must carry the bulk.
  **C is the arm the hypothesis predicts.** The weight is inherited from §6.7's `CPL_b`, the only
  value ever run; it is not calibrated here, and §5 states in advance what that costs.

## 3. VOIDING checks

1. **Coupling active** — already verified (`verify_coupling_PROBE.py`): OFF bit-identical to the
   uncoupled composition (max|diff| 0.000e+00), ON == OFF at beta=0, and driving beta_k at +3σ
   moves vertices **2.30× further** on average with coupling on. Re-asserted in the scorer via the
   fitted `scaledirs` contribution being non-zero in B and C.
2. **Shape space open in every arm** — betas mean |z| > 0.5. A frozen arm is not evidence (Z1).
3. **Fit not collapsed** — `chamfer_l1` and `fscore@0.01` in B and C within 10% of A. Coupling
   removes freedom, so the fit getting worse is a real possible outcome and is bounded here rather
   than excused afterwards.

## 4. Endpoints

**PRIMARY (mechanism).** Share of the applied per-joint log-scale that the shape space drives:
`sd(driven) / sd(driven + free)`, over the 50 specimens. In A this is 0 by construction.

- **PASS** — in C the shape space drives **≥ 50%**, i.e. betas carry the majority of joint scale.
- **PARTIAL** — ≥ 20% but < 50%.
- **FAIL** — < 20%: the free residual still out-competes the betas even when penalised.

**SECONDARY, reported alongside and required for a PASS to mean anything.**

- `gen@20/spread` on rest-space shaped geometry, directly comparable to A's 0.5975. The corpus is
  `bench50_clean`, which despite its name is 50/50 a subset of the 757-WORKER corpus (0/50 overlap
  with ALL_ANTS_CLEAN); workers are the only corpus of interest. **A PASS on the primary with gen/spread flat or worse is reported as
  PARTIAL, not PASS** — moving variance between parameters without making it transfer is exactly
  the proxy-without-mechanism pattern this project has hit eight times.
- §6.10 head-carried deform ratio (A ≈ 0.82, corrected value 1.0).
- `b_a_5` p99 |log s|, head max |log s|, free-residual sd.

## 5. What each outcome licenses

- **PASS on both** → this is the shipping change: `COUPLE_JOINT_BLENDSHAPES` on by default plus a
  calibrated `w_jresid`, and the shape space must then be **rebuilt** (`build_shape_space.py`
  currently PCAs `v_template + deform_verts + shapedirs·betas` and would need the joint channels
  folded in). Calibrating `w_jresid` is licensed *only* at that point, as a deployment parameter.
- **PARTIAL** → report; do not ship; do not sweep `w_jresid` hunting for a better number.
- **FAIL** → §6.6's root cause is refuted *on the pipeline where it could actually be tested*, and
  the joint channels are closed. That is a real result and it must be recorded as one.

## 6. Out of scope

Correspondence, scan quality, pose, leg axes, anterior rotation limits (closed by Z2), and the
`b_a_5` scale runaway as an endpoint in its own right.
