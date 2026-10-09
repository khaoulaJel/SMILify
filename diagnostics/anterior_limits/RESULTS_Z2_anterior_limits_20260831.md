# Z2 — FAIL. Anterior rotation freedom does not sustain the head defect.

Bar fixed in `PREREGISTRATION_Z2_anterior_limits.md` before the run. Job 3352880, 8 min 29 s,
COMPLETED. `bench50_clean`, 50 real workers, `D1_PROD.yaml` unchanged in both arms; arms differ
only in `SMILIFY_SMAL_FILE`.

## Endpoint — MECHANISM. Unmoved.

§6.10's head-carried deform ratio, paired across the 50 specimens:

| | A (stock) | B (anterior limits) | delta |
|---|---:|---:|---:|
| mean head/thorax ratio | **0.8202** | **0.7948** | −0.0254 |
| distance to 1.0 | 0.2786 | 0.2814 | **−0.0028** (bar ≥ +0.05) |
| specimens closer to 1.0 in B | — | 29/50 | sign-test **p = 0.322** |

**Verdict FAIL.** The ratio moves *away* from 1.0, by an amount indistinguishable from noise.

## The voiding checks all passed, so this is a real null

**4.1 — the bands bind.** Arm B is inside every authored band; arm A exceeds on 5 of 12 joints,
and the clipping is substantial:

| joint | band | A max | B max |
|---|---:|---:|---:|
| `b_a_5` gaster tip | 45° | **83.2°** | 45.0° |
| `an_3_r` | 75° | **100.0°** | 74.4° |
| `an_2_l` | 90° | **100.0°** | 74.8° |
| `an_3_l` | 75° | 88.8° | 74.2° |
| `b_a_4` | 40° | 50.4° | 40.0° |

**4.2 — shape space open in both arms.** A mean |z| 1.106, B 1.118.

**4.3 — thorax denominator** 0.00450 / 0.00459, consistent between arms; the ratio is read paired
within-arm, as pre-registered.

So the intervention did what it was built to do — it removed real, substantial anterior rotation
excess, including a 100° antennal funiculus and an 83° gaster tip — **and the head defect did not
care.**

## Why the hypothesis was wrong

**`b_h` never exceeded its band even in arm A: head max rotation 49.6°, against a 60° band.**

The 101.1° head rotation that motivated Z2 was measured on `bench50_G1_learned`. Under
`D1_PROD.yaml` — the shipped recipe — the head only reaches 49.6°. The head's rotation freedom is
real in the model file, but the production fitter was never using it to excess. The hypothesis
rested on a number from a different arm, and it should have been re-measured on `D1_PROD` before
the model was built. That is the same class of error as Z1's stale anchor: **a number is evidence
only about the pipeline that produced it.**

## What this closes

Per §6 of the pre-registration: **the anterior representation gap is closed as a hypothesis.** The
head defect is sustained by something other than rotation freedom. Do **not** re-run with
different band values — that is the scale-cap-variant pattern X1 closed.

The eighth proxy/mechanism split, and the first where the *proxy* also declined to move: gen@20
went 0.5975 → 0.6050 (worse by 0.0075, negligible), head max rotation 49.6° → 51.7°, `b_a_5` p99
`|log s|` 0.6970 → 0.6977. Constraining the anterior changes nothing measurable in either family.

`OmniAnt_25PCs_anterior_limited.pkl` is a test instrument that failed its test. It is **not** a
shipping candidate and D1_PROD should continue to point at
`OmniAnt_25PCs_joint_limited.pkl`.

## What arm A is worth keeping

Arm A is the first `D1_PROD.yaml` fit of `bench50_clean` whose parameter `.npz` is retained
locally, so it re-anchors Z1 and Z1b on the shipped recipe rather than on `G1_learned`:

| | Z1/Z1b (`G1_learned`) | **Z2 arm A (`D1_PROD`)** | published (closed shape) |
|---|---:|---:|---:|
| gen@20/spread, 50 workers | 0.5676 | **0.5975** | 0.98 |
| §6.10 head ratio | 0.70 | **0.8202** | 0.72–0.76 (X1: 0.793) |

Z1's conclusion is unchanged and slightly softened: on the shipped recipe the worker anchor is
**0.5975** against a published 0.98 and a clean-corpus 0.4899. Arm A also reproduces X1's 0.793
head ratio (0.8202) more closely than `G1_learned` did, confirming X1's figure as the right
reference for the head defect.

## Artifacts

`z2_build_limited_model.py`, `z2_score.py`, `submit_Z2_20260831.sbatch`,
`sbatch_logs/Z2_3352880.log`, `out_Z2/z2_results.json`; fits at
`diagnostics/moonshot/runs/Z2_{A_stock,B_limits}`.
