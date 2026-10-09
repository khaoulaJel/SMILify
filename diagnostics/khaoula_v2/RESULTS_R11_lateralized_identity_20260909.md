# R11 — FAIL by the pre-registered rule. But the failure is not the one the bar was written for.

Job 3892718, 3m00s, c25g. Bar fixed in `PREREGISTRATION_R11_lateralized_identity.md` before running.
Probe trained on the template only, tested on 6 held-out `synth_power48` specimens.

---

## Verdict: FAIL — DINO does not beat plain geometric proximity.

| specimen | coverage | **LATERAL** | geometric baseline | shuffled control | part acc | mirror-conf |
|---|--:|--:|--:|--:|--:|--:|
| synth_000 | 80.9% | 76.0% | 82.9% | 48.9% | 40.4% | 9.4% |
| synth_001 | 89.6% | 83.4% | 87.7% | 41.5% | 60.3% | 3.2% |
| synth_002 | 85.0% | 81.5% | 84.0% | 52.1% | 45.0% | 7.4% |
| synth_003 | 88.2% | 87.2% | 90.9% | 48.7% | 60.5% | 3.1% |
| synth_004 | 85.1% | 82.0% | 86.6% | 39.1% | 54.0% | 10.4% |
| synth_005 | 90.7% | **85.5%** | 84.1% | 57.6% | 51.2% | 13.4% |
| **mean** | 86.6% | **82.6%** | **86.1%** | 48.0% | 51.9% | 7.8% |

- **Clears the 70% bar** (82.6%) and is overwhelmingly above chance (binomial p ~ 0).
- **Loses to the geometric baseline** (86.1%), winning on **1 of 6** specimens. PASS required
  beating it. -> **FAIL.**
- **Mechanism checks pass**: bilateral classes balanced on the template (antenna 181/181,
  midleg 669/669, hindleg 931/931); shuffled-label control 48.0% ~ chance; baseline scored on the
  identical covered subset.

## The failure is not the one either FAIL row anticipated

Section 5 pre-registered two FAIL readings, and **neither describes what happened**:

- Not *"high mirror-confusion — semantics recovered, laterality not"*: mirror-confusion is **7.8%**,
  low. Errors are not left/right flips.
- Not *"features carry no part identity at this granularity"*: they clearly do — 82.6% lateral and
  ~52% part accuracy across 13 classes, far above chance.

**Laterality is actually recovered.** What fails is that DINO does not beat rigid proximity — the
same outcome W1 (handcrafted descriptors), W2 (CSE head), T1.1 (raw DINO) and W5 (deployed
correspondence) all reached. This is the **fifth** independent instance in this project of a learned
or semantic descriptor losing to plain geometric proximity.

## The design limitation that stops this being the final word — stated plainly

**The geometric baseline is unusually advantaged in exactly this test.** The synthetic specimens
share the template's topology *and* are roughly co-oriented in the normalised frame, so
nearest-template-vertex proximity preserves side almost for free. That is precisely the regime where
geometry works when it is available.

On real scans it is not available: R7 found no reliable bilateral reference frame could be
constructed, and R8 found the body rotationally degenerate about its long axis. In that regime the
geometric baseline should degrade sharply, while DINO's view-based features — which do not depend on
a canonical frame — plausibly would not degrade the same way.

> **R11 as designed does not settle the case that actually matters.** It shows semantic features add
> nothing over geometry *where geometry already works*. It does not show what happens where geometry
> provably fails, which is the real-scan regime the whole identity problem lives in.

That is a limitation of the experimental design, not a property of the method, and it is the single
most important thing to fix before concluding anything about the foundation-feature route.

## What to do instead of escalating

The decision table said FAIL points to geometry-aware semantic correspondence
(*Telling Left from Right*). Given the actual result — laterality recovered, but geometry stronger in
a frame-aligned setting — the sharper and much cheaper next experiment is:

**R11b: re-run the identical comparison with the geometric baseline's advantage removed.** Apply a
random global rotation per specimen (and/or evaluate on real scans) so nearest-vertex proximity no
longer inherits a canonical frame. If DINO's lateral accuracy holds ~82% while the geometric
baseline collapses toward chance, the ordering reverses and the semantic route is vindicated for the
regime it is needed in. If DINO also collapses, the route genuinely closes — and that would be the
decisive negative this experiment was meant to produce.

One more 3-minute run, and it answers the question R11 was actually built to ask.

## Limits

- The logistic probe hit `max_iter` (convergence warnings in the log) — possibly mildly underfit,
  which would understate DINO. It does not change the direction: a better-converged probe would have
  to close a 3.5-point gap and win on 5 more specimens.
- 6 held-out synthetic specimens, shared topology, exact labels. Nothing here is about real scans.
- Level 3 (per-segment laterality) deliberately not attempted, per the preregistration.

## Artifacts

`r11_lateralized_identity.py`, `out_R11/r11_results.json`, `sbatch_logs/R11_3892718.log`.
Two implementation notes for reuse: `T10.backproject` returns `(features, seen_mask)` and zero-fills
unseen vertices (not NaN); `render_3d.py` hardcodes `DEV = "cuda:0"` and must be overridden at
runtime to run CPU-only.
