# W3b — at convergence: the coxa result is real and grows. The rest does not close, and the line ends here.

Bar fixed in `PREREGISTRATION_W3b_convergence.md` **before** the run and **identical to W3's**.
Job 3338567, 150 epochs / 7879 s (5× W3's budget). Only `--epochs` changed: 30 → 150. Same splits,
same eval specimens (3988–3999), same seed, same density, same six checks.

---

## Endpoint: PARTIAL, same shape as W3 — but the meaning has changed

| seg | XYZ_RIGID | **W3b** | rel. | 95% CI | better | sign p | verdict | *(W3 @30ep)* |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| **`co`** | 0.2689 | **0.2012** | **+25.2%** | **[+21.7%, +28.2%]** | 732/960 | 7.4e-65 | **PASS** | *+16.4%* |
| `tr` | 0.0753 | 0.0798 | −6.1% | [−8.6%, −3.0%] | 435/960 | 4.1e-03 | no | *−17.1%* |
| `fe` | 0.0881 | 0.0969 | −9.9% | [−13.8%, −6.9%] | 370/960 | 2.9e-12 | no | *−9.9%*→ from −20.5% |
| `ti` | 0.0917 | 0.1148 | −25.1% | [−29.3%, −20.3%] | 245/960 | 4.3e-53 | no | *−41.2%* |

**The coxa result is not a budget artefact.** It was +16.4% undertrained and is **+25.2%** at
convergence — it *strengthened* with training, which is what a real effect does. The
pre-registration fixed this reading in advance (§3): *"If `co` alone passes again at convergence,
that is no longer a budget artefact — it becomes the finding: a learned descriptor helps exactly
where rigid proximity is weak, and nowhere else."* That is now the finding.

## Convergence — the check that governed W3's reading

| epoch | 0 | 29 | 59 | 89 | 119 | 149 |
|---|---:|---:|---:|---:|---:|---:|
| loss | 3.5860 | 3.1557 | 3.0677 | 3.0095 | 2.9686 | 2.9490 |
| in-part top-1 | 0.0482 | 0.0847 | 0.0930 | 0.0987 | 0.1036 | **0.1055** |

Gain per 30 epochs: **+0.0365 → +0.0083 → +0.0057 → +0.0049 → +0.0019.** The curve is strongly
decelerating — the last 30 epochs bought **4× less** than the preceding 30. It is *still rising
marginally*, so by the letter of the pre-registration this is not fully converged and I say so; but
W3b was registered as **the last extension**, and I am not extending again.

**That matters for how the failing segments read.** More training helped them substantially and
uniformly — `tr` −17.1% → −6.1%, `fe` −20.5% → −9.9%, `ti` −41.2% → −25.1% — roughly 10–16
percentage points for 5× compute. But the bar is **+15%**, so `tr` needs a further ~21 points and
`ti` ~40. On a decelerating curve that has just delivered its smallest increment, extrapolating
another 5× to cover that is not supportable. **The honest reading is that `tr`/`fe`/`ti` do not
reach the bar by training alone**, and this is no longer "inconclusive on budget" — it is a
reasoned negative with the trend measured rather than assumed.

## The objective-mismatch finding strengthens

Same subsets, identical backbone and warm start, differing **only** in training objective:

| seg | CSE_C11 | W3b | rel. reduction | sign p | *(W3 @30ep)* |
|---|---:|---:|---:|---:|---:|
| `co` | 0.4305 | 0.2012 | **53.3%** | 1.4e-276 | *47.2%* |
| `tr` | 0.2259 | 0.0798 | **64.6%** | 7.3e-279 | *60.2%* |
| `fe` | 0.3052 | 0.0969 | **68.2%** | 7.7e-257 | *65.7%* |
| `ti` | 0.3592 | 0.1148 | **68.1%** | 2.9e-235 | *63.7%* |
| `ta` | 0.3977 | 0.2139 | **46.2%** | 1.3e-141 | *44.1%* |

**46–68% on every segment.** This is the most transferable result of the W-series and it is now
confirmed at convergence: W2's failure was **objective mismatch**. C11 was optimised on negatives
spread across all 10,235 vertices — overwhelmingly other parts — and judged on within-part ones.
Train on the negatives the evaluation uses and two-thirds of that gap closes.

## Mechanism checks — all six

| check | result |
|---|---|
| 1 — split disjointness | PASS (train 0–3699 / val 3700–3799 / eval 3988–3999, asserted) |
| 2 — checkpoint loaded | PASS (strict, 0 missing / 0 unexpected, `head.0.weight` byte-matched) |
| 3 — warm start transferred | PASS (196 backbone tensors, probe tensor changed) |
| 4 — chance control | PASS, ratios **0.978–1.007** |
| 5 — overfitting | PASS, gaps **−0.001 to +0.017** |
| 6 — convergence | decelerating, still marginally rising — see above |

Check 5 is worth a note: gaps widened slightly from W3 (−0.008..+0.012 → −0.001..+0.017) with 5×
the training, which is the expected direction and remains small. The network is not memorising
3700 specimens.

---

## What the W-series concludes

Four descriptor families have now been measured against plain part-local rigid proximity on the
same protocol, same specimens, exact ground truth:

| family | vs `XYZ_RIGID` |
|---|---|
| handcrafted local geometry (W1 `LOCAL_GEO`) | near chance |
| anatomy-conditioned coordinates (W1 `PART_FRAME`) | worse — and W4 showed the cause was metric anisotropy, not lost morphology |
| learned, mismatched objective (W2 `CSE_C11`) | worse on 3/4, tie on `co` |
| learned, matched objective, converged (W3b) | **`co` +25.2%**; `tr`/`fe`/`ti` still worse |

**The pattern is consistent and has a mechanism.** `XYZ_RIGID` wins wherever it has good geometry
to work with — `tr`/`fe`/`ti` at 0.075–0.092 against a ~0.31 chance floor, i.e. ~75–80% of the gap
to zero already closed. It is *weak* only on `co` (0.269), where stubby coxae give a poor principal
axis (W1: 7.5% degenerate frames, concentrated in `co`/`ta`). The learned descriptor wins exactly
there and nowhere else. **What identifies a point within a correct segment is largely where it is;
a learned descriptor adds value only where "where it is" is hard to establish.**

**Per §7, PARTIAL → report and stop.** No fitter run, no scaling up, and none was done.

**The one thing that would be worth doing** — and it needs its own pre-registration, not this
document's permission — is testing whether the `co` gain reaches the fitter, because the coxa is
precisely where SMILify's residual has been concentrated. I would set expectations low: retrieval
gains have failed to reach the fitter four times, and the coxa specifically was already shown inert
to four separate interventions (correspondence alone −0.009, tolerance reweighting −0.010, GT pose
init −0.009, frozen GT pose −0.005). A fifth failure is the default expectation.

**Not claimed:** anything about real scans. Synthetic, exact correspondence by construction, oracle
part label on both sides, no noise, partiality, or topology defects.

## Artifacts

- `out_W3b/w3_results.json`, `w3_rows.csv`, `w3_best.pt` (epoch 149).
- `submit_W3b_20260830.sbatch`; `sbatch_logs/W3b_3338567.log` (gitignored, `*.log`; the full
  loss/accuracy history is in `w3_results.json` under `history`).
