# W3 — PARTIAL. The coxa passes; the objective mismatch is confirmed as W2's cause; training did not converge.

Bar fixed in `PREREGISTRATION_W3_contextual_embedding.md` **before** the run. Job 3337436, `c23g`,
30 epochs / 1636 s. Train 0–3699, model-selection val 3700–3799, eval **3988–3999** — the same 12
specimens W1 and W2 used, disjoint from training and asserted at runtime.

---

## Endpoint: PARTIAL — 1 of 4 bar segments passes

Median normalised 3D within-part retrieval error on density-matched subsets (4096 verts × 8
subsets), every arm recomputed on the **same** subsets. Ceiling = 0.

| seg | RANDOM | XYZ_RIGID | PART_FRAME | CSE_C11 | **W3** | W3_SHUFFLED |
|---|---:|---:|---:|---:|---:|---:|
| `co` | 0.4232 | 0.2689 | — | 0.4253 | **0.2247** | 0.4213 |
| `tr` | 0.3443 | **0.0753** | — | 0.2213 | 0.0881 | 0.3458 |
| `fe` | 0.3232 | **0.0881** | — | 0.3091 | 0.1062 | 0.3155 |
| `ti` | 0.3094 | **0.0917** | — | 0.3563 | 0.1295 | 0.3038 |
| `ta` | 0.3429 | **0.1616** | — | 0.4135 | 0.2312 | 0.3450 |

Against the bar (≥15% relative + sign p<0.05 vs **subset-matched** `XYZ_RIGID`):

| seg | XYZ_RIGID | W3 | rel. | 95% CI (bootstrap) | better | sign p | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| **`co`** | 0.2689 | **0.2247** | **+16.4%** | **[+12.6%, +19.8%]** | 640/960 | 7.8e-26 | **PASS** |
| `tr` | 0.0753 | 0.0881 | −17.1% | [−20.2%, −13.4%] | 314/960 | 9.1e-27 | no |
| `fe` | 0.0881 | 0.1062 | −20.5% | [−24.9%, −16.6%] | 268/960 | 1.1e-43 | no |
| `ti` | 0.0917 | 0.1295 | −41.2% | [−46.4%, −36.3%] | 154/960 | 1.3e-105 | no |

**The coxa passes**, with the bootstrap CI clear of the 15% bar at its lower end. This is the first
time in this investigation that anything has beaten the geometric baseline on `co` in a
correspondence setting — the segment that resisted four separate pre-registered mechanisms
(correspondence alone −0.009, tolerance reweighting −0.010, GT pose init −0.009, frozen GT pose
−0.005) and the one segment where W2's C11 managed a tie rather than a loss.

**Why the coxa and nothing else — a mechanism, not a coincidence.** `XYZ_RIGID` is *weakest*
exactly where W3 wins: 0.2689 on `co` versus 0.075–0.092 on `tr`/`fe`/`ti`. Coxae are stubby, so
their principal axis is a poor proxy for the anatomical one (W1 measured 7.5% of part frames
degenerate, concentrated in `co` and `ta`) and rigid proximity has little to work with. The learned
descriptor helps precisely where geometry has the least to offer, and cannot compete where
geometry is already closing ~75–80% of the gap to zero.

**Kept in proportion:** `co` remains the *worst* segment in absolute terms. 0.2247 is still ~2.5×
`tr`'s 0.0881. W3 improves the hardest segment; it does not solve it.

## The secondary result is the cleanest thing here

Same subsets, W3 vs C11 — i.e. the identical backbone and warm start, differing **only** in
training objective:

| seg | CSE_C11 | W3 | rel. reduction | sign p |
|---|---:|---:|---:|---:|
| `co` | 0.4253 | 0.2247 | **47.2%** | 3.8e-268 |
| `tr` | 0.2213 | 0.0881 | **60.2%** | 4.6e-255 |
| `fe` | 0.3091 | 0.1062 | **65.7%** | 4.8e-243 |
| `ti` | 0.3563 | 0.1295 | **63.7%** | 3.7e-226 |
| `ta` | 0.4135 | 0.2312 | **44.1%** | 8.4e-133 |

**44–66% on every segment.** This confirms the hypothesis W3 was built to test: W2's failure was
**objective mismatch**, not a limitation of learned descriptors. C11 was optimised on negatives
drawn across all 10,235 vertices — overwhelmingly other parts — and judged on within-part
negatives. Training on the negatives the evaluation uses recovers roughly two-thirds of that gap.

This is a reusable finding about how to train these heads, and it holds regardless of the endpoint.

## Mechanism checks — six run, all clean, one of them changes the verdict's reading

**Check 1 — split disjointness: PASS.** train 0–3699 / val 3700–3799 / eval 3988–3999, asserted at
runtime. No leak.

**Check 2 — checkpoint loaded: PASS.** `strict=True`, 0 missing / 0 unexpected, `head.0.weight`
byte-matched against the raw file.

**Check 3 — warm start transferred: PASS.** 196 backbone tensors copied from C11 and
`sa1.conv_blocks.0.0.weight` verified changed from fresh init.

**Check 4 — chance control: PASS, and tightly.** Independently permuted W3 embeddings land at the
`RANDOM` floor on every segment: ratios **0.976–1.006**. The signal is in the descriptor, not
leaking through the pipeline.

**Check 5 — overfitting: PASS.** Train-vs-held-out gaps are **−0.008 to +0.012** across all
segments. The network learned correspondence, not specimen identity — notable given it trained on
3700 specimens and is evaluated on 12 unseen ones.

**Check 6 — convergence: FAILED, and this governs how the failing segments are read.**

| | epoch 0 | epoch 29 |
|---|---:|---:|
| loss | 3.5864 | 3.1183 |
| in-part top-1 | 0.0483 | **0.0874** |

In-part top-1 **still nearly doubled over the run and was still climbing at the last epoch.** The
network is undertrained; 30 epochs was a budget choice, not a convergence point.

**The pre-registration fixed the consequence in advance (§8): "If training has not converged in
that budget the arm is reported as *inconclusive on budget*, not as a FAIL."** So the honest
verdict is *not* the script's headline PARTIAL, which applies the endpoint alone:

> **`co`: PASS. `tr`/`fe`/`ti`: INCONCLUSIVE ON BUDGET, not FAIL.**

Every W3 number here is a **lower bound**. I am flagging this as a limitation of my budget choice,
not softening a result after seeing it — the clause was written before the run precisely so this
reading could not be chosen afterwards.

---

## What this licenses

Per §7, PARTIAL → **report the segments and stop. No fitter run, no scaling up.** That reading
stands and I have not run a fitter experiment.

The one registered remedy for check 6 is a longer run **at the identical bar** — extending training
is not moving the goalposts, and the bar in the pre-registration is unchanged. That is the natural
next step and it needs its own brief pre-registration recording the epoch count before it starts.

**What is now established:**
- Objective mismatch, not learned descriptors as such, explains W2's failure (44–66%, every segment).
- A learned descriptor **can** beat rigid proximity — on the one segment where rigid proximity is
  weak — and does so with clean chance and overfitting controls.
- Where geometry is already strong (`tr`/`fe`/`ti` at 0.075–0.092 against a 0.31 floor), an
  undertrained descriptor does not compete, and the remaining headroom there is small.

**Not claimed:** anything about real scans, and anything about the fitter. Synthetic, exact
correspondence by construction, oracle part label on both sides, no noise or partiality. Retrieval
gains have failed to reach the fitter four times; a `co` retrieval gain is a hypothesis about the
fitter, not a result about it — which matters here specifically because the coxa's fitter-side
behaviour was already shown inert to four separate interventions.

## Artifacts

- `w3_contextual_embedding.py` — model, loss, training, evaluation.
- `submit_W3_20260830.sbatch` — GPU job (`c23g`, account `rwth2151`).
- `out_W3/w3_best.pt`, `w3_results.json` (bars, CIs, all six checks, loss history), `w3_rows.csv`.
- `sbatch_logs/W3_3337436.log` — on disk; `*.log` is gitignored repo-wide (`.gitignore:153`), so it is not in the commit. The full loss/accuracy history is in `w3_results.json` under `history`.
