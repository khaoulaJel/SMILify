# Pre-registration — F1: is circumferential information present in the backbone's features?

Written **before** the probe is run, 2026-08-28. No GPU retrain; the C11 backbone is frozen.

## The gap this fills

Hard-negative mining (C11) was a **supervision-side** fix — more training pressure — and it moved
direct circumferential retrieval on `tr`/`fe` without the fitter benefiting (C11/C12). A separate
hypothesis has been open since early in this investigation and never tested: that the max-pooling
inside `sa1`/`sa2` **destroys directional information before the loss ever sees it**. No amount of
harder negatives repairs information that is already gone.

The two prior probes left this ambiguous by design: the LRF check said the *geometry* carries the
signal (`circum_gap` 0.38–0.44 on `tr`/`fe`, against a cylinder's 0.031 floor), and the
rotation-ensemble test that could have checked the *network* was found invalid. F1 asks the
network directly.

## Method

Frozen C11 checkpoint (`out_C11_hardneg_20260826/best_model.pt`). Linear (ridge) probes on
intermediate features at four depths, fit on train specimens and scored on the **held-out tail**
of `synth_b2_train_corrb06.npz` — the same split C3/C11 used, so nothing leaks.

| depth | tensor | note |
|---|---|---|
| `xyz` | raw input (3) | trivial upper bound — the angle *is* a function of position |
| `sa1` | 320-d at 512 centroids | after the first grouped max-pool |
| `sa2` | 640-d at 128 centroids | after the second |
| `fp1` | 128-d per point | what the embedding head actually consumes |

**Target — circumferential angle.** Defined once on the **rest template**, per template vertex, so
it is a canonical surface coordinate with no per-specimen frame ambiguity: for each leg segment,
take its PCA principal axis (`segment_frame`, reused verbatim from
`cse_feasibility_retrieval_20260826.py`), project the vertex into the plane orthogonal to that
axis, and take its angle there. Each sampled point inherits θ from its true template vertex.
Regressed as `(cos θ, sin θ)` and scored as **mean angular error**; chance is 90°.

**Positive control — axial coordinate.** The same probe, same features, predicting normalized
position along the segment axis. Prior results say axial is well captured. **If the axial probe
fails, the machinery is broken and no circumferential conclusion may be drawn** — the E0 lesson:
a null is only interpretable when the measurement is shown to be sensitive.

**Chance control.** Targets shuffled within segment; must land at ~90°.

Restricted to `tr` and `fe` — the only segments with a measured well-conditioned circumferential
signal in the raw geometry. `ta`/`pt` are excluded in advance, not after looking.

## Readings fixed in advance

Let `E(depth)` be held-out mean circumferential angular error.

- **DECODER BOTTLENECK** — `E(sa1)` or `E(sa2)` is **≤ 70°** (i.e. clearly better than the 90°
  chance level) while `E(fp1)` is **≥ 80°**. Signal exists in the features and is lost downstream:
  the fix is aggregation/decoding (e.g. attention instead of symmetric max-pool), and a full
  equivariant retrain is **not** justified by this evidence.
- **POOLING DESTROYS IT** — every depth except `xyz` is **≥ 80°**, while `xyz` is ≤ 70° and the
  axial control succeeds at every depth. First direct evidence that the pooled representation does
  not carry the direction. This is the result that would *earn* an equivariant retrain.
- **SIGNAL IS PRESENT THROUGHOUT** — `E(fp1)` ≤ 70°. Then the information reaches the head and the
  circumferential failure is in the *loss/retrieval*, not the architecture — and neither an
  equivariant retrain nor an attention decoder is justified.
- **INCONCLUSIVE** — the axial control fails, or the shuffle control does not land near 90°.
  Report as inconclusive and fix the probe; draw no architectural conclusion.

## What will NOT be claimed

- A linear probe measures *linear decodability*, not presence in an absolute sense. A null bounds
  what a linear head can extract, not what any head could. Stated in the result either way.
- Nothing about real scans; this is the synthetic corpus the backbone was trained on.
- No claim that fixing circumferential retrieval would move `leg_acc` — C11/C12 twice showed
  retrieval gains not reaching the fitter, and F1 does not test the fitter at all.
