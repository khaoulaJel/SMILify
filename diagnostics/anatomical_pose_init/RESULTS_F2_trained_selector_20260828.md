# F2 result — Tier 1 FAILS on both segments. The multi-hypothesis line closes.

Bar fixed in `PREREGISTRATION_F2_trained_selector_20260828.md` (committed `a6baa027`) before the
selector was built. C11 backbone loaded strictly, 80 train / 60 test **disjoint specimens** from
the corpus tail the network never trained on, 5 candidates per point, error as a fraction of
segment length.

| seg | top1 | oracle@5 | cycle | **learned** | cycle gap | **learned gap** | rank acc (chance 0.20) |
|---|---|---|---|---|---|---|---|
| `tr` | 0.1181 | 0.0817 | 0.1161 | 0.1172 | 5.4% | **2.5%** | 0.287 |
| `fe` | 0.1824 | 0.1370 | 0.1799 | 0.1805 | 5.7% | **4.2%** | 0.303 |

Tier 1 required **≥ 40% on both segments individually**. Actual: 2.5% and 4.2%. The learned
selector is also **worse than the cycle baseline computed on the same checkpoint and specimens**
(5.4% / 5.7%).

**Verdict — FAIL, per the pre-registered reading:** *"the ~30% headroom is not capturable by a
learned ranker on these features, and the multi-hypothesis line closes."* Tier 2 is not run; the
bar made it conditional on Tier 1.

## The threshold was kept, deliberately, after it became easier to justify lowering

Mid-experiment it emerged that the published cycle baseline (fe 27.0%, tr 15.7%) came from the
**C3** checkpoint at **n=10**, not C11 — so it was never the right reference for a C11-based
selector, and C11's own cycle is much weaker (5.4% / 5.7% here). That is a real technical reason to
re-anchor. It was **not** acted on, because it was discovered *after* seeing that re-anchoring would
make Tier 1 easier to pass. The reason can be true and the timing still wrong. ≥40% stands as
written, and the internal C11 cycle baseline is reported as context, not as the bar.

## Why it failed, and the fifth instance of the pattern

The classifier is genuinely learning: rank accuracy 0.287 / 0.303 against a 0.20 chance level. It
picks the truly-best candidate ~50% more often than chance — **and that buys ~3% of the gap.**

The two are not the same quantity. When the five candidates are all similarly wrong, picking the
best one more often barely moves mean error; and on the occasions the selector overrides `top1` and
is wrong, it can land worse than `top1` would have. Net: an above-chance ranker, a null result.

This is the **fifth** instance of this investigation's recurring decoupling, and the first to occur
*inside a single tier*:

1. C11/C12 — retrieval improved, the fitter saw nothing.
2. bench50's framing — geometric fit improved, correspondence was never checked.
3. H_A — surface correspondence improved 127%, skeletal pose got worse.
4. F1's first run — a plausible "signal at sa1, lost by fp1" pattern from a network that never loaded.
5. **F2 — rank accuracy above chance, error reduction ≈ 0.**

The operational lesson is now specific: **rank accuracy is not a proxy for error reduction**, and
any future selector work must be scored on the error metric directly, never on how often it picks
the best candidate.

## Infrastructure fixed along the way

`multihypothesis_feasibility_20260827.py` wrote to a fixed `multihypothesis.json` with no run
stamp. Re-running it on a different checkpoint silently overwrote the committed C3/n=10 result —
which happened here and was restored from git. The script now stamps the filename with checkpoint
and specimen count, keeps `multihypothesis.json` only for the canonical C3/n=10 config, and
**refuses to overwrite an existing file written with a different config** unless `--force`.

`LAB_RECORD_correspondence_20260827.md` §11 now carries a flag that its figures are n=10 and
checkpoint-specific.

## What is NOT claimed

- Not that no selector could work — this bounds *these features* with *this model class* on
  *this checkpoint*. A different candidate generator or richer features is untested.
- Nothing about real scans; synthetic corpus throughout.
- Oracle@5 is a ground-truth ceiling and is never deployable.
