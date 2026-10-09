# F6 — geodesic-softened targets: PARTIAL by the letter, indistinguishable from noise in fact

Bar in `REGISTER_F_series_20260828.md`, committed `a35e5539` before job 3280205 ran. 120 epochs
completed; identical recipe to C11 except `--soft_targets`.

| | top-1 | median 3D error |
|---|---|---|
| C11 (one-hot targets) | 0.0709 | 0.02687 |
| **F6 (geodesic-softened)** | 0.0670 | **0.02606** |
| bar for PASS | — | ≤ 0.02284 (−15%) |

- **Endpoint: −3.0%.** Falls, but nowhere near the registered 15%. Per the bar's own wording —
  *"3D error falls but by <15%"* — the verdict is **PARTIAL**.
- **Mechanism check: passes.** Top-1 fell (0.0709 → 0.0670), which is what was registered as
  expected: a soft target deliberately trades exact-vertex precision for neighbourhood
  correctness. The loss did what it was designed to do.

## The honest reading is weaker than PARTIAL

One training run, one number, no replicate. The measured epoch-to-epoch fluctuation in this same
metric, over epochs 40–119, is **0.00096 mean and 0.00238 max** for C11 and 0.00088 / 0.00290 for
F6. The F6-vs-C11 gap at the final epoch is **0.00081** — *smaller than the mean noise between two
consecutive evaluations of a single run.*

**So the effect is not distinguishable from run-to-run noise, and PARTIAL should not be read as a
small real gain.** Establishing whether 3% is real would need several seeds per arm, which was not
pre-registered and is not worth spending on a 3% effect against a 15% bar.

The epoch-matched curves say the same thing: F6 was ahead of C11 at epochs 15, 20, 35, 50 and
behind at 25, 30, 40, 45, 55, 60, 70 — the two trajectories interleave throughout.

## What this closes

The loss-side fix was **the strongest-motivated idea in the series**. Three independent results
pointed at it: F1 (circumferential angle linearly decodable at 52.9° against 90° chance, on the
head's own input tensor), E0 (the metric's measured floor of 0.0908, because near-misses are
genuinely unavoidable), and C3's own docstring calling exact-vertex accuracy "harsh on a dense
mesh". The implementation was verified before launch — σ read off the mesh, self-weight 0.172
against 0.042 uniform, cross-leg target mass masked to exactly 0.0.

It changed essentially nothing.

**That is informative about where the gap is not.** The distance between "the information is
linearly decodable at 52.9°" and "retrieval top-1 is 0.071" is **not** the target encoding. Nor is
it, per F1, the backbone architecture. Nor, per F4 and F2, the selection stage — two independent
attacks there landed at ~4.5% and ~3%.

Four candidate locations for that gap have now been tested and excluded: architecture, target
encoding, selection, and post-hoc assignment. What remains untested is the **embedding geometry
itself** — a 16-dimensional space asked to separate 10,235 vertices, where the head's capacity and
the temperature schedule are the unexamined variables. That is a hypothesis, not a result, and it
is recorded here as the residual rather than started.

## Register — F-series final

| id | idea | verdict | one line |
|---|---|---|---|
| F4 | optimal-transport assignment | **FAIL** (4.3% / 4.5% vs a 5% bar) | selection is not where the headroom is |
| F5 | conformal fit-quality intervals | **FAIL** (width ratio 1.078; worst-decile coverage 0.200) | the GT-free signals do not predict error |
| F6 | geodesic-softened targets | **PARTIAL**, within noise | the target encoding is not the gap either |

Three advanced, literature-grounded ideas, each pre-registered, each built from artifacts already
on disk, none requiring new data. None cleared its bar. Every one of them narrowed where the
remaining gap can be.

**Correction to a flag raised before this result:** I warned that F6 would hit the 12 h wall at
~85 epochs and need epoch-matched comparison. That was wrong — I read cumulative elapsed time as
per-epoch. Actual pace was ~79 s/epoch; the run completed all 120 epochs in 2.6 h. The confound did
not exist and the final-checkpoint comparison is valid.
