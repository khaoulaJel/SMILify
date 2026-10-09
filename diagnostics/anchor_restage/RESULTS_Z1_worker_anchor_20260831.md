# Z1 — the worker registration failure is ~85% an artifact of a shape space that was switched off

Same 50 real worker specimens. Same metric. Same code. The only thing that changed is whether
the fitter was allowed to use its shape space.

## The number

`gen@20 / spread`, all rows on the **identical 50 AntScan workers** (verified: `LIM_0` and
`bench50_*` metrics.csv specimen lists intersect in 50/50):

| run | betas sd | mean \|z\| vs prior | shape space | **gen@20/spread** |
|---|---:|---:|---|---:|
| `M7_handoff_midline` | 0.00042 | ~0.001 | **frozen** | **0.9794** |
| `baseline` | — | — | closed | 0.9539 |
| `BPX_noprior` | — | — | prior removed, hier. | 0.9208 |
| `LIM_0` | — | — | closed | 0.9028 |
| **`bench50_G1_learned`** | **0.37935** | **1.103** | **open** | **0.5676** |
| **`bench50_G3_zero`** | **0.36691** | **1.065** | **open** | **0.5491** |
| *ALL_ANTS_CLEAN (n=81, reference)* | | | | *0.4899* |

Rows 1–4 are REFUTE1's published figures (`shape_decomp_REFUTE_PROBE_out.txt`, R3 `S_raw`,
gen/POPspread — the published definition). Rows 5–6 are new, produced by
`z1_rescore_worker_anchor.py`, which **imports** `loo_gen_over_spread` and `rest_space_shaped`
from `y1_shape_space_analysis.py` rather than reimplementing them, so the Y1 ladder and these
rows come from one implementation.

**The published worker anchor is 0.98. On the same specimens with the shape space open it is
0.55–0.57 — and the clean corpus reference is 0.49.**

The worker/clean gap collapses from **0.48** (0.98 vs 0.50) to **0.07** (0.55 vs 0.49).

## Mechanism — and it moves, which is the point

Per this project's proxy/mechanism rule, the proxy alone is not the finding. The variance split
did what the proxy says it did:

| run | free-form share of shape variance | gen@20 of parametric part | gen@20 of free-form part |
|---|---:|---:|---:|
| `bench50_G1_learned` | **18.7%** | **0.2538** | 0.9872 |
| `bench50_G3_zero` | 17.6% | 0.2269 | 0.9824 |
| REPORT §6.4 step 2 (closed era) | **100.0%** | 0.0008 (var 0.0002) | 0.9794 (var 416.91) |

REPORT.md §6.4 step 2 established that in the closed era **100% of rest-space shape variance
lived in `deform_verts`**, and concluded "probe-19's null was measuring `deform_verts`, whose
non-generalisation is close to tautological for 30k per-specimen free parameters."

That is exactly what has changed. With the shape space open, 81–83% of the variance moved into
the parametric channel, and **the parametric channel generalises at 0.23–0.25** — better than
ALL_ANTS_CLEAN's whole-shape 0.49. The free-form residue is still unlearnable (0.98), as it was
and as it must be; it is simply no longer the whole signal.

The R6 in-span fraction stays low (0.15–0.24% vs LIM_0's 0.899%) and is now **the wrong
statistic to read** — it measures where the free-form field sits, and the free-form field is no
longer carrying the shape.

## What this retires

**The "63% unexplained worker gap" does not exist.** Y1 measured the distance from a synthetic
ladder to an anchor produced by a pipeline whose shape channel was inert. Y1's arithmetic is
correct and its VOIDING checks all passed; the anchor was stale. Pose explains the clean corpus,
and on this evidence there is no large worker residual left for scan noise, partiality, topology
defects or mesh quality to explain.

Therefore, and each of these was a live proposal as of this morning:

- **Z3 / the scan-degradation ladder is not licensed.** It was designed to explain a residual
  that shrank from 0.48 to 0.07.
- **The whole-body correspondence programme — classical functional maps, functional-map
  derivatives, DPFM — is not licensed by this evidence.** It was aimed at the same residual.
- **W1–W5's nulls and C14p's ~0.94 redundancy are re-read, not overturned.** They were measured
  against an endpoint dominated by a free-form field carrying 100% of the shape. An intervention
  on correspondence had nothing to move.
- **§6.4's headline — "the registrations contain no learnable shape structure ... this pipeline
  produces surface fits, not registrations" — is false of the current pipeline.**

## The honest limit of this result

This is an **observational comparison across 14 existing runs ordered by betas \|z\|**, not a
controlled A/B. `bench50_G1_learned` and `LIM_0` differ in more than the shape prior. The
`|z|`-vs-`gen@20` ordering is clean and monotone across all 14 real-worker runs scored
(`z1_rescore_worker_anchor_out.txt`), and `bench50_G3_zero` — a control arm, no learned
correspondence — reads 0.5491, so the effect is not the "learned" ingredient. But the
confirmatory experiment is a genuine one-factor A/B and it is cheap:

> **Z2 (proposed):** the same 50 workers, `D1_PROD.yaml` as shipped, two arms differing only in
> `w_beta_prior` (0.0 as shipped vs the closed-era value). Endpoint `gen@20/spread`; voiding
> gate `betas |z|` must separate the arms. One job, Y1-sized.

Note `D1_PROD.yaml` already ships `w_beta_prior: 0.0`, so the shipped pipeline is on the open
side. Z2 confirms the mechanism; it does not gate the conclusion that the anchor was stale.

## Artifacts

- `z1_rescore_worker_anchor.py`, `z1_rescore_worker_anchor_out.txt`, `out_Z1/z1_worker_anchor.json`
- Specimen-identity check: `comm -12` on `LIM_0` and `bench50_G1_learned` metrics.csv → 50/50.
