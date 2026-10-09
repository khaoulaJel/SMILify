# W1 — anatomy-conditioning FAILS its pre-registered bar. Plain geometry is already the better descriptor.

Bar fixed in `PREREGISTRATION_W1_within_part_descriptors.md` **before** the run. 12 held-out
specimens (corpus tail 3988–3999), 20 ordered pairs, 30 leg parts, exact correspondence by
construction (shared template topology → vertex-index identity). No network trained, no fitter run,
no live module touched.

---

## Endpoint: FAIL, and not marginally — the anatomy-conditioned arm is *worse* on every segment

Median normalised 3D within-part retrieval error, lower better, ceiling = 0:

| seg | RANDOM | **XYZ_RIGID** | LOCAL_GEO | **PART_FRAME** | PART_FRAME_AXIAL |
|---|---:|---:|---:|---:|---:|
| `co` | 0.4118 | **0.2813** | 0.4007 | 0.2871 | 0.3213 |
| `tr` | 0.3429 | **0.0790** | 0.3252 | 0.1020 | 0.0942 |
| `fe` | 0.3289 | **0.0993** | 0.3114 | 0.1108 | 0.1099 |
| `ti` | 0.3112 | **0.0910** | 0.2216 | 0.1329 | 0.1121 |
| `ta` | 0.3213 | **0.1706** | 0.3063 | 0.1901 | 0.1899 |

Against the bar (`PART_FRAME` beats `XYZ_RIGID` on all four of `co`/`tr`/`fe`/`ti`, ≥15% relative,
sign p<0.05):

| seg | XYZ_RIGID | PART_FRAME | rel. change | better | sign p | verdict |
|---|---:|---:|---:|---:|---:|---|
| `co` | 0.2813 | 0.2871 | **−2.1%** | 28/120 | 1.4e-08 | no |
| `tr` | 0.0790 | 0.1020 | **−29.1%** | 20/120 | 5.5e-14 | no |
| `fe` | 0.0993 | 0.1108 | **−11.6%** | 34/120 | 2.3e-06 | no |
| `ti` | 0.0910 | 0.1329 | **−46.0%** | 7/120 | 9.5e-26 | no |

**0 of 4 segments pass. The sign is negative on all four**, with the significance running strongly
*against* the hypothesis (`ti`: `PART_FRAME` wins on 7 of 120 part-pairs, p = 9.5e-26). This is not
a near-miss like F4; it is a clean directional refutation.

## What the arms actually mean, and the one-line reading

`XYZ_RIGID` and `PART_FRAME` are built on the **same** part frame and differ in exactly one thing:
`PART_FRAME` normalises the axial extent by *L* and the radius by *R*. So the comparison isolates
normalisation, with nothing else varying.

> **Normalising a part to a canonical shape destroys information rather than adding it.** The
> variation in a segment's length and thickness is not nuisance to be divided out — it is
> *correspondence-bearing signal*. Two ants' femora differ in proportion, and that difference is
> what tells you which point is which. Rescale it away and you have made the matching problem
> harder, uniformly.

The second reading is about `LOCAL_GEO`. Multi-scale local-PCA eigen-features — the standard
handcrafted rotation-invariant descriptor family, Level 1 in the proposal — land essentially at
**chance**: 0.3252 vs a 0.3429 random floor on `tr`, 0.3114 vs 0.3289 on `fe`. They close ~5% of the
gap to the ceiling. On a smooth, near-tubular chitinous segment there is almost no discriminative
local geometry to read; the surface is locally self-similar nearly everywhere. `ti` is the only
segment where they do meaningfully better than chance (0.2216 vs 0.3112).

And the third, which is the practically useful one: **once the part is granted, plain rigid
proximity is already good** — 0.079–0.099 of segment length on `tr`/`fe`/`ti`, i.e. ~8–10%, against
a 0.31–0.34 chance floor. It closes ~70–77% of the available gap with no descriptor at all.

## Mechanism checks — all four reported, one failed prediction

**Check 1 — self-retrieval no-op: PASS.** 0 harness violations. With A = B every arm retrieves a
point at descriptor distance exactly 0. The check was **re-specified during the run** from
index-identity to descriptor-distance, because index-identity flags a legitimate exact tie (two
distinct vertices with identical descriptors) as a bug. `LOCAL_GEO` has such ties at rate 0.0081,
`XYZ_RIGID` and `PART_FRAME` at exactly 0.0000. The tie rate is a property of the descriptor, which
the endpoint already measures; voiding a run for it would have been the wrong reason.

**Check 2 — frame quality: flagged, and shown not to matter.** Median |cos| between the PCA axis
and the anatomical proximal→distal direction is **0.913**; 7.5% of frames are degenerate
(|cos| < 0.5), concentrated in the stubby segments (`co`, `ta`) where the part's longest principal
direction is not its anatomical axis.

*This check was also re-specified, and the reason must be recorded plainly.* The registered version
compared each specimen's axis to a **reference specimen's** axis in world coordinates — but the
specimens are differently **posed**, so a leg legitimately points elsewhere and a negative dot
measures pose, not a broken frame. It fired at 18.9% and would have voided the run for a fault it
could not distinguish from correct behaviour. The replacement (|cos| against the anatomical
proximal→distal direction) is pose-invariant, which is the property the original was reaching for.

The re-specification **cannot rescue the endpoint, and that is why it is safe**: the frame is
shared by both arms, so degeneracy penalises them equally, and `PART_FRAME` *lost*. Tested
directly rather than argued — endpoint recomputed on non-degenerate frames only:

| seg | XYZ_RIGID | PART_FRAME | rel. change | pairs kept |
|---|---:|---:|---:|---:|
| `co` | 0.2434 | 0.2647 | −8.8% | 65/120 |
| `tr` | 0.0790 | 0.1020 | −29.1% | 120/120 |
| `fe` | 0.0971 | 0.1107 | −14.0% | 105/120 |
| `ti` | 0.0910 | 0.1329 | −46.0% | 120/120 |

Same verdict, same direction, every segment. The endpoint is not driven by frame quality.

**Check 3 — axial/circumferential decomposition: the registered prediction FAILED.** Predicted from
F1 (circumferential angle decodable only at 52.9° vs 90° chance) and SurfEmb: the residual should be
circumferential-dominated. Measured:

| seg | axial | circ | circ/axial |
|---|---:|---:|---:|
| `co` | 0.1412 | 0.1757 | 1.24 |
| `tr` | 0.0705 | 0.0515 | **0.73** |
| `fe` | 0.0790 | 0.0487 | **0.62** |
| `ti` | 0.1042 | 0.0592 | **0.57** |
| `ta` | 0.1217 | 0.1095 | 0.90 |

On `tr`/`fe`/`ti` the residual is **axial-dominated** — the opposite of the prediction. Only `co`
is circumferential-dominated. This is worth flagging rather than burying: the standing assumption
that the hard direction on a leg segment is the rotationally-ambiguous circumferential one does not
hold once the part is correctly placed and the frame is normalised. What remains hard is *where
along the segment* a point sits.

**Check 4 — chance control: PASS.** `RANDOM` is the worst arm on every segment.

## Excluded

`pt` (pretarsus) on all six legs, 7–15 template vertices, below the 30-vertex threshold — consistent
with the CSE-feasibility finding that `ta`/`pt` are unsupported. 30 of 36 leg parts used.

---

## What this licenses, and what it explicitly does not

Per the pre-registration's fixed reading, **FAIL** → the within-part gap is not addressable by
anatomy-conditioning, and no network is licensed by this result.

**What is genuinely closed:** handcrafted local-geometry descriptors (Level 1) and
anatomy-conditioned normalised part coordinates (Level 2) — the two cheap options the proposal
put first. Both are now measured, and both are *worse than doing nothing beyond placing the part*.

**What is NOT closed, stated plainly so this is not over-read:** W1 tests *handcrafted* descriptors.
It does not test a *learned* one, and it would be wrong to conclude from this that a learned dense
descriptor cannot beat rigid proximity. What W1 does supply is the number such a descriptor must
now beat, which did not exist before:

> **Any learned within-part descriptor must beat 0.0790 (`tr`) / 0.0993 (`fe`) / 0.0910 (`ti`) /
> 0.2813 (`co`) — plain rigid proximity given a correct part — on held-out specimens, per segment.**

That bar is stricter than it looks, and it is the honest framing for the descriptor question: the
baseline is not chance, it is ~8–10% of segment length. Note also the standing caution that a
retrieval win is not a fitter win — that decoupling has now occurred four times (C11, C12, F2, and
the mechanism-check series).

**Not claimed:** anything about real scans. This is synthetic, with exact correspondence by
construction, no scan noise, no partiality, no topology defects, and an *oracle part label on both
sides*. Whether any of it transfers is what the bench_10 annotation benchmark exists to test.

## Artifacts

- `w1_within_part_retrieval.py` — the experiment, self-contained.
- `out_W1/w1_results.json` — full summary, per-part frame stats, all mechanism checks.
- `out_W1/w1_rows.csv` — 3,000 rows (pair × part × arm), for independent re-analysis.

---

## CORRECTION (2026-08-30, from W4)

**The "Bottom line" mechanism above is too strong and is superseded.** W4 ablated the two
normalisations separately and found:

- **Isotropic size normalisation costs nothing** (+0.1% to +4.1% on every segment). Removing a
  specimen's overall size while keeping its shape is free.
- The damage came from **anisotropic** rescaling — dividing axial by *L* and radial by *R*
  independently — which silently re-weights the Euclidean NN metric away from the axial direction,
  where check 3 above already showed the residual lives. `÷R only` costs −12% to −128%; `÷L only`
  is roughly neutral and mildly *helps* on `tr`/`fe`/`ti`.

**W1's endpoint is unaffected** — `PART_FRAME` did lose on every segment, and W4 reproduces those
numbers to ≤0.01%. Only the explanation ("normalising away morphological proportion destroys the
signal") was wrong. The corrected statement is: *anisotropic rescaling of a part frame is harmful
because of metric re-weighting; morphological proportion is not the thing being destroyed.*

See `RESULTS_W4_normalisation_ablation_20260830.md`.
