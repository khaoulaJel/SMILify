# W4 — W1's endpoint stands, but W1's *explanation* was wrong. The damage was metric anisotropy, not lost morphology.

Predictions fixed in `PREREGISTRATION_W4_normalisation_ablation.md` **before** the run. CPU-only,
same 12 held-out specimens (3988–3999), same 20 pairs, same 30 parts, same metric as W1.

---

## Headline

**W1 concluded:** *"normalising a part to canonical shape destroys correspondence-bearing signal —
inter-specimen differences in segment proportion are what identify a point, not nuisance to divide
out."*

**That explanation does not survive.** Removing overall size **isotropically** costs essentially
nothing. The catastrophic degradation W1 measured came from rescaling the axial and radial
directions by *different* factors, which distorts the nearest-neighbour metric — not from throwing
away morphology.

W1's **endpoint** is unaffected: `PART_FRAME` did lose to rigid proximity on every segment, and
that number is reproduced here exactly. Only the mechanism I attributed it to was wrong.

## The ablation

Cost relative to `PF_none` (no normalisation). Negative = normalising made it worse.

| seg | `PF_none` | ÷L only | ÷R only | ÷both | axial-only | **ISO ÷L all** |
|---|---:|---:|---:|---:|---:|---:|
| `co` | 0.2813 | −2.3% | −12.1% | −1.4% | −11.6% | **+0.1%** |
| `tr` | 0.0790 | +2.2% | −52.1% | −20.2% | −10.2% | **+0.5%** |
| `fe` | 0.0993 | +4.0% | −33.3% | −7.4% | −3.2% | **+4.1%** |
| `ti` | 0.0910 | +1.4% | −128.5% | −26.8% | +1.0% | **+1.9%** |
| `ta` | 0.1706 | −2.9% | −69.0% | −5.6% | −3.4% | **+2.0%** |

**`ISO ÷L all` — dividing all three coordinates by the same scalar — is neutral-to-positive on
every segment (+0.1% to +4.1%).** That is the biologically meaningful normalisation: remove the
specimen's overall size, keep its shape. It is free.

**`÷R only` is catastrophic (−12% to −128%).** And `÷L only` is roughly neutral (−2.9% to +4.0%).

## The mechanism, and why the confound had to be controlled

`PF_axial`, `PF_radial` and `PF_both` each divide **one** direction by **its own** scale. That does
two things at once, and they are not separable in an anisotropic arm:

1. it removes information about that direction's absolute extent, and
2. it changes the **relative weighting of axial vs radial inside the Euclidean NN metric**.

`PF_uniform` was added specifically to separate them: one scalar for all three coordinates leaves
the relative weighting *exactly* untouched and removes only overall size. It costs nothing — which
localises the entire effect to (2).

The direction of the effect confirms it. W1's own mechanism check 3 found the residual on
`tr`/`fe`/`ti` is **axial-dominated** (circ/axial 0.57–0.73). Dividing the radius by `R` inflates
the radial coordinate to O(1) while the axial coordinate stays in absolute units, so the metric
*down-weights the axial direction* — precisely the direction that carries the signal. Hence
−128% on `ti`. Dividing by `L` does the opposite, up-weighting axial, and is mildly *helpful* on
`tr`/`fe`/`ti`.

So the real lesson is narrower and more useful than W1's:

> **Anisotropic rescaling of a part frame is harmful because it silently re-weights the
> nearest-neighbour metric away from the axial direction, which is where the within-part signal
> lives. Isotropic size normalisation is free. Morphological proportion is not the thing being
> destroyed.**

## Registered predictions

**P1 (monotonicity) — VIOLATED on all five segments,** which the pre-registration fixed in advance
as the outcome requiring W1's lesson to be narrowed:
- `÷L HELPS` on `tr`/`fe`/`ti` — normalising *improved* on not normalising.
- `÷both < single` on all five — normalising *both* is better than normalising radius alone,
  because the second division partly restores the weighting the first one broke.

**P2 — FALSIFIED, decisively and in the opposite direction.** Predicted axial normalisation would
be more harmful than radial. It is the reverse on **every** segment, sign p from 7.3e-05 to
1.5e-36. My stated reason for P2 (length varies more across specimens than thickness, so carries
more identifying proportion) was reasoning about *information*, when the dominant effect is
*metric weighting*.

## Mechanism checks

**Check 1 — identity, `PF_none` == `XYZ_RIGID`: PASS at exactly 0.0e+00** across all 600
part-pairs. This is the voiding check, and **it voided the first run**.

*Run 1 was VOID and its numbers were not read.* My registered claim that `PF_none` is provably
identical to `XYZ_RIGID` was **wrong as implemented**: `frame_coords` subtracted a per-specimen
`t_min` (proximal-tip offset) while `rigid_into` is centroid-referenced, so the two applied
*different* translations to A and B. That is a genuine re-alignment, not a rigid change of basis,
and the identity broke by 0.24. Fixed by making the axial offset an explicit parameter and running
the ladder centroid-referenced; W1's `tipmin` convention is retained as its own arm so check 2
still has something to reproduce. The check did exactly what a voiding check is for — it caught my
error rather than the hypothesis's.

**Check 2 — `PF_both_W1CONV` reproduces W1's `PART_FRAME`: PASS**, to ≤0.01% on every segment
(0.2871 / 0.1020 / 0.1108 / 0.1329 / 0.1901). Nothing drifted between W1 and now.

**Check 3 — chance control: PASS.** `RANDOM` is the worst arm on every segment.

## What this changes

**Corrects, in the lab record:** W1's stated lesson, which is too strong as written. It has already
propagated once — W3's "no canonicalisation" design decision cites it. That decision was not
harmed (W3 uses raw coordinates and passed on `co`), but its *justification* should now read
"avoid anisotropic rescaling", not "morphological variation is the signal".

**Practical guidance this yields for any future descriptor:** isotropic size normalisation is safe
and free, so a method may use it. Independent per-axis normalisation is not, and if a learned
method builds part-relative coordinates it should either keep them isotropic or learn the metric
rather than fixing it to Euclidean.

**Not claimed:** anything about real scans or the fitter. Synthetic, exact correspondence by
construction, oracle part labels on both sides. W4 is a decomposition of an existing result and
licenses no new method.

## Artifacts

- `w4_normalisation_ablation.py` — the ablation; imports W1's frame and retrieval unchanged.
- `out_W4/w4_results.json` — all arms, costs, P1/P2, the identity deviation.
- `out_W4/w4_rows.csv` — 4,800 rows (pair × part × arm).
