# X1 — FAIL. The scale cap compresses the scale statistic and leaves the anterior anatomy untouched.

Bar fixed in `PREREGISTRATION_X1_scalecap_mechanism.md` **before** the runs. Job 3350026, 8 min,
`bench50_clean` (50 real scans), two arms differing **only** in `w_scale`.

---

## The two measurements, side by side

### PROXY — the shipped condition. **Reproduces.**

`exp(log_beta_scales).max()/min()` per anterior group, the statistic the 838-specimen A/B passed
its third decision condition on:

| group | A mean | B mean | Δ mean | A max | B max | **Δ max (tail)** |
|---|---:|---:|---:|---:|---:|---:|
| head | 1.47 | 1.45 | −1.2% | 5.47 | 2.94 | **−46.3%** |
| mandible | 1.71 | 1.69 | −1.3% | 3.63 | 2.87 | **−21.1%** |
| antenna | 2.26 | 2.08 | −7.8% | 6.02 | 4.10 | **−31.9%** |

B compresses on all three. Same direction as the shipped result (max tail −58/−61/−70%), smaller
magnitude on 50 specimens rather than 838. **Voiding check 2 PASSES** — the intervention is engaged
and doing on this corpus what it was reported to do.

### MECHANISM — §6.10's head-carried ratio. **Unmoved.**

Per-group `deform_verts` magnitude relative to thorax:

| group | A | B | Δ | mean \|A−1\| | mean \|B−1\| | closer to 1.0? | sign p |
|---|---:|---:|---:|---:|---:|---|---:|
| **head** | **0.793** | **0.802** | +0.010 | 0.272 | 0.282 | **No** | 0.32 |
| mandible | 1.783 | 1.856 | +0.073 | 0.897 | 0.970 | No | 0.32 |
| antenna | 1.668 | 1.704 | +0.036 | 0.751 | 0.760 | No | 1.00 |
| gaster | 1.382 | 1.399 | +0.016 | 0.482 | 0.486 | No | 0.89 |
| legs | 1.163 | 1.181 | +0.017 | 0.288 | 0.308 | No | 0.065 |

**Head: 0.793 → 0.802.** Distance to 1.0 changed by −0.011 against a ≥0.05 bar, p = 0.32. Not one
group moves toward 1.0 significantly. **FAIL.**

**Addendum, 2026-09-08 — the `gaster` row above is mislabeled, non-gating, correction only.**
`fitter_3d/part_groups.py`'s `PART_GROUPS_COARSE` at the time this ran defined `waist: [24, 25, 44,
45]` and `gaster: [1, 2, 3, 4, 5]`. Joints 24/25/44/45 are actually WING joints (bilateral, dorsal,
thorax-parented; workers are wingless and the template carries no wing geometry, so these hold
~0.0002% of all skinning mass and deform nothing) — they were never the waist. The true waist is
the petiole (`b_a_1`, joint 1) and postpetiole (`b_a_2`, joint 2), which this old grouping instead
lumped into `gaster`. The grouping has since been corrected (uncommitted fix in the working tree as
of this addendum: `wing: [24,25,44,45]`, `waist: [1,2]`, `gaster: [3,4,5]`). **This does not affect
X1's verdict** — head/mandible/antenna/legs/thorax are unchanged, and the gating row (head) is
unaffected and independently reproduced under the corrected grouping (X2,
`RESULTS_X2_20260908.md`). It DOES mean the `gaster` value in the table above is actually
gaster+true-waist combined, not gaster alone — don't cite it as a clean gaster measurement; a
correctly-grouped re-read would need a fresh run, not a relabeling of this table.

## The verdict, per the pre-registered decision table

Row 2 fired: **proxy reproduces, head ratio unmoved.**

> **Scale regularisation is a proxy improvement without mechanism correction. The anterior
> representation defect remains open, and the shipped term's third decision condition is a proxy
> that does not track the symptom it was adopted for.**

This reading was fixed before the run precisely so it could not be chosen afterwards, and the
positive control is what makes it readable: had the proxy *not* reproduced, a flat head ratio would
have meant only that the intervention never engaged here, and I would have reported nothing.

## The §6.10 signature independently replicates

Worth stating separately, because it strengthens the negative rather than weakening it. The
un-capped arm A reproduces the original signature on a fresh run, with `probe_22`'s script long
gone and the grouping reconstructed from `PART_GROUPS_COARSE`:

| | §6.10 (original) | X1 arm A |
|---|---|---|
| head | 0.72–0.76× | **0.793×** |
| antenna | 1.72–1.82× | **1.668×** |
| thorax abs | 0.019–0.043 | in range (check 4) |

So the head-under-carried / antennae-reaching defect is real, reproducible, and **the shipped scale
cap does not address it.** The two facts together are the finding.

## Mechanism checks

| check | result |
|---|---|
| grouping provenance | **PASS** — `PART_GROUPS_COARSE` joint counts (1/5/4/1/2/6/36) asserted equal to probe_22's CLAIM 1 table at runtime |
| 1 — arms differ only in `w_scale` | **PASS** — configs diff on exactly two lines, both `w_scale` (0.0 vs 0.052) |
| 2 — proxy reproduces (VOIDING) | **PASS** — B compresses on all three anterior groups |
| 3 — `deform_verts` non-trivial | **PASS** — not pinned at zero in either arm |
| 4 — thorax reference non-degenerate | **PASS** — within §6.10's 0.019–0.043 range, so no near-zero denominator inflating ratios |

## What this changes, and what it does not

**Changes:** the anterior blind spot is **not** closed. `w_scale` should not be cited as having
fixed it. It bounds an outlier tail in a scale parameter — which is a real and defensible thing to
do, and its other two shipped conditions (genus non-inferiority, integrity) are untouched by this
result — but the head-carried defect §6.10 documented is still fully open.

**Does not change:** the shipped A/B's own three conditions, measured on 838 specimens on another
cluster, stand exactly as recorded. X1 does not retract them. It adds a fourth measurement that was
never taken, and that measurement is negative.

**Scope, per §8:** 50 real scans, not the 838; a different corpus and cluster from the shipped
decision. The negative is about the mechanism **on `bench50_clean`**.

## The pattern, now at seven

This is the seventh instance in this investigation of a component improving on its own measure while
the thing that measure stood for did not move — and the first where the proxy in question had
already been **shipped into the production recipe** on the strength of it. The standing rule
("every arm reports a mechanism check alongside its endpoint") was written for arms under test; X1
suggests it is worth applying retrospectively to decision conditions that promoted something.

## What I would do next

Not another regulariser. The §6.10 diagnosis names the actual gap: **45 of 48 non-leg, non-mandible
axes are unconstrained** — head, gaster, waist and antennae have no authored joint limits at all.
The head is free to stay behind while the antennae reach. That is a *representation/limits* gap on
the anterior chain, not a scale-magnitude gap, and it is what a fix would have to address.

## Artifacts

- `x1_anterior_mechanism.py` — both measurements in one script; `anterior_ratio()` copied verbatim
  from `integrity_and_anterior_check.py:41` so the proxy is the same statistic, not a re-derivation.
- `A_nocap.yaml`, `B_scalecap.yaml`, `submit_X1_20260830.sbatch`.
- `out_X1/x1_results.json`; runs at `diagnostics/moonshot/runs/X1_{A_nocap,B_scalecap}`.
- `sbatch_logs/X1_3350026.log` — gitignored (`*.log`); all numbers are in the JSON.
