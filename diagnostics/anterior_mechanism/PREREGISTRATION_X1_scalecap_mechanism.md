# Pre-registration — X1: did the scale cap fix the anterior anatomy, or only the scale statistics?

**Written and committed BEFORE the runs.** Bars fixed here are not revised after seeing a number.

Date: 2026-08-30. Folder: `diagnostics/anterior_mechanism/`.

---

## 1. The question

`w_scale` (scale_cap, 0.052) was **shipped** on 2026-08-13 and promoted into `D1_PROD.yaml`. Its
third decision condition was *"anterior joint-scale pathology measurably reduced"*: max ratio head
8.91×→3.77× (−58%), mandible −61%, antenna −70% on 838 specimens.

That condition is a **scale statistic** — `exp(log_beta_scales).max()/min()` per anterior group
(`integrity_and_anterior_check.py:41`). The symptom it was built to fix is a **different quantity**:
REPORT.md §6.10's *head-carried ratio* — free-form `deform_verts` magnitude per group relative to
thorax, where the head deforms **0.72×** and the antennae **1.72×**, i.e. *"the head is not being
carried forward and the appendages reach to cover the gap."*

**Nothing has ever measured whether the shipped term moved the head ratio.** X1 does.

> **H: bounding per-joint scale corrects the anterior deformation defect, not merely its scale
> statistics.**

This is the sixth instance of the pattern the lab record already tracks — a component improving on
its own measure while the thing that measure stands for is unchecked — which is what makes it worth
one run rather than an assumption.

## 2. Four constraints discovered before designing this, each of which would have invalidated a naive rerun

Recorded because each was a live trap, not hindsight:

1. **`D1_PROD.yaml` now CONTAINS `w_scale: 0.052`** (lines 45, 65) — it was promoted into the
   production recipe. Reusing the original A/B scripts verbatim would have run **two identical
   arms** and produced a guaranteed null. Arm A is therefore a copy with `w_scale: 0.0`.
2. **The original A/B runs are gone.** No `AB_A_*`/`AB_B_*` directories exist; `out_A/` holds only
   `analysis.json`. This is **not** a re-score — the fits must be redone.
3. **The 838-specimen corpus is on another cluster** (`/p/scratch/cias-7/...`, absent here).
4. **`probe_22_anterior_blindspot_PROBE.py` was never committed** (gitignored `probe_*.py`); only
   its `_out.txt` survives. The grouping is reconstructed from `fitter_3d/part_groups.py`
   `PART_GROUPS_COARSE`, which is derived programmatically from `kintree_table` and whose joint
   counts match probe_22's own table exactly (thorax 1, gaster 5, waist 4, head 1, mandible 2,
   antenna 6, leg 36).

## 3. Design

- **Corpus: `bench50_clean`, 50 real scans** — the corpus §6.10's 0.72× was measured on, and
  available locally. Not the 838; this is a smaller, repeatable run, and the effect under test is
  large (0.72× vs 1.0×).
- **Arm A** — `A_nocap.yaml`: `D1_PROD.yaml` with `w_scale: 0.0` on both stages.
- **Arm B** — `B_scalecap.yaml`: `D1_PROD.yaml` as shipped, `w_scale: 0.052`.
- Verified: the two configs differ **only** in `w_scale` (diff is exactly two lines). `results_dir`
  is overridden on the CLI so the configs stay otherwise byte-identical.
- Both arms retain `Stage_3_deform_fine.npz`, which `trainer_moonshot.py:747` writes with
  `deform_verts` and `log_beta_scales`.

## 4. The two measurements, computed side by side in one script

This is the design's whole point: **the proxy is the positive control for the mechanism.**

- **PROXY (already shipped, expected to reproduce):** `exp(log_beta_scales).max()/min()` per
  anterior group, arm B vs A. Reusing `anterior_ratio()` from
  `integrity_and_anterior_check.py:41` unchanged.
- **MECHANISM (never measured):** per-group `deform_verts` magnitude relative to thorax, using
  `PART_GROUPS_COARSE`; vertices assigned to a group by dominant skinning weight. Reported for
  head, mandible, antenna, gaster, waist, legs. **Head is the endpoint.**

## 5. Pre-registered bar

Endpoint: **head deform ratio (head/thorax)**, arm B vs arm A, paired across the 50 specimens.
§6.10's arms sit at 0.72–0.76×; the corrected value is 1.0×.

- **PASS (mechanism corrected)** — head ratio in B is closer to 1.0 than in A by **≥0.05 absolute**,
  with paired sign test **p < 0.05**.
- **PARTIAL** — moves toward 1.0 significantly, but by <0.05.
- **FAIL** — no significant movement toward 1.0.

Reported alongside, gating nothing: antenna ratio (should *fall* from ~1.72 toward 1.0 if the
mechanism is corrected), mandible, and the other groups.

## 6. What each outcome licenses — fixed now

| outcome | reading |
|---|---|
| proxy reproduces **and** head ratio → 1.0 | the representation bound is the mechanism; a shipped intervention is doing what it was meant to |
| **proxy reproduces, head ratio unmoved** | **decisive negative: scale regularisation is another proxy improvement without mechanism correction.** The anterior representation defect remains open, and the shipped term's third decision condition is revealed as a proxy that did not track its symptom |
| proxy does **not** reproduce | we have not reproduced the shipped intervention on this corpus; the mechanism reading is **VOID** and nothing is concluded about H |

The third row is a **voiding** condition, not a result.

## 7. Mechanism checks

1. **Arms differ, and only in the intended way (VOIDS).** Assert `w_scale` is 0.0 in A and 0.052 in
   B, and that no other config key differs.
2. **Proxy reproduction (VOIDS the mechanism reading).** Arm B must compress the anterior
   joint-scale range relative to A, in the direction the shipped A/B reported.
3. **`deform_verts` present and non-trivial in both arms.** If either arm has `deform_verts` absent
   or identically zero, the ratio is undefined and the run is void — note some D1 variants pin
   `deform_verts` at exactly 0, which would make this measurement meaningless.
4. **Thorax reference non-degenerate.** The ratio divides by thorax deformation; report its absolute
   value per arm (§6.10's was 0.019–0.043) so a near-zero denominator cannot silently inflate ratios.

## 8. Not claimed

This is 50 real scans, not the 838 the shipped decision used, and not the same cluster or corpus. A
negative here says the mechanism did not move **on bench50_clean**; it does not retract the shipped
A/B's own three conditions, which were measured on their own corpus and stand as recorded.
