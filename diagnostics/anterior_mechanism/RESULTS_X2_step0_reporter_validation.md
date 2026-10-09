# X2 Step 0 — differentiable head-width reporter: validation result

**Date: 2026-09-07. Gate for `PREREGISTRATION_X2_allometric_prior.md` §3 Step 0.**

Script: `diagnostics/anterior_mechanism/x2_head_width_reporter.py` (runnable standalone).

## Verdict: **FAIL** at the pre-registered ≤1% bar — but the reporter's algorithm is verified
## correct. The failure is a vertex-source mismatch, not a bug in the replicated formula.
## **Step 1 (the loss term) stays BLOCKED**, with a specific, testable next step below.

---

## 1. Algorithm replicated (source-cited)

Extracted `diagnostics/atta_reference/blender_bundle/smil_importer.zip`. The relevant code:

- **"10 nearest, inverse distance"** — `smil_importer/core_mesh.py::find_nearest_neighbors`
  (lines ~374-409):
  ```python
  distances = np.linalg.norm(vertices - joint_loc, axis=1)
  nearest_indices[i] = np.argpartition(distances, n)[:n]      # n = 10
  weights = 1.0 / nearest_distances                            # power = 1, i.e. 1/d
  weights /= weights.sum()                                      # normalized to sum to 1
  ```
  Confirmed: **1/d**, not 1/d². Weights **do** sum to 1 (explicit normalization).

- **Which vertex array** — `core_mesh.py::mesh_to_numpy` reads `mesh.data.vertices` (`vert.co`),
  called from `export_J_regressor_to_npy` (line ~613: `vertices, _ = mesh_to_numpy(mesh_obj)`).
  Per `REPORT_ATTA_HEADWIDTH.md` (§"Head-width reporter bones", confirmed during the actual
  Blender session that produced this CSV): `mesh.vertices` always holds **Basis** coordinates
  regardless of which shape key is displayed — i.e. this is the pkl's `v_template`, not a posed
  or shape-keyed array.

- **Reference point for "10 nearest"** — **not** `b_h`. `export_J_regressor_to_npy` builds
  `joint_locations = np.array([bone.head_local for bone in joints])` for **every bone in the
  current armature**, `joints = armature_obj.data.bones` (line ~614-615). `b_h_l`/`b_h_r` are
  bones in that same collection, each with its **own** `head_local` — the position the artist
  placed it at in Blender ("the widest point of the head capsule", per
  `REPORT_ATTA_HEADWIDTH.md` line ~76-82), not `b_h`'s position. **This corrects an assumption
  in the preregistration text** ("presumably nearest to `b_h`'s own position") — confirmed wrong
  from the addon source, exactly the kind of assumption this task's Step A was designed to catch.

- **Combining into a final row / matrix** — `measurements.py::build_J_regressor_with_reporters`
  (lines ~102-143): rows for the model's 55 already-trained joints are copied **verbatim** by
  name from the stored `J_regressor`; only `b_h_l`/`b_h_r` get a freshly computed
  inverse-distance row via `export_J_regressor_to_npy(..., n=10, influence_type="inverse_distance")`
  (line ~118, `n=10` hardcoded at the `export_joint_distances` call site, line ~161). Reporter
  positions every frame are `J_regressor @ vertex_positions`
  (`measurements.py::recalculate_joint_positions`, lines ~674-694) — the same pattern as
  `joints = J_regressor @ verts` in `smal_model/smal_torch.py:394-399`.

- **`b_h_l`/`b_h_r` rest-pose position** — no Blender access from this environment, so recovered
  by trilateration from the CSV's own `Base` rows (exact pairwise Basis-pose distances) against
  the 55 trained joints' Basis positions, which are exactly `J_regressor_trained @ v_template`
  (confirmed **byte-identical** to the CSV's `Base` distances for trained pairs, e.g.
  `b_t`–`b_a_5`: computed `0.9048573021251339` == CSV `0.9048573021251339`). With 55
  overdetermined anchors, linear least-squares trilateration residual is **~3.6e-9 model units**
  — essentially exact, not an approximation:
  - `b_h_l` → `[0.40507574, 0.16124934, 0.06171087]`
  - `b_h_r` → `[0.39643314, -0.16170239, 0.05901498]`

## 2. Rest-pose vertex data

`diagnostics/atta_reference/blender_bundle/OmniAnt_25PCs_joint_limited.pkl` (the production model
per project memory), `d["v_template"]`, shape `(10235, 3)` float64 — this is the Basis mesh the
addon's nearest-vertex search reads. `d["J_regressor"]` (55, 10235), `d["J_names"]` (55,) give the
model's own trained joints for trilateration anchoring and for the cross-checks in §4 below.

## 3. Implementation (Step B)

`x2_head_width_reporter.py::build_reporter_matrix()` computes the frozen `(2, 10235)` matrix `R`
once (row 0 = `b_h_l`, row 1 = `b_h_r`), using the addon's own `float32` dtype for the
nearest-neighbor search itself (trilateration anchors kept at float64 for precision). `head_width`
lives on `HeadWidthReporter`:

```python
def head_bone_pos(self, verts):   # (2, V) @ (V, 3) -> (2, 3), or batched
    return torch.einsum("jv,...vc->...jc", self.R, verts)

def head_width(self, verts):
    pos = self.head_bone_pos(verts)
    return torch.linalg.norm(pos[..., 0, :] - pos[..., 1, :], dim=-1)
```

Differentiability confirmed two ways (script output):
- `.backward()` on a toy random `(V, 3)` input populates `.grad` on exactly the 20 vertices `R`
  has nonzero weight on (10 per side), grad norm 0.891357 — nonzero, sparse, exactly where
  expected.
- `torch.autograd.gradcheck` on the restricted 20-vertex subproblem: **passed** (`eps=1e-6,
  atol=1e-4`).

## 4. Validation against the Blender ground truth (Step C, the gate)

CSV confirmed to exist at the expected path:
`diagnostics/atta_reference/blender_export/OmniAnt_25PCs_joint_limited_joint_distances.csv`
(33,516 rows, 57 joint names including `b_h_l`/`b_h_r`, 21 "shapes": `01.obj`–`20.obj` + `Base`).

Posed/fitted vertices: `diagnostics/atta_reference/blender_bundle/ATTA20_ARM_A.npz["verts"]`
(20, 10235, 3) — confirmed **identical** (`np.allclose`) to
`diagnostics/atta_reference/runs/ARM_A_master/Stage_3_deform_fine.npz["verts"]`, the fitter output
that was actually imported into Blender to produce this CSV (per
`REPORT_ATTA_HEADWIDTH.md`'s reproducibility appendix). `verts` already includes `deform_verts`
baked in (`fitter_3d/trainer.py:349`: `verts = verts + _deform_verts`), so no additional term
needs to be added.

### 4a. Algorithm-only check (Base/rest pose) — **PASS, 0.0047%**

Running the torch `head_width()` on `v_template` itself (no specimen fit involved at all):

```
predicted = 0.32306340
csv Base  = 0.32307861
rel_err   = 0.0047%
```

This isolates the *formula* from any question about the posed vertex source, and it is correct to
five decimal places.

### 4b. Per-specimen check (posed/fitted `verts`) — **FAIL**

| specimen | pred | gt (CSV) | abs err | rel err % |
|---|--:|--:|--:|--:|
| 01 | 0.31622 | 0.33204 | -0.01582 | -4.764 |
| 02 | 0.38237 | 0.35352 | +0.02884 | +8.159 |
| 03 | 0.32797 | 0.33387 | -0.00590 | -1.766 |
| 04 | 0.31792 | 0.33862 | -0.02070 | -6.112 |
| 05 | 0.35493 | 0.33201 | +0.02291 | +6.902 |
| 06 | 0.33413 | 0.31976 | +0.01437 | +4.492 |
| 07 | 0.35167 | 0.33193 | +0.01974 | +5.946 |
| 08 | 0.32861 | 0.32409 | +0.00451 | +1.393 |
| 09 | 0.31851 | 0.33967 | -0.02116 | -6.231 |
| 10 | 0.28893 | 0.30462 | -0.01568 | -5.148 |
| 11 | 0.28736 | 0.27470 | +0.01265 | +4.606 |
| 12 | 0.26900 | 0.27161 | -0.00261 | -0.962 |
| 13 | 0.30559 | 0.30353 | +0.00206 | +0.677 |
| 14 | 0.32601 | 0.31172 | +0.01428 | +4.582 |
| 15 | 0.36905 | 0.35261 | +0.01644 | +4.663 |
| 16 | 0.32578 | 0.34086 | -0.01508 | -4.424 |
| 17 | 0.36777 | 0.36309 | +0.00468 | +1.289 |
| 18 | 0.33828 | 0.35312 | -0.01483 | -4.201 |
| 19 | 0.35936 | 0.35584 | +0.00351 | +0.987 |
| 20 | 0.39041 | 0.37387 | +0.01654 | +4.425 |

**mean |rel err| = 4.086%, max |rel err| = 8.159%, only 3/20 specimens within ≤1% (12, 13, 19).**

**Bar (§7.1, ≤1% relative error per specimen): NOT MET.** Verdict: **FAIL.**

## 5. Diagnosis — vertex-source mismatch, not a reporter bug

Three independent checks isolate the cause to the **posed vertex source**, not the reporter
matrix `R` or the replicated formula:

**(a) At Base/rest pose, error is exactly 0.0% for every bilateral pair tested**, not just for
`b_h_l`/`b_h_r`:

| pair | pred | CSV Base | rel err |
|---|--:|--:|--:|
| ma_l–ma_r | 0.244710 | 0.244710 | 0.000000% |
| w_1_l–w_1_r | 0.103494 | 0.103494 | 0.000000% |
| an_3_l–an_3_r | 0.688417 | 0.688417 | 0.000000% |
| b_t–b_a_5 | 0.904857 | 0.904857 | 0.000000% |

If `R` (or the trilaterated position feeding it) were wrong, this would not be exact — it is,
to full float64 precision. `R` is correct.

**(b) The same inflation appears on TRAINED joints using their VERBATIM-correct regressor rows**
(no reporter machinery involved at all — these are the model's own `J_regressor` rows, taken
straight from the pkl), evaluated on `ATTA20_ARM_A.npz["verts"]` vs. the per-specimen CSV rows:

| pair | mean rel err (posed) | max rel err (posed) | Base distance (model units) |
|---|--:|--:|--:|
| ma_l–ma_r | 4.854% | 10.041% | 0.2447 |
| w_1_l–w_1_r | 13.821% | 55.246% | 0.1035 |
| an_3_l–an_3_r | 6.312% | 12.920% | 0.6884 |
| b_t–b_a_5 (body length) | 0.095% | 0.240% | 0.9049 |
| **b_h_l–b_h_r (this reporter)** | **4.086%** | **8.159%** | 0.3231 |

The reporter's error (4.09% mean) sits squarely inside the range already produced by
**verbatim-correct trained-joint regressors** on small bilateral distances (0.10–0.69 model
units), and is *not* an outlier relative to them. `b_t`–`b_a_5` (large distance, 0.90 units) is
two orders of magnitude more accurate than the small bilateral pairs — consistent with a roughly
constant *absolute* per-vertex discrepancy (~0.01–0.03 model units between what
`ATTA20_ARM_A.npz["verts"]` holds and whatever mesh state actually produced this CSV) being
divided by a much smaller denominator for small pairs, inflating the relative error.

**(c) A whole-skeleton reproduction using this same `verts` source and the verbatim trained
regressor** (all 55×54/2 pairs, all 20 specimens, 29,680 total pairs) gives median 1.164%, mean
2.496%, max 304.6%, 85.5% of pairs within 5% — the same *shape* of distribution
`REPORT_ATTA_HEADWIDTH.md` itself reports for its own whole-skeleton check (median 1.05%, mean
1.59%, max 66.76%, 96.0% within 5%), just somewhat worse across the board. This confirms
`ATTA20_ARM_A.npz["verts"]` is *close to* but **not byte-identical to** whatever vertex state
Blender actually evaluated for this specific CSV export — plausibly a slightly different run/save
of the same fit (the two files are stamped ~12h apart: `blender_bundle/` 00:01, `blender_export/`
12:47 the same day), not a bug in this Step 0 implementation.

**Conclusion:** the reporter formula (Step A/B) is verified correct — exact at Base pose, gradient
sanity-checked, `gradcheck`-passed. The FAIL is caused by the *specific* `ATTA20_ARM_A.npz` snapshot
used here not being the exact vertex state the CSV was generated from; small bilateral distances
(head width included) are the most exposed to this by simple denominator size, as shown by (b)
above using zero reporter-specific code.

## 6. Next steps to unblock Step 1

1. Locate (or regenerate) the exact `verts` array that was actually imported into
   `bone_placement.blend` for this CSV export — check whether a fresher/later fitter run exists
   than `ATTA20_ARM_A.npz`'s Sep 6 00:01 timestamp, given the CSV/blend pair is stamped 12:47 the
   same day. If a matching run can be found or re-run deterministically, re-validate §4b with it.
2. If no matching source can be recovered, the practical floor for this check is whatever floor
   (a)/(b)/(c) above establish for *any* small bilateral pair, not specific to this reporter —
   worth flagging to whoever owns the ≤1% bar, since it may be systematically unreachable for any
   quantity this local given the presently-available vertex source, regardless of which two
   points are chosen.
3. Do **not** proceed to Step 1 on `bench50_clean`/production fits until either (1) resolves this,
   or the bar is explicitly revised with the evidence above — building the loss term on an
   unvalidated measurement is exactly what §7.1 of the preregistration forbids.

## 7. Status

**Step 1 (the `l_allo` loss term) remains BLOCKED.** The reporter algorithm itself (Step A/B) is
verified correct at rest pose and is differentiable; what is not yet established is that the
available posed-vertex source reproduces the Blender ground truth to the pre-registered tolerance,
and per §5(b)/(c) this looks like a property of the `ATTA20_ARM_A.npz` snapshot rather than of the
reporter. This is a genuine, diagnosed FAIL, not a fudged tolerance — per this project's CLAUDE.md
rule to state failures clearly rather than proceed past them.
