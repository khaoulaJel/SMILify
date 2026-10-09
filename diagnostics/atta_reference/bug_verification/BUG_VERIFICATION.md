# Bug verification: are the two reported defects real?

Adversarial re-check of two claims made during the Atta head-width replication. Both were
originally asserted from *reading* code, which is not evidence. This document tests each one
against the standard: **a minimal reproducible case with observed wrong output**, plus checks
for "already known", "used it wrong", and "intentional behaviour".

## Verdicts

| # | Claim | Real bug? | Already reported? | Action |
|---|---|---|---|---|
| 1 | `export_joint_distances` — inverted `static_joint_locs` guard | **Yes**, reproduced | **Yes — upstream issue [#92](https://github.com/FabianPlum/SMILify/issues/92) item A1, filed by the maintainer** | **Do not send.** Reference #92 A1. |
| 2 | `SMPL_Object_measurements.csv` — scaling factor stuck at rest pose | **Yes**, observed wrong output | **No** — not in #92, not in any issue | **Send.** Genuinely new. |

### Correction to an earlier claim of mine

In the draft bug report I originally wrote that commit `6cde0d34` added the guard "in six
places, five as `if not static_joint_locs:` and this one inverted." **That is factually
wrong.** The commit added 4 positive guards and 2 negative ones, and each is correct for its
own context. The sign count proves nothing. The actual argument is narrower and stronger, and
is given in §1.3 below. The draft has been corrected.

---

## 1. `export_joint_distances` inverted guard

### 1.1 Read the condition literally

`upstream/master:3D_model_prep/SMIL_processing_addon.py`, function `export_joint_distances`
(lines 1868–2057). AST walk of every `J_regressor` reference inside the function:

```
line 1886  ASSIGNED   J_regressor = export_J_regressor_to_npy(mesh_obj, armature, 10, ...)
line 1955  READ       joint_positions = recalculate_joint_positions(vertex_positions, J_regressor)
line 2000  READ       joint_positions = recalculate_joint_positions(vertex_positions, J_regressor)
any 'global J_regressor' declaration? False
```

Line 1886 sits inside `if mesh_obj.get("static_joint_locs", False):`. Lines 1955 and 2000 sit
inside the corresponding `else:` branches. There is no `global` declaration, so Python binds
`J_regressor` as a function-local; reading it on a path where it was never assigned is an
`UnboundLocalError` by language semantics, not by interpretation.

### 1.2 Minimal reproducible case — observed output

`repro_bug1_j_regressor.py` extracts the **real** function source from `upstream/master` via
AST (`ast.get_source_segment`) and executes it with `bpy` stubbed. The function body is not
modified. Two runs differ **only** in the value of one custom property:

```
$ python repro_bug1_j_regressor.py

--- A. static_joint_locs = True : obj props = {'static_joint_locs': True} ---
  RESULT: returned ok=True  msg='Distances exported to /tmp/_repro_out.csv'

--- B. static_joint_locs absent (the production model's actual state) : obj props = {} ---
  RESULT: RAISED UnboundLocalError: local variable 'J_regressor' referenced before assignment
```

`OmniAnt_25PCs_joint_limited.pkl` has no `static_joint_locs` key (verified by loading the pkl
and listing its keys), so `.get(..., False)` returns `False` and every real export of this
model takes path B.

The operator wrapper `SMPL_OT_ExportJointDistances.execute()` has no `try/except`, so the
error surfaces as an unhandled traceback in Blender's console rather than a clean error
dialog.

**Caveat, stated plainly:** this is a stubbed-`bpy` execution of the real function body, not a
live Blender run. It cannot be run live here — the cluster has no Blender. See §1.5, where an
independent party confirms the same failure in a real Blender session.

### 1.3 Is the guard intentional? (the real argument)

Three independent reasons it is not:

1. **The value is dead in the branch that computes it.** In the `static` branch, joint
   positions come from `bone.head_local` — `J_regressor` is computed and then never read on
   that path. A guard that gates a computation used *exclusively* by the opposite branch is
   the signature of an inverted condition, regardless of how many sibling guards share its
   sign.
2. **The comment above it describes unconditional intent:** *"Recalculate J_regressor for
   current mesh state using selected method / This ensures it works even if mesh topology has
   changed."* Nothing there is conditional on static joints.
3. **The line was unconditional before the guard was added** (`17837fc9`, 2025-10-06). The
   guard arrived in `6cde0d34` (2025-11-24, *"Fixed incorrect updating of joint locations and
   regressor eval"*).

### 1.4 Is `force_static_joint_locs = True` a valid workaround? (i.e. "did I use it wrong?")

No — the alternative path is also wrong, in a way that silently produces plausible numbers.
`repro_bug1b_static_workaround.py` runs the same real function with two shape keys whose
geometry differs by 3×:

```
'small' shape-key distances: [1.0, 2.0, 1.0]
'big'   shape-key distances: [1.0, 2.0, 1.0]   (geometry is 3x larger)
IDENTICAL despite 3x different geometry? True
```

In the static branch joint positions are read from `bone.head_local`, which does not change
with shape keys — so every specimen reports identical distances, and only the scaling factor
differs. That is useless for morphometrics. **There is no setting under which this workflow
works on master**, which is itself evidence the non-static path is the intended one.

### 1.5 Already reported — yes

**Upstream issue [#92](https://github.com/FabianPlum/SMILify/issues/92), "Blender addon:
inherited legacy bugs surfaced during the #88 review", opened by `FabianPlum` on 2026-07-24,
still open. Item A1:**

> **`export_joint_distances` crashes for every non-static model** (`measurements.py:115`,
> legacy L1885). The `static_joint_locs` condition is inverted: `J_regressor` is *computed*
> only when static, but *used* only when non-static → `UnboundLocalError`. Fix: compute when
> `not static` (and initialise `J_regressor = None`). Flagged by Copilot on #88; re-confirmed
> on #89.

Same defect, same diagnosis, same fix, already known to the maintainer — and independently
re-confirmed twice during PR review, which is stronger corroboration than the repro above.

Also checked: the guard is still present and unfixed on **every** upstream branch
(`master`, `paper-draft`, `feature/registration_moonshot`, `fix/resume-batch-size-mismatch`,
`ci/*`, `cleanup/*`, `chore/*`). Only `upstream/SMILy_Mouse` has the pre-regression
unconditional form. So this is not an "on an old version" situation — it is unfixed upstream
and known.

**Consequence: do not send this as a new bug report.** The correct move is to reference
#92 A1 — and to note that the local fix on this branch resolves it, if that is useful to the
maintainer.

---

## 2. `SMPL_Object_measurements.csv` scaling factor stuck at rest pose

### 2.1 Observed wrong output — from the maintainer's own 2025 data

No theory required for this one; the defect is visible in the shipped reference file.

```
1) Surface area DOES vary per shape key: 81 distinct values across 81 shapes
      01 -> 1.1635,  02 -> 1.4834,  03 -> 1.2030,  04 -> 1.3071 ...

2) implied b_t-b_a_5 distance across all 20 Atta shapes:
      1 distinct value  ->  [0.845135]

3) Base (rest-pose) b_t-b_a_5 in SMPL_Object_joint_distances.csv : 0.845135
   constant implied by SMPL_Object_measurements.csv               : 0.845135
   MATCH: True

   the per-shape values that SHOULD have been used range 0.5903 to 0.9197 (20 distinct)
```

Reconstruction method: `scaling_factor = reference_length / current_distance`, so
`current_distance = reference_length / scaling_factor`. Applying that to all 20 Atta rows of
`SMPL_Object_measurements.csv` yields **one** value, and that value is **exactly** the rest-pose
distance recorded in the sibling export. Not approximately — exactly.

This is the experiment "export two shape keys with known different volumes and check whether
the CSV reports the same number twice", already run at n=20 on real data. It reports the same
number twenty times.

### 2.2 Is it intentional rest-pose reporting? (the strongest counter-hypothesis)

No — refuted on three independent grounds:

1. **The docstring states per-shape-key intent:** *"Export mesh surface area and volume
   measurements to a CSV file, including measurements for each shape key."*
2. **The same file's other columns do vary per shape key** — 81 distinct surface-area values
   across 81 shapes. If rest-pose reference were the design, area and volume would be constant
   too. Only the scaling factor is frozen.
3. **The inconsistency is internal to one function.** Within `export_mesh_measurements`:
   - line **2208–2213**: the surface-area/volume path builds its temp object from
     `obj.evaluated_get(depsgraph)` — the depsgraph-evaluated (posed) mesh. Correct.
   - line **2180**: the scaling-factor path reads `obj.data.vertices` — the unevaluated mesh.

   Setting `key.value = 1.0` does not move `obj.data.vertices[i].co`; shape keys only affect
   the evaluated mesh. So the scaling path reads Basis on every iteration. Two code paths in
   the same function, describing the same shape key, disagree about which mesh to read. That
   is not a design decision.

For contrast, the sibling function `export_joint_distances` reads
`mesh_obj.evaluated_get(depsgraph)` correctly — which is why
`SMPL_Object_joint_distances.csv` is unaffected.

### 2.3 Blast radius

Surface-area and volume "Scaled Value" columns are wrong by 8–43% depending on specimen
(the ratio between the frozen 0.845135 and each specimen's true 0.5903–0.9197). The raw
unscaled `Value` columns are fine.

**This does not affect the Atta replication numbers.** Every comparison in
`REPORT_ATTA_HEADWIDTH.md` was made against `SMPL_Object_joint_distances.csv`, which uses the
correct evaluated-mesh path. Confirmed directly, not assumed.

### 2.4 Already reported — no

- Issue #92 mentions `export_mesh_measurements`, "surface area", "volume", "scaling factor",
  `obj.data.vertices` and `evaluated_get` **zero times**. Its item 6 concerns
  `export_posedirs`, a different function with a superficially similar evaluated-mesh theme.
- GitHub issue search for `export_mesh_measurements` across the repo returns one hit, PR #20
  ("Bind SMIL addon export to the selected mesh's armature"), which only lists the function
  name among four call sites being changed; it mentions scaling/evaluated/depsgraph/shape key
  zero times.

**Consequence: this one is worth sending.** It is unreported, it is in shipped reference data,
and it silently produces plausible-looking wrong numbers rather than failing loudly.

### 2.5 Outstanding step, if wanted

Everything above is code inspection plus the maintainer's own output file. Not yet done: a
live Blender re-run of `export_mesh_measurements` on two shape keys, confirming the scaling
column repeats. Given §2.1 is that experiment already run at n=20 on real data, this would be
confirmation rather than new evidence — but it is the one thing that would make the report
fully self-contained.

---

## Reproduction

```bash
git show upstream/master:3D_model_prep/SMIL_processing_addon.py > /tmp/upstream_master_addon.py
python diagnostics/atta_reference/bug_verification/repro_bug1_j_regressor.py
python diagnostics/atta_reference/bug_verification/repro_bug1b_static_workaround.py
```

| File | Purpose |
|---|---|
| `repro_bug1_j_regressor.py` | Executes real master `export_joint_distances`; static=True passes, static absent raises `UnboundLocalError`. |
| `repro_bug1b_static_workaround.py` | Shows the static path returns identical distances for 3×-different geometry. |
| `../measurements/SMPL_Object_measurements.csv` | Bug 2's observed wrong output (maintainer's 2025 data). |
| `../measurements/SMPL_Object_joint_distances.csv` | The correct sibling export; source of the 0.845135 rest-pose value. |

Environment: no Blender and no `gh` CLI on this cluster; GitHub checked via the REST API.
Upstream state as of 2026-09-06, `upstream/master` = `b5bf9565`.
