# Two addon defects found during the replication

## 1. Needs filing — `export_mesh_measurements` scaling factor is stuck at rest pose

**Symptom.** In `SMPL_Object_measurements.csv`, all 20 Atta scaling factors imply the *same*
`b_t`–`b_a_5` distance:

```
implied distance, all 20 shapes      : 0.845135   (one distinct value)
Base rest-pose distance              : 0.845135   ← exact match
correct per-shape values             : 0.5903 – 0.9197 (20 distinct)
```

**Cause.** Inside `export_mesh_measurements`, the two paths disagree about which mesh to read:

| Path | Reads | Correct? |
|---|---|---|
| surface area / volume | `obj.evaluated_get(depsgraph)` | yes |
| scaling factor | `obj.data.vertices` | **no** — unevaluated |

Setting `key.value = 1.0` does not move `obj.data.vertices[i].co`; shape keys only affect the
evaluated mesh. So the scaling path re-reads Basis on every iteration.

**Impact.** Surface-area and volume **"Scaled Value"** columns are off by 8–43% depending on
specimen. The raw unscaled `Value` columns are correct.

**Not affected.** `export_joint_distances` already reads the evaluated mesh, so
`SMPL_Object_joint_distances.csv` — and everything derived from it, including the head-width
numbers in this replication — is unaffected. Verified directly.

**Fix.** Read the evaluated mesh in the scaling path, as the sibling path already does.

---

## 2. Already tracked — no action

`export_joint_distances` raises `UnboundLocalError: J_regressor` for any model without a
`static_joint_locs` key: the regressor is computed only when static, read only when
non-static. This is [issue #92](https://github.com/FabianPlum/SMILify/issues/92) item A1.
Noting only that `force_static_joint_locs = True` is not a usable workaround — that path reads
`bone.head_local`, which shape keys don't move, so every specimen reports identical distances.
Fixed locally on our branch; happy to open a PR.
