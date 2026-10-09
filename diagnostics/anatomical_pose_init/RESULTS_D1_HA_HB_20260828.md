# H_A / H_B result — mechanism confirmed, but the fix is not shippable as-is

Bar fixed in `PREREGISTRATION_D1_coxal_degradation_20260828.md`, committed `4de6cc6c` before either
job ran. n=48 P48, every arm re-running D1 from the same `C13_uniform_hier/H2_joint.npz` handoff
(co = 0.2564). Baseline is `C13_uniform` (co = 0.4115); the gap to recover is 0.1551; the bar is
≤ 0.3030 with sign p < 0.05, read against the E0 floor of 0.0908.

## H_B — all six FAIL, as predicted

| arm | leg_acc | co | Δ | recovery | sign p |
|---|---|---|---|---|---|
| `D1_no_offset` | 0.9436 | 0.3892 | −0.0224 | 14.4% | 0.013 |
| `D1_no_edge` | 0.9422 | 0.4056 | −0.0060 | 3.8% | 0.31 |
| `D1_no_lap` | 0.9439 | 0.4069 | −0.0047 | 3.0% | 0.67 |
| `D1_no_sym` | 0.9440 | 0.4102 | −0.0013 | 0.8% | 0.47 |
| `D1_no_midline` | 0.9411 | 0.4112 | −0.0004 | 0.2% | 0.67 |
| `D1_no_deform` | 0.9409 | 0.4274 | **+0.0158** | −10.2% | 0.67 |

No single D1 term is throwing the coxa away. `w_offset` is the only one with a statistically real
effect (sign p = 0.013) and it recovers 14%, below even the 0.05 PARTIAL floor. The
pre-registration recorded the prediction that none would clear the bar **before these ran**, so
this is a confirmed prediction, not a rationalised null. `D1_no_deform` also excludes
`deform_verts` as the culprit — removing it makes the coxa *worse*.

## H_A — clears the bar, trips the guard-rail, and the guard-rail is correct

| run | leg_acc | co |
|---|---|---|
| H2 handoff | 0.9649 | 0.2564 |
| `C13_uniform` | 0.9415 | 0.4115 |
| `D1_with_cse` | 0.9728 | **0.2142** |

co 0.4115 → 0.2142, **127% recovery**, 46/48, sign p = 0.0000, Wilcoxon p = 6.25e-13. On the
endpoint alone this is CONFIRMED by a wide margin.

It also beats the H2 handoff it started from, which the pre-registration named in advance as a
**suspected bug, not a win** — D1 cannot have more anatomical information than H2 did. The
mechanism check says what happened:

| | `C13_uniform` | `D1_with_cse` |
|---|---|---|
| mean \|`deform_verts`\| | 0.001665 | **0.004179** (2.5×) |
| coxa `joint_rot` error vs GT | 0.3024 rad | **0.3183 rad (worse)** |

**The correspondence term is being satisfied by free-form per-vertex deformation, not by moving the
skeleton.** The surface lands in the anatomically right place while the underlying pose gets
slightly worse. This is exactly the failure mode written into the port's own code comment before
the run: in D1 the term acts on the *deformed* surface (`scheme: all`), whereas every hierarchical
arm ran `--deform_its 0` and the term acted on the posed-only surface. Same formula, same flag
name, different surface.

**Verdict: H_A confirms the mechanism — D1's coxal regression is caused by dropping the
correspondence term, not by any term it adds — and delivers a real surface-correspondence gain
that is NOT an anatomical one. It is not shippable as "the fit is anatomically correct".**

The obvious shippable variant is the term with `deform_verts` disabled. That is a **new thread with
its own pre-registration**, not a continuation of this one: the diagnosis here is already made from
two numbers moving in opposite directions, and re-running would confirm a conclusion rather than
test one.

## The pattern this is the third instance of

Three separate times this investigation, a component has improved on its own measure while the
thing that measure was standing in for did not:

1. **C11/C12** — hard-negative mining improved direct retrieval; the fitter saw nothing.
2. **bench50's framing** — geometric fit (chamfer/fscore/normal consistency) improved without any
   check that correspondence was correct, because none existed for real scans.
3. **H_A (here)** — surface-label correspondence improved while skeletal pose got worse.

Each is the same shape: an optimizer or a metric finds a cheaper way to satisfy the proxy than the
mechanism the proxy was chosen to represent. This is a recurring property of this system, not three
coincidences, and every future arm should carry a mechanism check alongside its endpoint —
H_A cleared its endpoint by 127% and was still wrong.
