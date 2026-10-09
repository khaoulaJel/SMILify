# Pre-registration — W5: does W3b's coxa correspondence gain reach the fitter?

**Written and committed BEFORE the run**, before the correspondence npz was generated and before any
fitter arm was launched.

Date: 2026-08-30.

---

## 1. The hypothesis, and why it is worth testing despite a strong prior against it

W3b found a learned descriptor beats part-local rigid proximity on **`co` only** (+25.2%, CI
[+21.7%, +28.2%]), and nowhere else. The coxa is also precisely where SMILify's residual has been
concentrated. That gives an unusually specific hypothesis:

> **H: improving correspondence specifically in the coxa gives the optimizer enough additional
> anatomical information to improve the final fit.**

The prior against it is strong and quantified. Four pre-registered mechanisms have moved the coxal
error by almost nothing (correspondence alone −0.009, tolerance reweighting C13 −0.010, GT pose init
C14p −0.009, frozen GT pose E1 −0.005), and retrieval gains have failed to reach the fitter four
times (C11, C12, F2, and the mechanism-check series). **That is what makes the test informative
rather than redundant** — all three outcomes in §5 are worth knowing.

**No change will be made to the optimizer to make this work.** The question is whether the existing
optimizer can exploit what W3b provides.

## 2. The deployment gap — stated first, because it determines what the result can mean

W3b is **not** directly deployable in the fitter, and pretending otherwise would make the result
uninterpretable. Two mismatches:

1. **W3b has no key table.** It is a part-conditioned query-to-query matcher between two *posed
   specimens*; the fitter's term needs a per-template-vertex target. W5 therefore matches each
   target's points to **reference specimens of known vertex identity** (held-out members of the b2
   corpus, disjoint from W3b's training), within part, and aggregates.
2. **W3b requires part labels on both sides.** The fitter has none. W5 uses the target's **oracle**
   part labels, available because these are synthetic.

**Consequence, fixed now:** this arm is an **oracle-assisted upper bound**, not a deployable method.
Therefore the experiment is **decisive in one direction only**:

- **A negative is decisive.** If correspondence improved this way — with oracle part labels, i.e.
  strictly more information than any deployable version could have — still does not help the fitter,
  then the coxal residual lies downstream of correspondence, and no better descriptor fixes it.
- **A positive is NOT attributable.** A win could come from the oracle part labels rather than from
  W3b, and W5 will not claim otherwise. It would license a follow-up with predicted part labels, and
  nothing more.

Given the strong prior of failure, an experiment that is sharp on the negative side is the right
one to run.

## 3. Arms — the hybrid the W-series actually motivates

Everything is held identical to the established `D1_with_cse_nodeform` recipe (H_A3): same
`--mesh_dir synth_power48` (48 specimens), same `D1_no_deform.yaml`, same `--init_from
C13_uniform_hier/H2_joint.npz`, same seed.

| arm | correspondence fed to `--cse_correspondence_from` |
|---|---|
| **A (control)** | C3's `cse_p48_all.npz`, unchanged — the current best configuration |
| **B (experimental)** | C3's, with **coxa vertices only** replaced by W3b's |

The segment substituted is **`co` only**, and that choice is fixed **here, by W3b's held-out
retrieval result** — not by any fitter number. Substituting W3b everywhere would knowingly inject
*worse* correspondence on `tr`/`fe`/`ti`, where W3b loses; that is not a test of H, it is a test of
a method nobody proposes. This is also exactly the geometry-where-geometry-works /
learned-where-geometry-fails hybrid the W-series points at.

## 4. Endpoint and bar

Primary: **coxal leg-level error `co`**, arm B vs arm A, paired across the 48 specimens.

- **PASS** — `co` falls by **≥5% relative** with paired sign test **p < 0.05**, *and* `leg_acc` does
  not fall.
- **PARTIAL** — `co` falls but by <5%, or falls while `leg_acc` regresses.
- **FAIL** — `co` does not fall.

The 5% bar is set against the four prior interventions, which moved coxal error by roughly 2–4%.
Reported alongside, gating nothing: `leg_acc`, `seg_acc`, fscore, and **`betas` error** (the D1
coxa fix is known to cost +16.3% shape accuracy; W5 must not hide a further cost).

## 5. Decision tree — fixed now

| outcome | reading |
|---|---|
| correspondence improves **and** fitter improves | coxal correspondence is causally relevant — but see §2, not attributable to W3b alone |
| correspondence improves, fitter does not | **the coxal residual is downstream of correspondence.** Decisive. The descriptor line closes for the fitter, not just for retrieval |
| correspondence does **not** improve on these specimens | the retrieval result did not survive the actual deployment regime; reported as such, and the fitter arms are not run |

## 6. Mechanism checks — the first three void the run

1. **Training-set leak (VOIDS).** No `synth_power48` specimen may appear in W3b's training corpus.
   Checked empirically by exact vertex comparison against all 4000 b2 specimens, not assumed.
   (Pre-checked: 0/6 exact matches on a sample; the full check runs in the script.)
2. **The arms must actually differ (VOIDS).** Coxa entries of arm B's npz must differ from arm A's,
   and **non-coxa entries must be byte-identical**. If they do not differ, the arm tested nothing;
   if non-coxa entries differ, the substitution leaked and the comparison is not isolated.
3. **Correspondence gate (VOIDS the fitter runs).** Before any fitting, measure directly on these 48
   specimens whether W3b's coxal correspondence is actually better than C3's, against ground truth.
   If it is not, §5 row 3 applies and the fitter arms are **not launched** — spending GPU to fit with
   correspondence that is not better would answer nothing.
4. **Coverage.** Report the mask coverage of both npz files. A large coverage difference would
   confound the comparison, and is reported if present.

## 7. Not claimed

Anything about real scans. Synthetic, oracle part labels, exact vertex identity. And per §2, a
positive result is explicitly **not** claimed as a W3b result.
