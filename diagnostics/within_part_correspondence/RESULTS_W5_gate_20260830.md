# W5 — GATE FAILS. W3b's coxa advantage does not survive the deployment regime, and the fitter arms were not run.

Decision tree fixed in `PREREGISTRATION_W5_fitter_propagation.md` **before** the run. Job 3349877
(rerun with the fairness control added after run 3349864 exposed a confound). 48 `synth_power48`
specimens, ground-truth scoring, GPU, 56 s.

---

## The gate

Coxal correspondence error against ground truth, rest space, lower better:

| method | error | vs W3b | sign p | better |
|---|---:|---:|---:|---:|
| **W3b** (oracle part label + reference-specimen vote) | **0.05864** | — | — | — |
| C3, full vocabulary (its deployed mode) | **0.03040** | W3b **−92.9%** | 7.1e-15 | 0/48 |
| C3, restricted to the oracle part | **0.02326** | W3b **−152.0%** | 7.1e-15 | 0/48 |
| **C3, restricted to W3b's SAME candidate subset** | **0.02393** | W3b **−145.1%** | 7.1e-15 | **0/48** |

**W3b is worse on every one of the 48 specimens.** Per the pre-registered decision tree (row 3),
the retrieval result did not survive the deployment regime, and **the fitter arms were not
launched** — fitting with correspondence that is not better would answer nothing.

## The confound I suspected was real, and it was not the explanation

After the first run I flagged that the comparison looked rigged in C3's favour: W3b's candidate set
is limited to the 4096 sampled reference vertices (its embeddings are only valid at that density —
W2's lesson), so **only 40.1% of target coxal points have their true vertex reachable at all**,
while C3 retrieves over all 10,235. Comparing those directly measures a handicap I imposed.

So I added the apples-to-apples arm: **C3 restricted to exactly the same candidate subset.** It
scores **0.02393** — essentially unchanged from C3's unrestricted 0.02326 (+2.9%). The candidate-set
handicap costs almost nothing.

That is consistent with F7's finding that embedding crowding is *entirely local*: being forced to
pick among 40% of the vertices barely hurts a 3D error metric, because the reachable neighbours are
geometrically adjacent to the true one. **The handicap was real but negligible, and it does not
explain a 145% gap.** W3b genuinely loses.

## Why the advantage did not transfer — the structural reason

W3b's +25.2% coxa win is real *within its protocol*, and that protocol is not what the fitter needs:

| | W3b's evaluation (W1–W3b) | what the fitter requires |
|---|---|---|
| query | a **vertex** of specimen A | a **surface point** of the scan |
| candidates | the **same sampled vertex indices** on specimen B | **all** template vertices |
| baseline | part-local **rigid proximity** | C3's key table |
| output | a specimen-B vertex | a per-template-vertex target array |

W3b is a part-conditioned **query-to-query** matcher between two posed specimens. It has **no key
table**, so template identity has to be smuggled in via reference specimens — and a
density-limited candidate set cannot populate a 10,235-vertex target array. The symptom is in the
coverage line: W3b fills **1.8%** of coxa template vertices against C3's **28.1%**. Its votes
collapse onto a handful of vertices — a degenerate assignment, the same failure mode F4's mechanism
check was written to catch.

**The honest conclusion: W3b's advantage was measured against a weak baseline (rigid proximity) in a
framing that does not define a deployable correspondence.** C3 — which W3b beat by 46–68% on the
W-series protocol — is *far better* than W3b at the thing the fitter actually consumes. Those two
statements are both true, and the second is the one that matters for SMILify.

## What this does to the W-series conclusions

**Unchanged:** W3b beats C3 by 46–68% *on the within-part specimen-to-specimen task*, and the
objective-mismatch finding behind it stands. That was always a statement about how to train such a
head, and W5 does not touch it.

**Substantially weakened:** the practical significance of the coxa result. I wrote that W3b's
+25.2% "could open the fitter question". W5 says it does not — not because the fitter failed to
exploit it, but because the correspondence never got good enough to be worth feeding in. The
pre-registration anticipated exactly this as row 3 and it is the outcome that landed.

**Not established, and explicitly not claimed:** that the coxal residual is downstream of
correspondence. That was row 2, and it required a gate pass to test. **W5 does not answer whether
better coxal correspondence would help the fitter** — it only shows W3b does not supply better
coxal correspondence. The four prior interventions remain the evidence on that question.

## Mechanism checks

| check | result |
|---|---|
| 1 — training-set leak (voiding) | **PASS** — 0 of 48 `synth_power48` specimens found in W3b's 4000-specimen b2 training corpus, by exact vertex comparison |
| 2 — arms isolated (voiding) | **PASS** — coxa entries differ, non-coxa entries byte-identical |
| 3 — gate (voiding for fitter) | **FAIL** — see above; fitter arms not launched |
| 4 — coverage | reported: A(C3) 29.4% overall / 28.1% coxa; B(W3b) 27.0% / **1.8%** coxa |
| — checkpoint load | W3b `strict=True`, 0 missing / 0 unexpected, `head.0.weight` byte-matched; C3 likewise |

## What I would take from this

The W-series now has a consistent shape. Plain geometry is a strong baseline wherever geometry is
identifiable; a learned descriptor beat it only on the coxa, only under oracle part labels, and only
in a specimen-to-specimen framing that does not survive contact with the fitter's interface.

**The remaining honest statement about SMILify's coxal residual is the one that was already true
before the W-series: four pre-registered interventions have moved it by ~2–4%, and nothing here
changes that.** No further descriptor variant is opened from this result.

**Not claimed:** anything about real scans. Synthetic, oracle part labels, exact vertex identity.

## Artifacts

- `w5_generate_and_gate.py` — leak check, correspondence generation, gate, arm-B construction.
- `submit_W5gate_20260830.sbatch`.
- `out_W5/w5_gate.json`, `w5_armB_c3_with_w3b_coxa.npz`, `w5_w3b_raw.npz` (arm B is built and on
  disk but **was not fitted**, per the gate).
- `sbatch_logs/W5gate_3349864.log` (first run, no fairness control), `W5gate_3349877.log` (final)
  — gitignored (`*.log`); all numbers are in `w5_gate.json`.
