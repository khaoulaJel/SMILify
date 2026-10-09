# Correspondence in SMILify — where it stands, and the one decision that needs you

2026-08-28. Consolidates the session; supersedes nothing, links everything. Every number here is
from a committed run with its bar written before the run.

---

## 1. The headline: predicted correspondence helps real ants, broadly

A CSE-style correspondence head (trained purely on synthetic data, run on real scans at inference)
was added to the fitter as a dense term.

**Across the whole bench50 corpus: 44/50 specimens improved, mean fscore@0.01 +0.067,
sign p = 3.2e-08.** On the cleanest 14: fscore 0.713 → 0.787 (14/14), normal consistency
0.830 → 0.879 (14/14), Hausdorff-95 improved 14/14. The worst 14 (degradation 5.2–12×) still gained
+0.054 — 74% of the clean-set gain.

The useful finding underneath: **scan degradation predicts how hard a scan is to fit, not whether
correspondence helps it.** Those are separate axes. Usable scope is **~94% of the corpus**, not a
screened subset.

**The honest limit, stated first rather than buried:** bench50 has no correspondence ground truth,
so this is *geometric fit* — "did the fit get closer to the scan", never "is correspondence
correct". Closing that gap needs human-annotated ground truth, which does not exist yet.

## 2. The deep-dive: the coxa, the hardest problem in the investigation, is fixed

The coxa resisted **four** separate mechanisms, each pre-registered and each failing:

| mechanism | effect on coxal error |
|---|---|
| correspondence alone | −0.009 |
| gradient reweighting by measured tolerance (C13) | −0.010, saturating |
| pose initialization from ground truth (C14p) | −0.009, n.s. |
| pose **frozen** at ground truth, shape free (E1) | −0.005, n.s. |

The cause turned out to be structural: **the D1 refinement stage had no correspondence term of any
kind.** The hierarchical stage reached co = 0.2564 and D1 relaxed it back to 0.4115 — it was
dropping the term that held the coxa, not adding one that pushed it.

**The fix — carry the correspondence term into D1, with free-form deformation disabled:**

| | before | after |
|---|---|---|
| `co` leg-level error | 0.4115 | **0.2265** (~119% of the regression recovered, 43/48, p = 0.0000) |
| coxa surface placement vs GT | 0.01857 | **0.00794** (2.3× closer, 45/48, p = 0.0000) |
| `leg_acc` | 0.9418 | **0.9682** (~52% of the remaining gap to the 0.9926 ceiling) |

Replicated on an independent seed, with `deform_verts` pinned at exactly 0 — so it is not bought by
free-form surface deformation. Recommended configuration: `scheme: pose`,
`--cse_correspondence_from`, **no** `w_beta_prior`.

## 3. → The decision that is yours, not another experiment's

**The fix costs shape accuracy: `betas` error vs ground truth rises +16.3%.**

This is characterized, not a mystery:
- It is **not controllable by the shape prior.** Adding `w_beta_prior = 0.002` made it *worse*
  (+60.8%), because the Mahalanobis prior pulls `betas` toward *typical under the model*, not
  *correct for this specimen*. On a corpus with sampled shape parameters those directions diverge.
- Across two independent arms it is **uncorrelated** with the coxa gain (r = +0.13, p = 0.39) —
  consistent with an independent drift rather than shape paying for placement.

**The question: is a 16.3% shape-accuracy cost acceptable in exchange for the coxa fix above?**
That depends on how SMILify's downstream users actually consume shape parameters, which is your
call. No further experiment on this side resolves it.

## 4. Closed negatives, each with a transferable lesson

Not loose ends — each closed against a pre-registered bar, each leaving something reusable.

- **Equivariant retrain: not justified. Do not spend the GPU budget.** A linear probe on the frozen
  backbone recovers circumferential angle at **52.9°** against a 90° chance level, on the exact
  tensor the embedding head consumes — with a passing axial control (R² 0.85) and a passing shuffle
  control (89°). The signal reaches the head; max-pooling does not destroy it. Meanwhile retrieval
  top-1 is 0.071. **The information is present and the contrastive loss is not using it** — a
  loss/objective problem, far cheaper to attack than a new backbone.
- **Multi-hypothesis selection: closed.** A trained 1-of-5 selector combining similarity margin,
  cycle-consistency and local agreement closed 2.5% / 4.2% of the oracle gap against a 40% bar —
  worse than cycle-consistency alone. The lesson is specific and non-obvious: **rank accuracy is
  not a proxy for error reduction.** The classifier picked the best candidate ~50% more often than
  chance and it bought almost nothing, because the cases it fixed and broke roughly cancelled.
- **The coxa is not a model-expressiveness limitation.** A perfect parameter set scores 0.0908 —
  the metric's own floor, since adjacent coxae are close enough that ~9% of coxal points misassign
  even on an exact mesh. Representability was never the problem.

## 5. The pattern worth carrying forward

Six times this session, a component improved on its own measure while the thing that measure stood
for did not:

1. hard-negative mining improved retrieval; the fitter saw nothing
2. bench50's original framing improved geometric fit with correspondence never checked
3. carrying correspondence into D1 improved surface metrics 127% while pose worsened *(later shown
   to be a mis-specified check, not a real decoupling — see §2)*
4. a probe produced a plausible result from a network that **never loaded** (178 missing keys,
   silently)
5. a selector beat chance on rank accuracy and delivered no error reduction
6. a shape prior "controlling" a cost made it four times worse

**Operational consequence, now standing practice: every arm carries a mechanism check alongside its
endpoint.** The D1 arm cleared its endpoint by 127% and was still wrong; the probe in (4) produced
a believable pattern from random weights. Endpoints alone are not evidence.

## 6. Next steps — offered, not started

- **Human joint annotation on real scans** (tooling drafted, 10 specimens scoped). This is the only
  thing that can turn §1 from "the fit gets geometrically closer" into a correspondence claim.
  Note it yields *joint positions*, a pose-accuracy measure — **not** `leg_acc`/`seg_acc`, which are
  surface-label metrics and need a different definition to build from annotations.
- **Mean-teacher fine-tuning on bench50** — the last untried lever with genuine upside on real
  data, fully pre-registered but deliberately not started. It should follow your decision in §3,
  not precede it.

---

*Every result above is committed with its pre-registration timestamped before its run. Where a bar
was met by the letter but the instrument was later found wrong, both are recorded — the verdict was
never rewritten after the fact.*
