# The remaining gap, and the SOTA route that closes it

> **UPDATED 2026-09-08 after R6.** An existing in-repo candidate — Fabian's SDF part prior — was
> tested on real scans before committing to anything here. Its *field* transfers; its *instance
> recovery* does not (2/12 specimens, mandibles merging left-into-right on 9/12). See
> `RESULTS_R6_20260908.md`. **The blocker is laterality, not segmentation**, which narrows the
> problem below: a segmenter must separate left from right structures that are physically connected
> in the scan, and no connectivity-based method can. Read the plan below with that constraint.

**The gap, stated precisely.** R3/R4 showed the fix is **one bit per landmark: which anatomical part
should reach this location.** Supplying it moves the named parts from 34.7% of a body length off to
**0.2%**, for 1.25× chamfer. Everything else — capacity, the two channels, the measurement
convention — is settled. What is missing is a way to produce that bit **without a human**.

So the capability needed is not dense correspondence. It is:

> **Given a raw ant scan, label its surface by anatomical part.**

That is a **part segmentation / label transfer** problem, and it is far smaller than the
vertex-correspondence problem the W-series attacked and failed to solve.

---

## What the field currently offers

**Zero-shot 3D part segmentation is now mature.** SAMPart3D segments any 3D object into semantic
parts at multiple granularities without predefined label sets, using a point-cloud encoder
pretrained on Objaverse that distils DINO-v2 features. P3-SAM (Tencent Hunyuan, 2025) does native
3D part segmentation. GeoSAM2 adapts SAM2. MeshSegmenter and SATR work through rendered views.

**The catch for us:** these produce *unnamed* segments at generic granularity. We need `ma_r`
distinguished from `ma_l` — a named, bilateral, anatomically specific label. Nothing zero-shot will
give that.

**Label transfer is the standard bridge.** Co-segmentation transfers part labels from a labelled
source shape to an unlabelled target by estimating corresponding locations. That reduces our problem
to matching ~20 discovered segments onto 51 named template parts — a small assignment problem, not
dense correspondence.

**And there is a route that fits this project better than either.**
*Learning Part Segmentation from Synthetic Animals* (WACV 2024) trains part segmentation on
synthetic animals and handles the sim-to-real gap with spectral data mixing. **We have the
generator.** SMILify's own model produces arbitrary ants with **exact** part labels for free — the
dominant skinning weight gives a ground-truth part per vertex, on unlimited poses, shapes and
species. This is the same loop the project already has with replicAnt/WOLO.

---

## The proposed route

```
SMILify model  ──►  render/sample synthetic ants with EXACT part labels   (free, unlimited)
                              │
                              ▼
                    train a part segmenter
                              │
                              ▼
   real scan  ──►  per-vertex anatomical part label  ──►  R3's named-part supervision
                                                                   │
                                                                   ▼
                                                    anatomically correct fit
```

**Why this is unusually well-posed here, and why it is worth doing rather than another diagnostic:**

1. **Both endpoints are already measured.** Baseline 34.7% (free supervision, R3), ceiling **0.2%**
   (perfect part labels, R3). Any segmenter can be scored directly against a known upper and lower
   bound — a position most method papers do not have.
2. **Training data is free and exact.** No annotation, no label noise, arbitrary quantity.
3. **The required precision is known and low.** R1 showed localisation to ≈5% of the template
   diagonal captures the full benefit; exact vertex identity is unnecessary and costs 20% more
   surface distortion. A segmenter need only be roughly right about *which part*.
4. **The failure mode is known.** Sim-to-real. Real scans are fragmented, damaged and noisy in ways
   synthetic renders are not — which is exactly what the WACV work addresses and what this corpus
   has in abundance.

---

## Sequencing, and what is realistically achievable

**Step 1 — feasibility, ~1 day, no training.** Generate synthetic ants from the model with known
part labels, and test whether a simple geometric classifier separates the anatomical parts at all on
*synthetic* data. If parts are not separable even without the sim-to-real gap, the route is dead
cheaply. Expect the anterior structures (mandible vs scape) to be the hard case, and the distal
antennal segments to be unusable — `an_3_r`/`an_3_l` carry 4 vertices each in the template.

**Step 2 — sim-to-real, the real risk.** Apply the segmenter to the 12 annotated real scans and
score against the known part assignment. This is where the project's existing evidence says the
difficulty lies: geometry alone was never able to establish semantic identity (W-series), and scans
are fragmented.

**Step 3 — closing the loop.** Feed predicted labels into R3's named-part supervision and measure
against the 34.7% / 0.2% bracket.

**Honest scoping:** Step 1 is a day. Steps 2–3 are weeks, not days — a segmenter, a sim-to-real
strategy, and evaluation. **This is a handoff plan, not something that lands before the internship
ends.** What the internship delivers is the diagnosis, the quantified bracket, and the fact that the
required precision is low — which is what makes the build tractable for whoever picks it up.

---

## What I would not do

- **Not dense vertex correspondence.** W3/W5 attacked it and V7 later showed the target is
  ill-posed — no template vertex is one anatomical point across these species.
- **Not an off-the-shelf zero-shot segmenter alone.** It gives unnamed generic parts; the naming is
  the whole difficulty.
- **Not more landmark-loss variants.** R3 settled what supervision is worth; the open question is
  its source.

---

## Sources

- [SAMPart3D: Segment Any Part in 3D Objects](https://arxiv.org/pdf/2411.07184)
- [P3-SAM: Native 3D Part Segmentation (Tencent Hunyuan, 2025)](https://arxiv.org/pdf/2509.06784)
- [GeoSAM2: SAM2 for 3D Part Segmentation](https://arxiv.org/pdf/2508.14036)
- [Learning Part Segmentation from Synthetic Animals (WACV 2024)](https://openaccess.thecvf.com/content/WACV2024W/CV4Smalls/papers/Peng_Learning_Part_Segmentation_From_Synthetic_Animals_WACVW_2024_paper.pdf)
- [3x2: 3D Object Part Segmentation by 2D Semantic Correspondences](https://arxiv.org/pdf/2407.09648)
- [Foundational Models for 3D Point Clouds: A Survey and Outlook](https://arxiv.org/pdf/2501.18594)
