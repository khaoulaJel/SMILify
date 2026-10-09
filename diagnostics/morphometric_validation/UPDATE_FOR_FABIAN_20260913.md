# Update for Fabian — what I did with the joint annotations, and what we learned

Hi Fabian,

Here's a short summary of what the annotations were used for once they came back, what they showed,
and why it matters for SMILify.

---

## Why the annotations mattered

Before the annotations, every check we had on SMILify measured **consistency**: does the fit agree
with itself, is it repeatable, does it look right? None of them measured **accuracy**, meaning
whether the joints land where an expert says they should. The annotations gave us the first real
ground truth.

We used two sets:
- **First pass:** 14 specimens.
- **Your expert-corrected pass:** 12 specimens. This is the one everything is now judged against.
  Most joints didn't move, but the ones that did moved about **20%**, concentrated at the
  **mandibles and coxae**, which are exactly the structures the investigation cares about most.

## What we found, step by step

**1. SMILify is only a little more accurate than doing nothing.**
Compared with your annotations, the fitted skeleton is only **~17% better than the unfitted
template** (the "average ant"). The worst regions are the distal legs and the head (the anterior
improves only ~7%). *All earlier checks had missed this, because they only measured consistency.*

**2. The fit is roughly unbiased, but it can't rank individual specimens.**
Against the expert set the fitter is close to unbiased on average; most of the earlier apparent bias
came from the first annotation pass. But per-specimen correlation with the truth is about zero.
**Averages over many ants are usable; saying "this ant has a longer X than that one" is not.**
Surface-based measurements carry about **2.8×** more genus signal than joint-based ones.

**3. The model *can* represent the correct anatomy.**
When we pulled the joints toward your annotations while keeping the shape in its normal range, joint
error dropped from **38.9% to 11%**. Adding a joint term to the fitting objective gives
**38.9% → 27.5%** for only ~19% more surface error, and **9%** at about 2.3×. So capacity isn't the
problem, and accuracy costs very little.

**4. Annotating only some joints doesn't fix the rest.**
Supervising a subset of joints barely helped the unsupervised ones. The benefit only becomes large
at **~75% coverage**. *Annotations are useful dense; sparse annotations don't propagate.*

**5. The annotations exposed wrong landmark definitions and fixed them.**
With your surface landmarks, the fitted surface was **7.7%** of Weber's length from the true point,
but the template vertex we had *named* as that landmark was **48.8%** away. The fit was fine; our
landmark definitions were wrong. Recalibrating them from your annotations removed **all 44
"impossible" mandible/head-length ratios** in the 757-specimen corpus.

**6. Skeleton and surface are two separate problems.**
Fixing the joints doesn't improve the surface anatomy, and fixing the surface doesn't improve the
joints. They need separate supervision. With both, the model holds all 12 landmarks at **0.2%**
error, so again, capacity isn't the problem.

**7. The key missing ingredient is anatomical *identity*.**
Simply telling the fitter which part is which moved the named structures from **34.7% to 0.2%**
error, consistently across 12/12 landmarks and 11/12 specimens. That's the biggest effect in the
whole investigation.

**8. It isn't a loss-weights problem.**
The current objective scores the anatomically correct solution **67% worse** than its own fit, so it
actively prefers the wrong answer. Removing two regularisers reverses that, *but only while the
annotations are present*. Without annotations, removing them lets the fit fall apart: measurement
error doubles and correspondence breaks. So retuning the loss won't solve it.

**9. Getting identity without annotations didn't work.**
We tried four annotation-free routes (a signed-distance part prior, midline and symmetry
constructions, and DINO image features). All of them failed, mostly by **confusing left and
right**. DINO features were even beaten by plain geometric proximity.

## The bottom line

> **SMILify has enough capacity to represent the correct anatomy, and anatomical constraints are very
> powerful, but fitting alone can't work out which part is which from the scan geometry.**

Because that can't be fixed quickly, I then turned the question around and asked **which
measurements can already be trusted** despite it. Head width comes out clearly on top: it's
repeatable, holds up on all 757 scans, and reproduces your WOLO scaling (slope 0.385 vs your 0.395).
Your annotations were used there too, to check that each measurement sits on the right anatomy.

## Why this is useful

- **First true accuracy numbers for SMILify**, instead of consistency numbers.
- **It pins down the real bottleneck (identity)** and rules out capacity, pose and loss weights, so
  those don't need to be tried again.
- **It fixed our landmark definitions** and removed a whole class of fake "impossible" traits.
- **It shows how annotations should be used:** densely, and for skeleton *and* surface separately,
  because sparse or joint-only supervision doesn't carry over.

## Limitations, stated upfront

- **12 specimens, one expert, one pass.** There is no annotator-repeatability measurement yet, so
  the distal-leg numbers are best read as an upper bound on fitter error.
- **Some early experiments were run on the first (uncorrected) set.** The key conclusions were
  re-checked against your corrected set, and everything above uses it.

## What would help most next

1. **Blind re-annotation of 2–3 specimens.** About an hour of work, and it gives every number above
   a proper error bar.
2. **More specimens for the joint-alignment benchmark.** 12 is thin for ranking fitting strategies.
   Should I wait for your new annotations, or benchmark on the current 12?
3. **Physical scale of the antscan meshes.** Implied scale disagrees by about 7× across scans, and
   `voxel_size` doesn't explain it. Do you know what unit the STLs are in?

---

Full write-ups are in `diagnostics/groundtruth/`, and the presentation figures are in
`diagnostics/morphometric_validation/figures/`.
