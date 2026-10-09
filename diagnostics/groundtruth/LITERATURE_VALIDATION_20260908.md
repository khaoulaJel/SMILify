# Literature check: which of our findings the field already supports, which are ours, and where
# we diverge from what the SOTA actually claims

Searched 2026-09-08. Written to separate *"the literature agrees"* from *"the literature says
something adjacent that we should not claim as agreement"*.

---

## 1. VALIDATED, and strongly — landmark typology predicts our repeatability ordering

Geometric morphometrics has a formal landmark classification (Bookstein types). **Type III
landmarks — "constructed extremes", whose placement depends on other landmarks or on an extremal
construction — are known to be "deficient in geometric information" and least repeatable**
(von Cramon-Taubadel 2007 and the Type-I/II/III literature).

Our V11 repeatability data, collected without reference to this classification, reproduces it with
**perfect rank separation**:

| landmark | repeatability | type |
|---|---:|---|
| mandibular_apex_r | 0.9% | I/II anatomical point |
| cephalic_post_mid | 1.4% | I/II |
| scape_apex_l | 1.6% | I/II |
| clypeal_ant_mid | 1.8% | I/II |
| mandibular_apex_l | 1.8% | I/II |
| antennal_insertion_r | 2.4% | I/II |
| antennal_insertion_l *(pinned)* | 2.6% | I/II |
| scape_apex_r | 2.7% | I/II |
| **head_width_r** | **2.8%** | **III constructed extreme** |
| **wl_anterior_r** | **5.0%** | **III** |
| **wl_posterior_r** | **5.1%** | **III** |
| **head_width_l** | **11.1%** | **III** |

**Type I/II median 1.8%, Type III median 5.0% — 2.8×, every Type I/II below every Type III,
Mann-Whitney p = 0.002** (n = 12 landmarks, so underpowered but the separation is complete).

**Consequences.** The annotation is of textbook quality — the errors fall exactly where established
theory says they must. `trait_extract.py` already flagged Weber's length as *"BOTH endpoints are
Type III constructed extremes"* in its own docstring, written before any of this was measured. And
**Weber's length is the normaliser for every trait we report**, so its Type III status propagates
into every ratio. That is a known limitation of the protocol, not a defect of the fit.

---

## 2. VALIDATED IN DIRECTION — coarse-to-fine semantic → skeletal → dense refinement

*Training-Free Non-Rigid Registration of Articulated Animal Bodies via Vision Features and
Anatomical Priors* (IEEE 2025) is close to our problem: it renders multiple views, extracts **DINO
semantic features and anatomical keypoints**, matches coarsely, performs **skeletal pose alignment**,
then does dense geometric refinement. It cites morphological diversity, **symmetry-induced
correspondence ambiguity**, and complex articulation as the motivating difficulties — the same three
this project hit.

**This supports the architecture, not our specific findings.** It is a pipeline proposal; it does
not measure how coarse the coarse stage may be.

---

## 3. ADJACENT, AND WE MUST NOT CLAIM IT AS AGREEMENT

*Hierarchical Neural Semantic Representation for 3D Semantic Correspondence* (SIGGRAPH Asia 2025)
does global initialization → local refinement, and reports that global features localise
semantically similar regions while local-only matching "may introduce mismatches in regions with
similar local geometry but different global semantics".

**But it does not argue that region-level correspondence replaces point-level.** It is hierarchical:
coarse regions are a *route to* voxel-level precision, and it gives **no quantitative requirement**
for how precise the coarse stage must be. Checked directly — that claim is not in the paper.

**So R1's result is not "the SOTA agrees".** R1 found something the correspondence literature does
not test, because it is asking a different question:

> The correspondence literature evaluates correspondence **as an output** — and there, more precision
> is always better. We use correspondence **as a constraint on a parametric model**, and there
> R1 found exact vertex identity is *reachable but costs ~20% additional surface distortion*, with
> the cost saturating once regions reach ≈5% of body length.

**That distinction — precision is free when correspondence is the answer, and costly when it is a
constraint — is ours, and it is the more defensible framing for a contribution.**

DenseMatcher's use of semantic *groups* rather than identical per-vertex identity is the nearest
precedent, and it is a design choice there rather than a measured trade-off.

---

## 4. WHAT THE LITERATURE DOES NOT COVER

- **Nothing found measures how much anatomical supervision an articulated parametric model needs.**
  G8 (~75% joint coverage) and V13 (surface supervision generalises erratically below full coverage)
  appear to be unreplicated in the literature we can find.
- **Nothing found reports the withdrawal pattern we hit** — that a mis-specified landmark ruler can
  masquerade as a fitting pathology for four consecutive experiments (T2, V2, V3 all null before V6
  found the cause). Given how routinely template landmark indices are auto-generated, this is
  probably a common and under-reported failure.
- **The circumferential caution stands.** Our own CSE work found circumferential retrieval on
  tubular structures near chance; any anatomical coordinate field should be designed to degrade
  gracefully on that axis rather than assume both dimensions are equally learnable.

---

## Sources

- [Hierarchical Neural Semantic Representation for 3D Semantic Correspondence (SIGGRAPH Asia 2025)](https://arxiv.org/html/2509.17431)
- [Training-Free Non-Rigid Registration of Articulated Animal Bodies via Vision Features and Anatomical Priors (IEEE 2025)](https://ieeexplore.ieee.org/document/11302489/)
- [DenseMatcher: Learning 3D Semantic Correspondence for Category-Level Manipulation](https://arxiv.org/pdf/2412.05268)
- [The problem of assessing landmark error in geometric morphometrics (von Cramon-Taubadel, AJPA 2007)](https://onlinelibrary.wiley.com/doi/10.1002/ajpa.20616)
- [Landmark Typology in Applied Morphometrics Studies: What's the Point?](https://anatomypubs.onlinelibrary.wiley.com/doi/10.1002/ar.24005)
- [Non-Rigid 3D Shape Correspondences: From Foundations to Open Challenges](https://arxiv.org/pdf/2604.01274)
