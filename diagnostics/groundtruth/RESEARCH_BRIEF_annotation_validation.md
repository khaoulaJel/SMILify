# Validating automated 3D morphometrics for a hyperdiverse clade

**A research brief for the annotation-validation programme.**
Self-contained: written to open a new session with no prior context.

---

## 1. The question

> **Can a parametric 3D shape model produce morphometric measurements from museum scans that are
> accurate enough to support comparative biology — and how would we know?**

The second clause is the harder one, and it is where this programme sits. Automated morphometrics
pipelines are routinely validated against quantities they can compute cheaply — self-consistency,
population coherence, downstream classification accuracy. None of those measure whether the numbers
are *right*. This programme establishes accuracy against human ground truth, and treats the
validation methodology itself as a scientific contribution rather than a chore preceding one.

## 2. Why it matters, and to whom

**The bottleneck is real and current.** Open scanning hardware ([scAnt][scant], Plum & Labonte
2021, PeerJ) has made 3D digitisation of arthropods cheap enough for collections to run at scale;
the ~8 cm specimen limit covers most of Formicidae. Museums can now produce scans far faster than
anyone can measure them. Manual morphometrics does not scale to a clade of ~14,000 described
species, and it carries inter-observer variation that automated methods could in principle remove.

**Three consumers of the output, with different tolerances:**

1. **Comparative morphology and functional ecology.** Needs *proportions* that rank specimens
   correctly across taxa. Tolerates constant bias; cannot tolerate failure to rank.
2. **Synthetic training data.** [replicAnt][replicant] (Plum et al. 2023, *Nature Communications*)
   places rigged 3D models into procedurally generated scenes to train detection, tracking, pose
   and segmentation networks with little or no hand annotation. It consumes 3D models, so model
   *fidelity* propagates directly into every downstream network.
3. **Reference-free size and mass estimation.** [WOLO][wolo] ("Wilson Only Looks Once") regresses
   ant body mass from images without a scale reference, trained on replicAnt-generated synthetic
   data built from scAnt models of *Atta vollenweideri* across 20 weight classes, and integrated
   with OmniTrax for tracking + pose + size in one pipeline.

## 3. The strategic link to WOLO — and why validation is upstream of it

This is the part worth thinking about carefully, because it converts a validation exercise into a
research programme with a destination.

**WOLO is single-species by construction.** Its training corpus is one species across a weight
series, from hand-rigged models. That is exactly the limitation a *parametric, multi-species* shape
model removes: a shape space fitted across many genera can generate plausible novel specimens
spanning the family, rather than one species spanning one size axis. **The obvious next step for
WOLO is a family-general synthetic corpus, and a parametric ant model is the natural generator.**

**But the dependency runs the wrong way if the model is not validated.** If fitted shapes carry
systematic bias, and those fits seed a synthetic corpus, the bias is baked into every network
trained on it — invisibly, because the synthetic ground truth would be internally consistent. A
size-regression network trained on systematically distorted ants would learn a systematically
distorted size–appearance mapping and report excellent validation scores against its own synthetic
truth. **Validating the generator against human ground truth is therefore a precondition for the
WOLO extension, not an afterthought.**

**And the two halves are complementary.** The parametric-fitting route currently cannot recover
absolute size — both corpora lose scale before fitting, so everything is proportion-only, and the
entire size axis (a large fraction of ant morphological variation, and the axis WOLO exists to
measure) is discarded. WOLO learns size from appearance without a reference. **Shape from
registration, size from learned appearance** is a coherent joint architecture, and neither half
delivers comparative morphometrics alone.

## 4. Where the field is, and what is genuinely open

**Parametric articulated shape models** descend from [SMAL][smal] (Zuffi et al. 2017, *3D
Menagerie*) — a linear shape space over ~41 scans of quadrupedal mammals — extended by SMALR/SMBLD,
avian models (*Birds of a Feather*, 2021), and unified transformer approaches spanning Mammalia and
Aves ([AniMer+][animer], 2025). See also the 2025 survey on 3D reconstruction of animal shape and
motion ([arXiv:2508.16062][survey]).

**The open gap: there is no established parametric statistical shape model for insects or
arthropods.** Every model in that lineage targets vertebrates with broadly conserved body plans and
internal skeletons. Arthropods break the assumptions that make the lineage work — an exoskeleton
with discrete sclerites, appendages that are thin relative to body volume, tagmata that vary in
relative proportion far more than mammalian body segments, and taxonomically diagnostic structures
(petiole, propodeal spines, mandibles) with no vertebrate analogue. **Building one for a
hyperdiverse insect clade is a genuine contribution, and reporting honestly where it fails is part
of that contribution.**

**Automated landmarking is an active and increasingly rigorous field.** Deep-learning landmark
placement now reaches sub-millimetre precision comparable to an experienced morphometrician on
large datasets, with recent work explicitly on biases in *registration-based* placement (Springer,
*Evolutionary Biology*, 2026) — i.e. the exact class of method used here. That literature is a
direct comparator, and it means the honest failure modes of a registration approach are of interest
to a readership beyond myrmecology.

**One methodological hazard from the literature applies directly to us.** *On Procrustes
Contamination in Machine Learning Applications of Geometric Morphometrics* ([arXiv:2601.18448][proc])
shows Procrustes alignment introduces dependencies that violate standard statistical assumptions,
inflating classification performance and harming generalisation to independently-aligned data. Any
classification result computed on Procrustes-aligned shape data in this project must be audited
against that, and the audit reported.

## 5. The insights this programme is built on

Stated as principles rather than results, because they generalise beyond this dataset and they are
what the methodological contribution consists of.

**I. Precision is not accuracy, and the field routinely conflates them.** Self-consistency metrics —
population coherence, repeatability across replicates, downstream classification — are cheap and
feel like validation. They cannot distinguish a pipeline that is *noisy* from one that is
*systematically wrong*. Only external ground truth separates them.

**II. In-distribution validation is not validation.** Calibrating a model against synthetic
specimens generated *by that model* measures self-consistency wearing the costume of ground truth.
It certifies as reliable exactly the quantities the model is internally coherent about. This is the
single most consequential methodological trap in the area, and it is easy to fall into because the
synthetic route is the only one that gives unlimited "truth" for free.

**III. Correlation on size-uncorrected measurements is a size test, not a shape test.** Where fit
and truth share an absolute scale, a large specimen has large everything in both, and per-feature
correlation runs toward 1 regardless of whether shape was recovered. Reliability tables built this
way certify measurements that do not rank specimens correctly once size is removed — and size
removal is mandatory when absolute scale is not recoverable.

**IV. Bias and noise license different claims, and the distinction is decisive.** Imprecise but
*unbiased* measurements still support population-level comparative work, because errors average
out. Imprecise *and biased* measurements do not: the aggregate is wrong too. A pipeline's
entitlement to make comparative claims rests on this distinction, which cannot be established
without ground truth.

**V. Human annotation is the only thing that breaks the circularity** — and it must therefore be
treated as an instrument with its own error, not as truth by definition. An annotation without a
repeatability estimate cannot bound a fitter's error, because the two are confounded.

**VI. Convention is not accuracy.** A rig joint is a *rotation pivot*; an anatomical landmark is a
*position*. Annotating one where the schema means the other produces apparent error that is a
definition mismatch. Landmark protocols must be specified before annotation, and the two kinds of
point recorded separately.

## 6. The programme

### Phase A — Complete the ground truth *(prerequisite; nothing downstream is quantitative without it)*

**A1. Blind re-annotation for repeatability.** Re-annotate a subset with the annotator blind to
their first pass. This yields the annotation error floor and is the difference between "the fitter
errs by X" and "the fitter errs by X against an annotation uncertainty of Y". Distal appendage
joints are simultaneously the hardest to place and the largest source of measured error, so without
this every distal figure is an upper bound rather than a measurement.

**A2. Fix the joint convention.** Annotate the head joint per the model's schema (coincident with
the mesosoma joint, at the neck articulation), so head-related joints become comparable rather than
excluded.

**A3. Add explicit landmark points**, recorded separately from rig joints: maximum head width, head
length, mandible tips, and a scale reference where the original scan metadata permits. This is the
only route to testing head-derived measurements, which no joint annotation can reach, and which
carry the pipeline's headline morphological claims.

**A4. Recover absolute scale where possible.** If scan metadata retains voxel spacing or a physical
reference, the entire size axis re-enters — and it is the axis WOLO operates on. Worth an explicit
check of the acquisition records.

### Phase B — Establish what the pipeline actually measures

**B1. Per-block accuracy against human truth**, in proportions, with bias and correlation reported
separately, and with annotation error subtracted using A1.

**B2. Replace the reliability certificate.** Whatever survives becomes the supported feature set;
whatever does not is reported as failing. The comparison between the previous synthetic calibration
and this one is itself a result, and is the concrete instance of Insight II.

**B3. Characterise the bias.** Constant, or shape/size-dependent? A constant multiplicative bias in
log-proportion is correctable post hoc and preserves comparative claims. A bias that varies with
morphology is not correctable and invalidates them. **This single question decides whether the
pipeline's comparative output is salvageable**, and it is answerable with the data Phase A produces.

**B4. Audit for Procrustes contamination** per arXiv:2601.18448, on every classification or
regression result computed from aligned shape data.

### Phase C — The WOLO link

**C1. Fidelity of the generator.** Quantify how bias measured in B propagates into a synthetic
corpus rendered from fitted models — the concrete risk in §3.

**C2. Family-general synthetic corpus.** Sample the shape space across genera rather than one
species and evaluate whether a size-regression network trained on it generalises beyond the species
it was trained on. This is the actual extension of WOLO, and B is what licenses attempting it.

**C3. Joint architecture.** Test shape-from-registration combined with size-from-appearance against
each alone.

### Phase D — Write it

Two contributions, and the second may be the more durable:

1. **The first parametric shape model for a hyperdiverse insect clade**, with an honest account of
   which morphometric quantities it recovers and which it does not.
2. **A validation protocol for automated morphometrics** that distinguishes precision from
   accuracy and bias from noise, with a worked demonstration that the standard shortcut —
   in-distribution synthetic calibration — certifies measurements that fail against human truth.

Negative results are load-bearing here. A pipeline that reports honestly which of its outputs are
trustworthy is more useful to a comparative biologist than one that reports uniformly high
confidence, and the failure modes documented are properties of the method class rather than of one
implementation.

## 7. Standing rules for this programme

Adopted because each was learned by getting it wrong first.

- **Pre-register the bar and the voiding checks before the run.** State what would falsify the
  hypothesis, and what conditions make the result unreadable.
- **Score the proxy and the mechanism separately.** A proxy moving without its mechanism is a
  failure, not a partial success.
- **A voiding check must fail closed on missing data.** A check that silently passes when its input
  is absent is worse than no check.
- **Every anchor must be re-measured on the current pipeline before anything is expressed as a
  distance from it.** Anchors go stale when the pipeline is fixed underneath them.
- **Never validate a model against data that model generated.**
- **Report accuracy, not lift, when comparing models** — a stronger model has a higher chance
  baseline, so lift is not comparable across models. And where classes are imbalanced, the
  majority-class rate is the honest baseline, not a permutation null.
- **Report a reliability figure beside every measurement**, and conspecific agreement beside every
  classification claim.

---

[scant]: https://peerj.com/articles/11155/
[replicant]: https://www.nature.com/articles/s41467-023-42898-9
[wolo]: https://github.com/FabianPlum/WOLO
[smal]: https://smal.is.tue.mpg.de/
[animer]: https://arxiv.org/pdf/2508.00298
[survey]: https://arxiv.org/pdf/2508.16062
[proc]: https://arxiv.org/pdf/2601.18448
