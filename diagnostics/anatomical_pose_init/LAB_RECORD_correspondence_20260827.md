# Lab record — the correspondence line of work

A note to myself, tracking the *reasoning*: what each step was for, what was built, what came back,
what it meant, and what it forced next. Kept as a chain so the logic can be re-walked later without
re-reading every result file.

**The goal throughout:** make SMILify fit ants better by giving the fitter correspondence — which
scan point matches which template vertex — and show the improvement honestly, on real specimens.

---

## 0. Where this line started

Established before this stretch: correspondence is the lever. A dense oracle handed the fitter true
per-vertex correspondence and moved seg_acc 0.832 → 0.874, 12/12 specimens. That proved the *target*
was worth chasing but used ground truth, so it proved nothing about deployability.

Also established: the DensePose-style output (segment class + one scalar position along the segment)
is structurally insufficient. A leg segment is a tube — a 2-D surface — and one number cannot address
a 2-D surface. Confirmed three independent ways.

**So the open question was:** can a *learned* network supply correspondence good enough to matter,
and can it be fed to the fitter without ground truth anywhere in the loop?

---

## 1. Ask the frozen network what it already knows

**Idea.** Before spending GPU on a new head, measure whether the already-trained backbone carries
correspondence-usable structure at all. Cheap, decisive, and it gates everything downstream.

**Implementation.** Read the true per-point features the network already computes, and measure
nearest-neighbour retrieval against a *measured* random floor and achievable ceiling. Test
cross-specimen (query on one ant, match on another), not just within one ant.

**Result.** Within-specimen retrieval looked excellent (0.92–0.97 of ceiling). Cross-specimen was
roughly half that (0.27–0.55) — still far above chance on 4 of 5 leg segments. And the error split
by direction: strong *along* the limb (0.14–0.57), near chance *around* it (0.09–0.20).

**What it means.** The within-specimen number was measuring the wrong thing — features that encode
where a point sits in the current pose ace it while being useless for correspondence. The
cross-specimen number is the real one. The backbone carries genuine but partial signal, and its
weakness is specifically **circumferential**.

**Next.** That circumferential asymmetry is exactly what SurfEmb says a *deterministic* embedding
cannot fix and a *contrastive* one can. So the head design was decided by this measurement, not by
preference.

---

## 2. What the old conversion step was costing

**Idea.** Two fits were already queued that turned network output into pose via a centroid/IK
conversion. Rather than read them as "does the network help", use them to price the conversion.

**Implementation.** Compare the network-fed arm against an arm fed *true* labels through the *same*
conversion, and both against the dense oracle.

**Result.** True labels through the conversion reached seg_acc 0.8522. The dense oracle reached
0.8745. The network-fed arm, 0.8488.

**What it means.** The conversion discards ~41% of the achievable gain, and flattens the gap between
a perfect signal and a predicted one to 0.003. **The conversion, not the network, was the
bottleneck.** This also explained an old puzzle: the trochanter was the worst segment under
conversion and the *best* under direct retrieval.

**Next.** Whatever head gets built, feed its output to the fitter as **direct vertex identity** —
the way the dense oracle did — and never through the conversion again.

---

## 3. The CSE head

**Idea.** Replace only the decode head and its loss. Emit a continuous per-point embedding, hold one
learned embedding per template vertex, and correspond by nearest neighbour — no hand-designed
coordinate system anywhere.

**Implementation.** New module, leaving the live one untouched. Both sides L2-normalised so the
training loss optimises exactly the retrieval performed at inference. Supervised by **true vertex
index** — the quantity the dense oracle proved matters — rather than the scalar shown insufficient.
Loss is contrastive (InfoNCE) against vertex identity, chosen because §1 showed the residual is
circumferential. Backbone warm-started from the existing checkpoint.

**Result.** Cross-specimen retrieval improved on 4 of 5 segments, all p < 6e-3. The circumferential
component — the reason for the design choice — improved 2–4×, from 0.09–0.20 to 0.28–0.43.

**What it means.** The literature-derived prediction was made before the head was written and then
confirmed by the specific thing it predicted. That is stronger evidence than a general improvement.

**But:** direct template-vertex retrieval, which is what the fitter actually consumes, was only
usable on two segments (trochanter, femur). Tibia sat at chance, tarsus below it.

---

## 4. Feeding it to the fitter

**Idea.** Aggregate the network's predictions into "where does the network think vertex *v* is on this
scan", and add a masked term pulling the fitted mesh's vertex *v* onto that position. Additive,
flag-gated, off by default.

**Result.** seg_acc 0.8657 against a 0.8201 zero-init baseline — **84% of the dense oracle's gain,
from predictions alone**. And it beat the oracle fed true labels through the conversion (0.8522).

**What it means.** Worse information through the right mechanism beats perfect information through a
biased one. Independent confirmation of §2 on the fitter's own metric.

---

## 5. Making it deployable: which points to trust

**Idea.** Restricting to two segments only worked because held-out ground truth said which segments
were reliable. Real scans have no ground truth, so that restriction could never ship. Need a
confidence signal computable at inference.

**Implementation.** Test four candidates against held-out truth: embedding similarity, forward-backward
cycle consistency, mutual nearest neighbour, and multi-sample stability.

**Result.** Keeping the top half of points by **cycle consistency** cut mean retrieval error
0.726 → 0.449. Keeping the top half by **similarity** *raised* it to 1.044. MNN and stability did
nothing.

**What it means.** Similarity is entangled with ambiguity — a point in a confusable region scores
*high* precisely because it is confusable. Cycle consistency is a structural self-consistency test
and is not. This is the same pathology LightGlue describes for matchability vs similarity.

Cycle consistency also selects good points *within* every segment rather than excluding bad segments
— so the whitelist could be dropped entirely.

**Next.** Regenerate correspondences with cycle filtering across all segments. Coverage rose from
~10% to ~28% of template vertices, and the pipeline became ground-truth-free end to end.

---

## 6. Establishing the effect properly

**Idea.** At 12 specimens the configuration variants were indistinguishable from each other. Choosing
one on the strength of being numerically highest is not a decision. Synthetic fits are cheap; widen
the sample before committing scarce real-scan budget.

**Implementation.** 48 fresh specimens on an independent seed, four arms.

**Result.** +0.031 seg_acc, 37/48, significant on all three tests. The configurations differ by
~0.001 — genuinely equivalent, not merely unresolved.

**What it means.** The effect is real and robust, and *smaller* than the 12-specimen estimate of
+0.050. The corrected figure is the one to carry. Since configuration doesn't affect performance,
choose on deployability: all-segments, because it needs no whitelist.

---

## 7. What the weak segments are actually contributing

**Idea.** Adding tibia and tarsus helped, even though their per-point retrieval is worse than random.
That shouldn't happen if they contribute correspondence. Three-way test needed: real correspondence
vs shuffled-within-segment vs no constraint.

**Result.** Shuffled (0.8711) and real (0.8701) are indistinguishable — sign p = 1.0. Removing them
entirely costs ~0.02.

**What it means.** **The head is doing two different jobs.** On trochanter/femur it supplies genuine
correspondence. On tibia/tarsus it supplies only a coarse regional pull — which vertex is irrelevant,
only that *some* pull exists toward the right area.

**Consequence.** Improving tibia/tarsus *correspondence* would buy nothing, because their
correspondence is not what the fitter is using. That closed the planned class-balanced retrain before
it was built.

---

## 8. First contact with real scans

**Idea.** Everything so far was synthetic, shared topology. Before building anything on real data,
check the network doesn't simply collapse on it.

**Result.** Once real meshes are normalised into the model's frame, the retrieval distribution stays
healthy — coverage, entropy and per-segment shares all track synthetic closely. **No collapse.** But
cycle consistency, the signal the whole filter depends on, degrades 2.6× on average, and unevenly
across specimens.

**What it means.** The mechanism transfers; its confidence signal degrades. Since the degradation is
per-specimen, cycle consistency doubles as a ground-truth-free triage signal.

**Investigated the cause across all 50.** Not scan damage, not point density, not species (within-genus
spread exceeds between-genus). Partly aspect ratio (ρ = −0.502) — but only ~25% of variance, and in
the direction *opposite* to a simple distribution-distance story.

---

## 9. Does it actually help real ants?

**Idea.** Run the fit on real specimens with and without the term. No ground truth exists, so measure
geometric fit quality only — "did the fit get closer to the scan", never "is correspondence correct".

**Result, cleanest 14 specimens.** fscore@0.01 0.713 → 0.787, **14/14**. Normal consistency
0.830 → 0.879, 14/14. Hausdorff-95 improved 14/14.

**Then the important one.** Rather than assume the degraded specimens were unusable, fit them too.
The worst 14 (degradation 5.2–12×) still gained +0.054 — **74% of the clean-set gain**. Then the
remaining 22, completing all 50.

**Result across the whole corpus.** 44/50 improved, mean +0.067, sign p = 3.2e-08. Gain is **flat**
from degradation ratio 0 to 8, and 47 of 50 specimens are below 8.

**What it means.** Degradation predicts how *hard* a scan is to fit, not whether correspondence
*helps* it. Those are separate axes. The usable scope is **~94% of the corpus**, not the screened
subset — a much broader claim, and the honest one.

---

## 10. Why circumferential retrieval is still weak

**Idea.** The obvious explanation was geometric: a leg is a cylinder, a cylinder's cross-section has
no unique axis, so the disambiguating signal is architecturally unrepresentable — which would call
for an equivariant architecture. Test that premise before funding it.

**Implementation.** Measure the eigenvalue gap spanning the circumferential plane on real leg
neighbourhoods, with a non-cylindrical region as control, and the metric checked against toy shapes
whose answer is known by construction.

**Result.** Real trochanter/femur neighbourhoods score 0.38–0.44 against a perfect cylinder's 0.031
floor — 12–14× above it.

**What it means.** Ant leg segments are tapered and flattened, **not** cylinders. The disambiguating
signal **is present in the raw geometry**. So the deficit is not "unrepresentable" — it is "present
and unlearned".

**This changes the fix.** Not an architecture change; a training change. Which is much cheaper.

---

## 11. Multi-hypothesis: real headroom, no mechanism

> **FLAG (2026-08-28): the numbers in this section were measured at n=10 specimens**, on the **C3**
> checkpoint (`multihypothesis_feasibility_20260827.py --n_specimens 10`, default `--c3_ckpt`).
> Treat them with the same caution as every other small-sample figure in this investigation — the
> n=12 `leg_acc` artefact and the n=12 `seg_acc` figure were both wrong in the same regime. They
> have been cited across later work as if solid; they are not.
>
> They are also **checkpoint-specific**. Re-running the same script on the **C11** head gives
> `fe` cycle 15.8% (not 27.0%) and `tr` best-selector `coherence` 11.2–11.8% (not cycle 15.7%) —
> so hard-negative mining, which already failed its own circumferential bar, additionally
> **weakened cycle-consistency as a selector**. A `knn` control (12 vs 8) moves `tr` by 0.6pp and
> `fe` not at all, so the checkpoint explains essentially all of the difference.
>
> Superseded by `RESULTS_F2_trained_selector_20260828.md`, which closes this line.

**Idea.** If the ambiguity is genuinely multi-modal, forcing one confident answer is the worst
possible handling. Emit several candidates and select among them — the pattern that already worked
one level up, at pose initialisation.

**Implementation.** Measure headroom *before* designing a selector: how much better would an oracle
picking among the top 5 do?

**Result.** Headroom is real — an oracle would cut error ~30% on both good segments. But the ranking
check: cycle consistency picks the truly-best candidate 24% of the time against 20% chance, and isn't
significant on femur.

**What it means.** Cycle consistency works as a coarse binary filter and fails as a fine-grained
ranker — it goes flat on near-ties, the same way similarity did. So the headroom exists but **no
realisable signal has been shown to reach it**. This is an open problem, not scheduled work, and it
is parked deliberately rather than left to drift into a plan.

---

## 12. Current step, in flight

**Idea.** Following directly from §10: the circumferential signal is present and unlearned. Uniformly
drawn training negatives are almost all on *other* segments and get solved long before circumferential
detail matters — so the network is never forced to learn the distinction.

**Implementation.** Replace half the negative budget with vertices in the **same segment at a similar
position along it** — differing mainly circumferentially. Everything else identical to the previous
training run, so the comparison is clean.

**Bar fixed in advance:** circumferential retrieval must improve ≥ 0.05 with sign p < 0.05. Gains
along the limb don't count; that was never the problem. A drop elsewhere is a cost and gets reported.

**Result.** It **failed its own bar.** Circumferential gap-closed went 0.317 → 0.296 on trochanter
and 0.280 → 0.306 on femur — one slightly worse, one improving less than half the required amount.

**But** direct template-vertex retrieval improved across the board: femur 0.258 → 0.211, tibia
0.340 → 0.246 (crossing from at-chance to beating chance), tarsus 0.881 → 0.588, held-out top-1
0.0616 → 0.0709.

**What it means.** The intervention worked on a different axis than the one it targeted. Forcing
same-segment negatives made the head better at picking the right vertex overall, without improving
the circumferential component specifically. So the reasoning in §10 — "the signal is present, so
harder negatives will surface it" — is **not supported**. Something else is preventing that component
from being learned. The geometric finding that the signal *is* present still stands; what is
falsified is that this is the way to reach it.

The side-gains are real but were not reclassified as success: the bar was fixed on circumferential in
advance precisely so that could not happen afterwards.

---

## 13. Did the retrain change what `ti` is doing?

**Idea.** C11 moved `ti` direct retrieval from at-chance (0.340 vs 0.339 random) to clearly beating
chance (0.246). Step 7 had shown `ti` supplies only a coarse regional pull — shuffling its retrieved
vertices within segment cost nothing. So: has its *role* changed from "any plausible point in the
region" to "the actually correct vertex"? That is a categorical question, not a metric one, and the
shuffle control from step 7 is exactly the instrument for it.

**Implementation.** Three arms, all fed from C11, differing **only** in how `ti` is treated: real
identity / shuffled within segment / absent. Readings fixed in advance, including what would count as
"the mechanism changed" versus "retrieval improved but the fitter doesn't care".

**Result.**

| contrast | Δ | sign p | Wilcoxon |
|---|---|---|---|
| shuffled vs real `ti` | −0.0034 | 0.774 | 0.519 |
| real vs no-`ti` | +0.0122 | 0.039 | 0.002 |

For comparison, the same contrast under the previous head was +0.0010, p = 1.0.

**What it means.** Unchanged. Destroying `ti`'s identity still costs nothing; removing the segment
still costs something. `ti` is supplying coverage, not correspondence — a substantially better
retrieval signal did not change what the fitter uses it for.

**The wider point.** This is the third time in this line of work that a clear retrieval improvement
failed to reach the fitter. Retrieval quality and fitter sensitivity are only loosely coupled, and
retrieval gains should not be assumed to propagate — they have to be tested at the fitter each time.

---

## 14. Checking a null instead of repeating it

**Idea.** Every summary in this record had been carrying the same sentence: *leg accuracy has never
moved under any arm, including the oracle*. It had been repeated often enough to become a premise —
the starting point for a proposed fix (rigid per-leg initialization before the fitter runs) built
entirely on the assumption that it was true. Before building on a null, read what the metric actually
computes and re-test it at the sample size now available.

**Implementation.** Read `leg_confusion` in the audit's own code first. It never touches the
correspondence: for each *scan point* it does an unrestricted nearest-vertex search over the whole
fitted mesh and asks which leg that vertex belongs to. It is a purely geometric question — is fitted
leg L physically closest to the scan points of leg L. Then re-ran the paired per-specimen comparison
on the 48-specimen corpus rather than the original 12.

**Result.**

| contrast | n | Δ leg_acc | pos | sign p | Wilcoxon |
|---|---|---|---|---|---|
| dense oracle vs zero | 12 | +0.0650 | 8/12 | 0.388 | 0.176 |
| **CSE vs zero** | **48** | **+0.0578** | **34/48** | **0.0055** | **5.4e-05** |
| ceiling vs zero | 12 | +0.1193 | 12/12 | 0.0005 | 0.0005 |

**What it means.** The null was false. Correspondence moves leg accuracy decisively, by *more* than
it moves segment accuracy (+0.058 vs +0.031), and it closes roughly half the distance to the measured
ceiling. The n=12 arms had the effect at nearly full size and simply could not resolve it.

**The wider point.** This is the same failure mode as the earlier over-optimistic n=12 segment number,
in the opposite direction: small samples produce false nulls as readily as false positives, and a
result that gets restated across documents stops being re-examined. A claim repeated is not a claim
tested.

---

## 15. Where the leg-level residual actually lives

**Idea.** With leg accuracy established as movable, the question becomes what the remaining half of the
gap is made of. Three candidate shapes, each implying a different fix: a few catastrophically wrong
specimens (an initialization/basin problem), errors spread across all legs (a precision problem), or
errors concentrated in particular segments (a correspondence-quality problem).

**Implementation.** Three decompositions of the same fits, each against a *measured* reference rather
than an assumed one. Per-specimen distribution. The leg-to-leg confusion structure. Leg-level error
stratified by the point's true segment, with a ceiling built by substituting the ground-truth mesh for
the fit on the *same* 48 specimens — a paired ceiling, so what is irreducible is measured, not guessed.

**Result.** Not catastrophic specimens: correspondence rescues precisely those (0.560→0.936,
0.694→0.972; specimens below 0.8 go 10→1, below 0.7 go 3→0), and the confusion matrix puts ~0.000
mass on non-adjacent legs — errors are strictly between spatial neighbours. Stratified by segment,
against the paired ceiling:

| segment | zero | CSE | ceiling | share of *addressable* residual |
|---|---|---|---|---|
| co | 0.4177 | 0.4043 | 0.0949 | **57.8%** |
| tr | 0.1519 | 0.0846 | 0.0122 | **27.7%** |
| fe | 0.1515 | 0.0540 | 0.0048 | 9.6% |
| ti | 0.1564 | 0.0446 | 0.0010 | 2.4% |
| ta | 0.1831 | 0.0477 | 0.0016 | 2.2% |
| pt | 0.1814 | 0.0549 | 0.0000 | 0.2% |

**What it means.** The residual is proximal. Coxa and trochanter are 85.5% of it; the distal segments
this line of work has spent most of its effort on are 4.8%, and correspondence already improves them
*most* (−0.11 to −0.14). The coxa is not irreducible overlap either — the ceiling reaches 0.095 there,
so three quarters of its error is addressable in principle.

**The uncomfortable part.** `ta` had accumulated three independent lines of evidence marking it as the
system's limiting segment. It is 2.2% of this residual. The evidence about `ta` was correct; the
inference that it therefore mattered most was never checked against where the error volume actually
sat. Effort had been allocated by *how interesting a deficit looked*, not by how much of the metric it
owned.

---

## 16. Is the coxa reachable at all?

**Idea.** Why would the coxa resist correspondence when everything else responds? Measuring what
`leg_acc` actually rewards gives a candidate: it is decided by nearest-vertex, so what a placement
error *costs* depends on the distance to the nearest **other leg**, not on body size. Adjacent coxae
are packed on the thorax; tarsi are far apart. Measured on the corpus, error relative to that
tolerance — call it risk — is **0.995 at the coxa and ~0.21 at every other segment**. The coxa is in
fact the *best-placed* segment in absolute terms (0.0201 of body diagonal vs the pretarsus's 0.0397);
it is the tolerance that is different, not the fit. And an unweighted L2 gives each vertex gradient
proportional to its residual — the *least* gradient exactly where tolerance is tightest. So: reweight
the dense term by inverse measured tolerance, optimising risk instead of absolute error. This is the
per-keypoint sigma construction from COCO's OKS, applied per vertex, with the tolerance measured.

**Implementation.** Three arms sharing one correspondence file and one recipe, differing only in the
per-vertex weight: uniform, inverse tolerance (9.5× coxa:pretarsus), inverse tolerance squared (91×).
Bar fixed in advance, including guard-rails so a coxal gain bought by distal loss could not be reported
as a win. The uniform arm doubles as a check that the weighting plumbing left the unweighted path
intact — it did (+0.0007, n.s., against the previously-fit arm).

**Result.** FAIL, on every reading. Coxal leg error moved +0.0001 and −0.0015 under the two weightings
(sign p = 0.47). Both guard-rails tripped: distal regressed +0.019 / +0.043 and overall leg accuracy
*fell* (−0.0057 / −0.0142, Wilcoxon 0.026 / 0.0035). Measured on placement error, which is what the
term actually acts on, the coxa across the full 91× sweep went **0.0201 → 0.0201 → 0.0200**, while the
femur went 0.0294 → 0.0341 and the tarsus 0.0377 → 0.0499.

**What it means.** The weighting demonstrably had force — everything else moved, monotonically in the
weight. The coxa did not respond at any weight. So coxal placement is not gradient-starved; it is
unreachable by dense correspondence entirely. Decomposing where the offset comes from explains why:
the per-leg component is **0.5%** — the six coxae are not independently misplaced — while **70.7%** of
a specimen's coxal offset is *common to all six of its coxae*, a roughly 3-DOF rigid translation of the
whole group. The body itself is shifted along the long axis by +0.0066 against the coxae's +0.0108. And
the decisive number: **body per-vertex error 0.0203, coxa 0.0201**.

The coxa is fitted exactly as well as the body is. It is not badly fit at all. The fitter's general
precision floor — about 0.020 of body diagonal — simply *equals* the inter-coxa spacing of 0.0202.
That is the entire explanation for risk 1.0 at the coxa and 0.21 everywhere else.

**Next.** 57.8% of the addressable leg residual sits behind global fit precision, which no
correspondence-side work can move: not a better network, not a better confidence signal, not a better
weighting. Raising it is a different programme — the fitter's own convergence and precision — and it
should be argued for on its own terms rather than approached obliquely through correspondence again.

---

## Where this has arrived

**Holds.** Correspondence is a real, deployable lever. It improves reconstruction on 44 of 50 real
museum specimens from predictions alone, with no ground truth at any stage of the deployed pipeline.
On synthetic data it recovers most of what a ground-truth oracle achieves. It moves **both** the
within-part metric (+0.031) and the between-part metric (+0.058, sign p = 0.0055), and it rescues
precisely the specimens that fail worst without it. The consumption mechanism (direct vertex identity)
and the confidence signal (cycle consistency) are both settled and both ground-truth-free.

**Corrected.** Leg accuracy *does* move — the long-repeated null was an artifact of n=12. And the
remaining leg-level residual is proximal, not distal: coxa and trochanter own 85.5% of it against a
measured paired ceiling, while the distal segments that absorbed most of the investigation own 4.8%.

**Bounded.** The coxal share — 57.8% of the addressable residual — is bounded by the fitter's global
precision floor, not by correspondence. It is inert to the dense term across a 91× weight sweep, and
the coxa is fitted as accurately as the body itself; it only *looks* like a failure because the
inter-coxa spacing happens to equal the precision floor. Nothing on the correspondence side will move
it.

**Open.** Circumferential retrieval is still near chance. The tarsus is below chance and its
correspondence is unused anyway — though it is now known to own 2.2% of the leg residual, so the case
for spending on it is much weaker than it looked. Multi-hypothesis headroom sits unclaimed for want of
a selector.

**The shape of the argument, if it has to be defended:** every claim that survived did so by beating a
control designed to kill it — a measured floor and ceiling rather than an assumed one, cross-specimen
rather than within-specimen, an independent 48-specimen sample rather than the original 12, a shuffle
control rather than an ablation alone, a paired ceiling on the same specimens rather than a borrowed
one, and the whole real corpus rather than the subset that screened well.

**And the recurring lesson, now four times over:** read what a metric computes before reasoning from
it, and re-test a null before building on it. Three metric-definition errors and one false null were
all caught the same way — by checking the definition or the power against something with a known
answer — and each time the correction changed the conclusion, not merely the wording.
