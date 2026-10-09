# Session synthesis — CSE correspondence, 2026-08-26 → 08-27

> **READ THIS FIRST — corrections that supersede figures elsewhere in this document.**
>
> 0. **`leg_acc` DOES move. Every "leg_acc has not been moved by anything" statement below
>    (§104, §193) is WRONG and must not be quoted.** That claim came from n=12, the same power
>    regime that inflated the segment number in correction 1 — small samples produce false nulls
>    as readily as false positives. On the independent 48-specimen corpus:
>    **+0.0578, 34/48, sign p=0.0055, Wilcoxon p=5.4e-05** — a LARGER effect than the +0.031
>    segment-level gain, closing roughly half the distance to the measured ceiling. Correspondence
>    also rescues precisely the worst specimens (0.560→0.936, 0.694→0.972; specimens below 0.8 go
>    10→1). Any framing of correspondence as "within-part only" is wrong.
> 5. **The remaining leg-level residual is PROXIMAL, and its largest share is NOT addressable by
>    correspondence.** Against a measured paired ceiling on the same 48 specimens, coxa (57.8%) +
>    trochanter (27.7%) = 85.5% of the addressable residual; `ti`+`ta`+`pt` = 4.8%. The coxa is
>    inert to the dense term across a 91× weight sweep (placement error 0.0201 → 0.0200) while
>    every other segment responds monotonically, and it is fitted as accurately as the body itself
>    (0.0201 vs 0.0203). The fitter's global precision floor (~0.020 of body diagonal) simply
>    EQUALS the inter-coxa spacing (0.0202). That share is bounded by fit precision, not by
>    correspondence, the network, or the confidence signal. See
>    LAB_RECORD_correspondence_20260827.md §14–16.
>
> 1. **The headline effect is +0.031, NOT +0.050.** The +0.0500 in §6 came from n=12 and was
>    inflated by small-n luck. On an INDEPENDENT 48-specimen corpus the effect is **+0.0310**
>    (37/48; sign 2.2e-04, Wilcoxon 3.0e-05, paired-t 2.4e-04; **outlier-excluded +0.0137**).
>    Real and robust, but ~60% of the earlier figure. Quote +0.031.
> 2. **The configuration choice is DEPLOYABILITY, not performance.** At n=48 all three
>    configurations are equivalent within noise (all-segments vs tr/fe/co −0.0008, sign 0.471,
>    Wilcoxon 0.401, t 0.822; vs tr/fe −0.0014). These are negligible effect sizes, not
>    underpowered nulls. `CSE_cycle_all` is chosen because its cycle-consistency filter needs no
>    ground-truth whitelist — not because it scores higher.
> 3. **SCOPE ON REAL SCANS IS ~94% OF bench50, NOT 28%.** All 50 specimens fitted:
>    **44/50 improved (88%), mean fscore@0.01 gain +0.0672, sign p=3.2e-08, Wilcoxon
>    p=3.4e-09.** Gain is FLAT from degradation ratio 0 to 8 (+0.056 to +0.079, Spearman
>    vs ratio rho=-0.093 p=0.521 — no decline), and 47/50 specimens sit below ratio 8.
>    Only 3 specimens above ratio 8 show no gain. Cycle-consistency ratio predicts scan
>    DIFFICULTY, not whether correspondence helps. Any earlier "works on the clean 28%"
>    framing is wrong by a factor of three and must not be repeated.
> 4. **Two planned workstreams are DEAD. Do not revive them without rereading §9.**
>    Class-balanced retraining to improve `ti`/`ta` correspondence, and segment-quota weighting by
>    retrieval quality. Reason: **correspondence identity on `ti`/`ta` is irrelevant.** Shuffling
>    their retrieved vertices within-segment costs nothing (+0.0010, sign p=1.0, Wilcoxon 0.91),
>    while removing the segments costs ~0.02. Improving a signal the fitter does not use buys
>    nothing.

Supersedes `PROGRESS_SYNTHESIS_20260826.md` from "the path forward: CSE-style (Variant 2)" onward.
Everything that document marked settled remains settled and is not re-litigated here.

**Bottom line:** the correspondence goal is met on synthetic data. A learned CSE embedding head,
fed to the fitter as direct vertex identity, recovers ~84% of the dense-ground-truth oracle's
seg_acc gain using predictions only — and beats an oracle fed TRUE labels through the old
centroid/IK conversion. The conversion, not correspondence quality, was the bottleneck all along.

Commits: `6d0f6f75` (feasibility probe + D1 audit rows) · `a5925ccc` (CSE head + fitter term) ·
`955286c5` (C4 result).

---

## 1. Where this started

`PROGRESS_SYNTHESIS_20260826.md` closed the DensePose-style Variant 1 for a structural reason (a 1-D
scalar cannot address a 2-D tube surface) and named CSE as the alternative, with one weak
encouraging sign: r=0.547 distance correlation on 44 points, one band, one segment (`fe`), one
specimen, using a nearest-FPS-centroid proxy for the feature. It pre-registered three things that
had to happen before spending GPU quota: cover `tr`, measure retrieval accuracy rather than a
correlation, and replicate across specimens.

Two D1 fits were also queued (`Network_correspondence`, `GTLabel_IK_oracle`), both using the
centroid/IK conversion already known to carry a 2.03× bone-length bias.

## 2. Feasibility probe — zero GPU, and it decided the design

`cse_feasibility_retrieval_20260826.py`, 12 held-out specimens, frozen B2 checkpoint, CPU only.

Fixed the approximation for free: `smil_correspondence_net.py:137` decodes both heads from
`l0_points = fp1(...)`, shape (B,128,N) — a TRUE per-point feature. The r=0.547 probe had used a
centroid proxy. Reading `l0_points` directly removed that caveat with no retraining.

**Within-specimen retrieval scored 0.92–0.97 of the measured ceiling — and was misleading.** Those
numbers are achievable by features encoding absolute position in the current pose, which is useless
for correspondence. Adding a cross-specimen test (query on A → feature-NN on B, error between true
template vertices in canonical rest space) roughly halved them. The flattering number would have
been reported had the check not been run.

Cross-specimen gap-closed (fraction of the MEASURED random→ceiling gap):

| seg | gap closed | sign p | verdict |
|---|---|---|---|
| tr | 0.551 | 3.1e-20 | PASS |
| ti | 0.545 | 7.3e-04 | PASS |
| fe | 0.348 | 7.0e-13 | PASS |
| co | 0.265 | 5.8e-12 | PASS |
| ta | 0.138 | 0.27 | FAIL |

Two findings that steered everything after:

- **`tr` was the BEST segment, having been the worst offender everywhere else** (2.03×→3.85×
  bone-length bias under centroid and furthest-point conversion). Strong early evidence the failures
  were conversion-driven, not feature-driven.
- **Axial gap-closed 0.14–0.57 vs circumferential 0.09–0.20** — near chance, negative on `ta`, on
  all five segments. The backbone localises along the limb but essentially not around it.

That asymmetry is exactly the near-rotational-symmetry ambiguity SurfEmb (arXiv:2111.13489)
identifies as the failure mode of a *deterministic* embedding. The prediction was made from the
literature BEFORE the probe was written, then confirmed — which is why the loss below is contrastive
rather than a regression.

Also verified live: `chain_pos` is degenerate in current code (`l1_r_co` min == max == 0.0). It
silently zeroed every segment length on the first run; the axial/circumferential split was rebuilt
from posed-geometry PCA instead of that label.

## 3. The two D1 arms — historical, and one real finding

Both use the superseded centroid conversion, so neither says anything about network quality.

| run | seg_acc | Δ | sign | wilcoxon | t |
|---|---|---|---|---|---|
| Network_correspondence | 0.8488 | +0.0287 | 0.146 | 0.052 | 0.122 |
| GTLabel_IK_oracle | 0.8522 | +0.0321 | 0.0064 | 0.0024 | 0.091 |
| *Dense_GT_oracle* | 0.8745 | +0.0544 | 0.0005 | 0.0005 | 0.013 |

leg_acc failed every test for both, and went slightly NEGATIVE under outlier exclusion
(−0.0006/−0.0007 vs +0.065/+0.068 nominal).

**The one thing they established:** `GTLabel_IK_oracle` is fed TRUE labels and still reaches only
0.8522 versus `Dense_GT_oracle`'s 0.8745 on identical specimens. **The conversion discards ~41% of
the achievable gain** and flattens the network-vs-oracle distinction to +0.003. Independent
confirmation, on the fitter's own metric, of what the `tr` inversion showed geometrically.

## 4. C3 — the CSE head

`fitter_3d/pointcloud2smil/smil_cse_net.py` (new module; `smil_correspondence_net.py` left
byte-identical because it feeds the live D1 chain), trained by
`train_cse_head_20260826.py`, 120 epochs, warm-started from B2 (196 tensors, hard-fails rather than
silently training from scratch).

Design, each choice traceable to evidence:
- **Per-point embedding + learned per-vertex key table**, both L2-normalised, so the InfoNCE loss
  optimises exactly the nearest-neighbour retrieval performed at inference — no metric mismatch.
- **Supervised by TRUE VERTEX INDEX**, the quantity `Dense_GT_oracle` used for the only validated
  correspondence win — not the `chain_pos` scalar shown to be structurally insufficient.
- **Contrastive, not regression**, because the measured residual was specifically circumferential.

C3 vs B2 on the identical cross-specimen metric:

| seg | B2 → **C3** gap | sign / wilcox / t | **circumferential** |
|---|---|---|---|
| tr | 0.544 → **0.740** | 1.6e-07 / 3.5e-10 / 2.0e-10 | 0.090 → **0.317** |
| ti | 0.553 → **0.720** | 5.3e-04 / 2.2e-03 / 3.1e-03 | 0.107 → **0.380** |
| fe | 0.303 → **0.557** | 3.8e-08 / 8.9e-08 / 7.0e-08 | 0.144 → **0.280** |
| co | 0.278 → **0.436** | 1.6e-07 / 3.3e-08 / 2.3e-09 | 0.200 → **0.433** |
| ta | 0.339 → 0.288 | n.s. (0.48) | — |

The circumferential column is the point: a 2–4× gain on precisely the dimension the literature said
a contrastive loss would rescue. `ta`'s non-result was pre-called by the feasibility check.

**But direct template-vertex retrieval — what the fitter consumes — is only usable on two segments:**
`tr` 0.146 vs 0.377 random, `fe` 0.258 vs 0.345; `co` weak (0.324 vs 0.490), `ti` AT CHANCE (0.340
vs 0.339), `ta` WORSE than chance (0.881 vs 0.363). The point-to-point embedding improved broadly;
the vertex table is only well-trained where there are enough vertices and samples.

## 5. NEGATIVE RESULT — the confidence signal is anti-correlated with correctness

Pre-registered check: filter points by cosine similarity between query and retrieved key, then
verify the unreliable segments survive at a LOWER rate. **It failed, in the opposite direction:**

| seg | survival | held-out reliability |
|---|---|---|
| ta | 95.7% | worse than chance |
| ti | 94.5% | at chance |
| tr | 55.6% | best |
| co | 18.2% | weak |

Filtering by confidence makes the correspondence set worse. It was disabled rather than replaced
with another unvalidated heuristic. Correspondences were instead restricted to `tr`/`fe` on
held-out evidence fixed before any fit ran — an evidence rule, not post-hoc selection.

**This is open work and it gates deployment**: on real scans you cannot whitelist segments from
held-out ground truth in advance.

## 6. C4 — the goal-level result

New additive, flag-gated fitter term `--cse_correspondence_from` / `--w_cse_correspondence`, feeding
predicted correspondence as DIRECT vertex identity and bypassing the conversion entirely.
**Verified byte-identical with the flag off** across all four hierarchical stages, every array,
maxdiff exactly 0.0 — measured against a stashed baseline, not asserted.

seg_acc, n=12 paired, baseline `SYN_clean_zero_wsl` = 0.8201:

| run | mean | Δ | pos | sign | wilcox | t | outlier-excl |
|---|---|---|---|---|---|---|---|
| **CSE_correspondence** | **0.8657** | **+0.0456** | 9/12 | 0.146 | **0.0122** | **0.0337** | +0.0208 |
| Network_correspondence | 0.8488 | +0.0287 | 9/12 | 0.146 | 0.0522 | 0.122 | +0.0128 |
| GTLabel_IK_oracle | 0.8522 | +0.0321 | 11/12 | 0.0064 | 0.0024 | 0.0908 | +0.0090 |
| *Dense_GT_oracle* | 0.8745 | +0.0544 | 12/12 | 0.0005 | 0.0005 | 0.0128 | +0.0304 |
| *GT_as_fitted_ceiling* | 0.9769 | +0.1568 | 12/12 | 0.0005 | 0.0005 | 6.4e-05 | +0.1270 |

**Recovers 84% of the oracle's gain (+0.0456 of +0.0544) from predictions, using only `tr`/`fe` at
~10% vertex coverage.**

**And it beats the GT-fed oracle** (0.8657 vs 0.8522): worse information through the right mechanism
beats perfect information through the biased one. Cleanest possible confirmation that the conversion
was the bottleneck — predicted twice before being observed.

## 7. What is NOT established

- **Sign test is not significant** (9/12, p=0.146) while Wilcoxon and paired-t are. Two of three
  tests, versus the oracle's three of three. The gain rides on magnitude in a subset of specimens,
  not a uniform win.
- **Outlier-excluded Δ is +0.0208**, under half the headline.
- **leg_acc has not been moved by anything.** +0.0657 nominal but −0.0004 outlier-excluded, sign
  n.s. — and EVERY arm shows this null, `Dense_GT_oracle` included. No correspondence work in this
  investigation has touched that axis.
- **No usable confidence signal** (§5).
- `ta`/`ti` unsupported; `pt` never testable (8 template vertices).
- **Synthetic corpus, shared topology, one corpus.** Nothing here transfers to bench50 real scans,
  where the recorded gotcha stands: a point distribution unlike training breaks this network
  outright.

## 8. Next, in priority order

1. **A valid confidence signal.** Blocks any deployment that cannot whitelist segments in advance —
   i.e. blocks bench50. Candidates: embedding-space local density, agreement across repeated
   samples, cycle-consistency (retrieve A→B→A).
2. **Extend past `tr`/`fe`.** `ti`/`ta` failed because their vertex-table entries are undertrained
   (few vertices, few samples), not because the backbone lacks signal — cross-specimen point-to-point
   was fine on `ti` (0.720). Class-balanced sampling or a coordinate-conditioned key model.
3. **Bench50 sim-to-real**, with the distribution-mismatch failure mode explicitly handled.
4. C1 sampler ablation — still pending, unchanged.

## Standing discipline (unchanged, reaffirmed by this session)

Probe before editing the live D1 chain; additive, flag-gated, byte-identical when off, verified not
claimed. Never assume a metric's ceiling — measure it. Sign test + Wilcoxon + paired-t always, with
the outlier-excluded delta beside the full-sample one. Pre-register criteria. Report negative results
with equal prominence (§5 and the within-specimen inflation in §2 are among this session's most
useful findings). Re-scope encouraging results as hard as disappointing ones. Treat any
`.gitignore`d path as a standing risk — `probe_*.py` is still ignored at `.gitignore:217`, which is
why this session's scripts are NOT named `probe_*`.


---

## 9. Later 2026-08-27 — power run, shuffle control, and the bench50 gate

### 9.1 C6 power run (n=48, independent corpus, seed 20260827)

| arm | mean | Δ vs P48_zero (0.8241) | pos | sign | wilcox | t | outlier-excl |
|---|---|---|---|---|---|---|---|
| **P48_cse_all** | 0.8551 | **+0.0310** | 37/48 | 2.2e-04 | 3.0e-05 | 2.4e-04 | +0.0137 |
| P48_cse_trfeco | 0.8544 | +0.0303 | 28/48 | 0.312 | 0.009 | 0.003 | +0.0128 |
| P48_cse_trfe | 0.8538 | +0.0297 | 32/48 | 0.029 | 0.002 | 0.003 | +0.0098 |

The configuration question is **answered**, and the answer is that configuration does not matter.
See correction 1 and 2 in the banner.

### 9.2 C7 three-way shuffle control — the CSE head is doing TWO different jobs

| arm | mean | vs zero-init |
|---|---|---|
| real `ti`/`ta` correspondence | 0.8701 | +0.0500 |
| `ti`/`ta` shuffled within-segment | 0.8711 | +0.0510 |
| no `ti`/`ta` constraint | 0.8498 | +0.0297 |

shuffled vs real **+0.0010, 6/12, sign 1.0, Wilcoxon 0.91, t 0.81** — indistinguishable.
shuffled vs none +0.0213; real vs none +0.0203.

**On `tr`/`fe` the head supplies genuine correspondence. On `ti`/`ta` it supplies only a coarse
REGIONAL PRIOR** — which vertex is irrelevant, only that *some* pull exists toward the right area.
The no-constraint leg was essential: a shuffled-only control could not have separated "identity
irrelevant" from "these segments contribute nothing".

### 9.3 bench50 reality gate — ran BEFORE any mean-teacher commitment

First result was a **hard crash**: `IndexError: index 2048 is out of bounds` in PointNet++ ball
query. Real meshes are ~1800–3300× the model frame while ball-query radii are absolute, so zero
neighbours are found. Fixed by applying `fitter_3d/utils.py:load_meshes`' own normalisation.
C4/C5 were checked and are NOT affected (synth raw sits within ~3–5% of the training frame).

After the fix, over **all 50** specimens (an earlier 8-specimen figure was alphabetical and
optimistic — **correction**): cycle-consistency ratio median **3.35×**, range 1.10–11.98×, with
only **14/50 (28%)** at or under 2×. Coverage, entropy and segment shares stay healthy — **no
degenerate collapse** — but the signal the deployable arm filters on degrades badly.

Cause analysis over 9 ground-truth-free covariates: **not** damage (`boundary_edge_frac` −0.103),
**not** density (`nn_dist_cv` −0.125), **not** species (within-genus spread 1.132 > between-genus
0.950). **Significant: `aspect_ratio` ρ=−0.502, p=2.0e-04** (good 2.23 vs bad 1.71). The direction
contradicts a distribution-distance story — synthetic aspect ~1.74 matches the *bad* group.
Working hypothesis (~25% of variance, NOT established): elongated specimens have well-separated
limbs, compact ones have limbs bunched, and bunching is the same ambiguity regime as the
circumferential failure.

The clean 14 are **diverse** — 14 distinct genera in 14 specimens, 15× size range, covariate CVs
close to the full set — but **systematically elongated by construction** (aspect mean 2.23 vs 1.86,
min 1.69 vs 1.05). Any model trained on them has untested transfer to compact specimens.

### 9.4 Open, in order

1. **C8 (job 3238010, running)** — does the coarse regional prior transfer to real scans better
   than fine correspondence does? `REAL14_zero` vs `REAL14_cse_all` vs `REAL14_trfeco` on the clean
   14. bench50 has no ground truth, so these are geometric fit metrics (chamfer/fscore/part
   distance) — "did the fit get closer to the scan", never "is correspondence correct".
2. **Scope the mean-teacher pass to a question, not an improvement.** Given 9.2, the right question
   is *"does a coarse regional signal transfer better than fine correspondence, and can the fitter
   be given each on the terms that suit it"* — not "extend the CSE head to bench50".
3. The 36 degraded specimens are neither a fixable data problem nor a distinct species population
   on current evidence. They are the open hard case and must be named as such, not silently dropped.


---

## 10. The equivariance question is not "open" — the geometric case for it got WEAKER

This supersedes any earlier framing of "equivariance unresolved". The distinction matters because it
changes which fix to reach for.

**The original hypothesis required a geometric degeneracy.** The argument for an SE(3)-equivariant
retrain was: circumferential ambiguity is a real geometric degeneracy (a leg is a cylinder, a
cylinder's cross-section has no unique in-plane axis), therefore an ordinary network *architecturally
cannot represent* the disambiguating signal, therefore equivariant features are needed to supply one.

**The LRF check falsified that premise.** Using the toy-verified `circum_gap = (l2-l3)/l2`, real
`tr`/`fe` neighbourhoods score **0.38–0.44 at r=0.10** against a perfect cylinder's finite-sample
floor of **0.031** — 12–14× above it, and ~60–67% of the `body` control. Real ant leg segments are
tapered and flattened, not true cylinders. **The disambiguating signal genuinely exists in the raw
geometry.**

**So the diagnosis changes:** near-chance circumferential retrieval is NOT "this information is
architecturally unrepresentable". It is "this information is present and the trained network is not
extracting or using it". Those call for different fixes:

| diagnosis | fix | status |
|---|---|---|
| information unrepresentable | SE(3)-equivariant architecture | premise falsified — **do not fund on this basis** |
| information present but unlearned | harder negative mining on circumferential pairs; explicit circumferential supervision; longer training | **the live question** |

The second family is substantially cheaper than an architecture change, and is where the evidence
now points. The honest statement for Fabian is not "we don't know whether equivariance would help" —
it is **"the specific geometric reason we thought equivariance was necessary turned out not to hold,
so the remaining question is a training/representation shortfall on a signal that is measurably
there."**

Supporting evidence, and its limits: the rotation-ensemble test was INVALID (rotated inputs degraded
retrieval 5–6×, so the ensemble averaged garbage) and contributes nothing either way. The LRF result
shows the frame is well-DEFINED; it does not show it is STABLE across specimens and noise.

### 10.1 Multi-hypothesis (Task 6): headroom is real, but no selector can reach it yet

Oracle-over-top-5 would cut correspondence error **28.7% (tr) / 31.6% (fe)** — the multimodality is
real and exploitable in principle. But the ranking check, run before committing to build a selector:

| seg | rank_acc | chance | sign p | separation |
|---|---|---|---|---|
| tr | 0.244 | 0.200 | 0.002 | +0.0036 |
| fe | 0.230 | 0.200 | 0.344 | +0.0077 |

Cycle-consistency — which has won three contests as a binary filter and triage signal — picks the
truly-best candidate 24% of the time against 20% chance, and is not significant on `fe`. **It goes
flat when asked to discriminate near-ties, the same entanglement pathology that sank cosine
similarity.** Its coarse-decision track record does NOT transfer to fine-grained ranking.

Corrected framing: **not** "sound idea, selector is the work" — rather, *the headroom is real but no
realisable signal has been shown to access it, and the one signal with a track record performs at
roughly chance in this role.* An open problem with no identified mechanism, not an engineering task.


---

## 11. Final real-scan scope: all 50 bench50 specimens

| group | n | zero | cse_all | Δ | pos | sign | wilcoxon |
|---|---|---|---|---|---|---|---|
| clean14 | 14 | 0.7133 | 0.7868 | +0.0734 | 14/14 | 1.2e-04 | 1.2e-04 |
| rest22 | 22 | 0.5185 | 0.5902 | +0.0717 | 19/22 | 8.6e-04 | 1.2e-04 |
| dirty14 | 14 | 0.2801 | 0.3340 | +0.0540 | 11/14 | 0.057 | 0.011 |

**44/50 improved (88%), mean +0.0672, sign p=3.2e-08, Wilcoxon p=3.4e-09.**

| ratio band | n | mean gain | frac > 0 |
|---|---|---|---|
| 0–2 | 14 | +0.0734 | 1.00 |
| 2–3 | 7 | +0.0562 | 0.86 |
| 3–5 | 14 | +0.0791 | 0.86 |
| 5–8 | 12 | +0.0704 | 0.92 |
| 8–13 | 3 | −0.0044 | 0.33 |

Flat to ratio 8 (Spearman rho=-0.093, p=0.521); 47/50 specimens are below that. The drop above 8 is
**3 specimens** — an observation, not a finding, and not enough to call a cliff.

**Hypothesis retired:** `aspect_ratio` does NOT predict gain (rho=+0.133, p=0.356). It predicts scan
difficulty only. The bunched-limb mechanism proposed as the transfer-limiting explanation is not
supported at the fitter-outcome level, even though aspect_ratio does correlate with
cycle-consistency degradation (rho=-0.502). Degradation and benefit are separate axes.

**What this means for deployment:** the correspondence term can be applied to essentially the whole
corpus rather than a screened subset. No ground-truth-based whitelist is needed at any stage — the
cycle-consistency filter inside the generator is ground-truth-free, and the specimen-level ratio is
useful for setting expectations about fit quality, not for gating inclusion.
