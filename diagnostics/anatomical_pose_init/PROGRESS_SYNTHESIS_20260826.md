# Where things stand: leg-pose init / correspondence investigation (through 2026-08-26)

Informal working synthesis, not a final report — two D1 fits are still queued (`c23g_low`, not
yet run) as this is written. Purpose: one place that says what we tried, what the evidence
actually showed, and what that means for what to do next. Numbers are cited from the source docs,
not re-derived.

## READ THIS FIRST — settled vs. open, for whoever (including future-you) picks this up next

This session went through five real pivots (IK-conversion -> centroid bug found -> furthest-point
fix tried and falsified -> dimensionality diagnosis -> CSE direction proposed). Each had its own
evidence; none of it should need to be re-derived or taken on faith. Draw the line here:

**Settled — safe to build on without re-verifying:**
- Correspondence (not pose) is the real lever. `Dense_GT_oracle`: +0.057 seg_acc, 12/12
  specimens, sign p=0.0002, Wilcoxon p=0.0002, survived every falsification thrown at it (Part 2).
  This used TRUE vertex indices directly — untouched by anything below.
- The backbone (sa1/sa2, per-point features instead of the old collapsed global vector) is sound
  and trains cleanly (B2: clean convergence to val_seg_acc 0.84, no overfitting, Part 3).
- A 1D scalar position (DensePose-style Variant 1, what got built) is STRUCTURALLY insufficient
  to address a leg segment — confirmed by two independent checks (circumferential 3D spread
  11-18% of segment length; class-controlled residual R2 collapses to 0.10/-1.50), not just a
  labeling bug. No free fix exists (no authored UV coordinates anywhere in the model). Closed,
  don't reopen without new evidence (Part 5).
- Centroid-based IK-conversion has a real, large bias (2.03x bone length on the worst segment),
  and furthest-point-per-segment is NOT the fix — it's worse. Don't try that specific fix again
  without a new reason it would behave differently this time (Part 5).

**Open — nothing below this line has been tested, do not assume it works:**
- Whether a CSE-style embedding head (one continuous per-point embedding, matched by
  nearest-neighbor to template-vertex embeddings) actually closes the gap. One narrowly-scoped,
  small-n feasibility signal exists (r=0.547, see Part 5 for exactly what it does/doesn't cover)
  — encouraging, not validating. The embedding head, its loss, and the retrain do not exist yet.
- Whether that r=0.547 signal even replicates on other segments/specimens — check this BEFORE
  spending GPU time on a full retrain, per Part 5's own note.
- What `Network_correspondence`/`GTLabel_IK_oracle`'s D1 fits show, once they land — they're
  centroid-method results only, confounded by the bias above; useful as historical "before"
  context, not as an answer to whether the network's correspondence is good.
- C1 (sampler ablation) and Phase D (bench50 sim-to-real) — not started, not blocked on anything
  above, just not reached yet.

---

## The big-picture arc

We started by asking "can we build a smarter *pose initializer* for the fitter?" and ended up
concluding the real bottleneck isn't pose initialization at all — it's **correspondence** (which
scan point matches which template vertex). Everything below is either evidence that built that
case, or the first attempt at acting on it.

---

## Part 1: What we tried BEFORE believing correspondence was the answer (all closed doors)

### 1a. Hand-built geometric pose estimators — closed, doesn't scale
Three different ways to guess a coherent (whole-leg, coxa-only) rotation from real scan geometry:
PCA direction, cluster-axis direction, tip direction. All three **lose to just leaving the pose at
zero** (0.804–0.835 leg_acc vs. zero-init's 0.867–0.872). Pooling all of them together barely
moved the *best possible* outcome (oracle ceiling 0.906→0.915) while making the practical
GT-free selector picking among them *worse* (75%→58% correct picks). Cross-estimator disagreement
does not predict which legs are wrong either (r≈-0.12 to -0.24, never significant).

**What it means:** more hand-built geometric candidates is a dead end. Diminishing returns, real
cost to selection reliability.

### 1b. Learned linear pose-subspace (PCA over real joint angles) — closed, not enough data
Tried fitting a low-D PCA subspace over real ground-truth chain rotations and doing IK inside it.
Every mode count (2–15) performed at or worse than plain IK. Explained variance was flat, not
concentrated — because there are only **66 usable training rows** (11 held-out specimens × 6
legs). Checked hard for a bigger real corpus before giving up: `joint_limits.py` is hand-authored
by an artist (not data-derived), and the only other real-specimen source (AntScan museum scans)
has *no per-joint ground truth at all* — extracting it would require solving the correspondence
problem this whole investigation exists to fix. **12 `synth_clean` specimens are the permanent
ceiling on any data-driven pose prior in this project**, full stop.

**What it means:** don't try to learn a pose prior from data. There isn't enough of it, and there
never will be without a different kind of ground truth than we currently have any way to get.

### 1c. IK given the true tip position — closed, it's a genuine ambiguity, not a bug
Even with the *true* 3D tip position, IK's mean per-joint error only gets to ~21°. Adding more
constraint points along the chain helps but never closes the gap (21°→19°→17°→16°→14° for
1→5 points). This is real: 5 joints × 3 axes = 15 unknowns, and a position only tells you which
way a bone points, never its twist around its own axis.

**What it means:** single-point (or even few-point) position-based IK has a hard floor around
14–21°, independent of how good the position estimate is. This matters directly for C2 below.

---

## Part 2: The reframe — why we moved to "predict correspondence, not pose"

The disentangling probe (`coherent_multi_estimator_oracle_PROBE.py`) found real leg poses spread
**~50° of bend across the whole chain**, not concentrated at the coxa — so no coxa-only estimator,
however accurate, can ever recover the true per-joint decomposition. That's a structural ceiling,
not an estimation-quality problem.

Meanwhile, `smil_pointnet.py` (the existing learned-init network) has a specific, code-level flaw:
its last layer (`sa3`, `group_all=True`) pools *everything* into one 1024-d vector before decoding
any output — every joint's rotation, all 250+ parameters, comes from the same blended vector.
There's no way for one leg's evidence to stay separate from another's. This matches a known
pattern in human pose estimation (PARE, built for exactly this reason) and points at DensePose's
solution: classify which body part a point belongs to, then regress its position *within* that
part, from features that never got collapsed together.

### The decisive test (ceiling check BEFORE building anything)
Before building a whole network, we tested with an *oracle* — give the fitter perfect
correspondence, see what happens:
- **Group-level oracle** (perfect leg assignment): leg_acc 0.867→0.937. But this only ever
  touches 16.7% of total correspondence error (the between-part slice) — confirmed by tracing
  through `why_partitions_null.py`.
- **Dense per-vertex oracle** (perfect assignment down to the vertex): **seg_acc +0.057,
  12/12 specimens positive** (sign p=0.0002, Wilcoxon p=0.0002) — the first broad,
  non-outlier-driven effect in the entire investigation. This reaches the other 83.3% of the
  error (within-part).
- Checked whether the residual gap (seg_acc capped at 0.874, not 1.0) was just a
  weight-balancing artifact: cranked the oracle term's weight 50x, seg_acc **didn't move**
  (0.870 vs 0.874). So the gap is a real fitter-capacity ceiling, not a tuning issue.
- Also checked the metric's own ceiling: ground truth run through the SAME metric scores 0.974,
  not 1.0 — so "gap to 1.0" was overstating the real remaining gap (~0.10, not ~0.17).

**What it means:** correspondence is a real, validated, broad lever — the strongest evidence in
the whole investigation. But there are two SEPARATE risks going forward: (i) the usual gap
between giving the optimizer the perfect answer and *estimating* it (every prior oracle-vs-learned
comparison in this project has shown a large gap here), and (ii) a real ~0.10 capacity ceiling
that exists even with perfect information — a different problem (shape-space/mesh capacity), not
something a better correspondence estimator can ever fix.

---

## Part 3: Building the correspondence network (B1/B2) — what actually happened

### B1 — architecture
Built `SMILCorrespondenceNet`: same verified PointNet++ backbone as the existing network, but
decodes from `sa1`/`sa2`'s still-localized per-point features (via feature propagation back to
per-point resolution) instead of the collapsed `sa3` vector — directly fixing the diagnosed flaw.
Two heads per point: which leg-segment it belongs to (classification, 37 classes) and where along
that segment (0–1 regression), mirroring DensePose's split.

### B2 — training
Trained on 4000 synthetic specimens (corrected, whole-chain-correlated pose sampler — the old
sampler generated every joint's rotation independently, which doesn't match how real legs bend).
**Result: val_seg_acc climbed 0.80 (epoch 20) → 0.836 (epoch 60) → plateaued at ~0.84 by epoch 80
and stayed flat through epoch 120.** Best checkpoint: epoch 91, val_seg_acc=0.8384. Train/val gap
stayed narrow throughout — this is a real plateau, not overfitting.

**What it means so far:** the network learned something real (0.84 point-classification accuracy
across 37 classes is well above chance), and it converged cleanly rather than crashing or
diverging. Whether 0.84-accuracy point classification actually translates into a *better fitter
outcome* is a completely separate question — that's what C2 exists to answer.

### An infrastructure near-miss worth remembering
Switching from WSL to the cluster, the network's own source file
(`smil_correspondence_net.py`) got silently dropped — it lived in a directory that was
blanket-`.gitignore`'d (meant for vendored third-party code, but by now held first-party modules
too), so it was never committed before the machine switch. Had to be reconstructed from a saved
transcript and verified byte-exact against the existing checkpoint (`strict=True` load, zero
mismatched keys) before trusting it. Fixed the `.gitignore` root cause. Filed as a reminder: any
directory-level ignore rule is a standing risk if first-party code ever lands inside it.

**A second, related risk found the same day:** 40 of the 61 rows `run_audit.py` tracks have no
backing `.npz` on this cluster (only their small JSON result survives — the multi-GB fit outputs
themselves were never carried over, by design, since they're regenerable). `run_audit.py`'s
`main()` used to silently overwrite the whole JSON with only whatever it could recompute that run
— would have deleted 38 historical rows (including the zero-init baseline and both oracle rows
everything here is scored against) the moment it was next run. Fixed to merge instead of
overwrite. Accepted, not fully closed: if the metric code (`confusion.py`/`labels.py`) ever
changes, those 40 rows become unrecoverable without a full refit — a known risk now, not a future
surprise.

---

## Part 4: C2 — turning the network into an actual candidate (in progress)

Converted the network's per-point predictions into IK constraints using the *existing*
`solve_chain_ik_multi` machinery (no new IK code): for each leg, take the points predicted to
belong to each of its 5 non-coxa segments, use their centroid as a position constraint, feed
whichever constraints have enough points into the same multi-constraint solver already built for
the oracle/IK experiments in Part 1c.

**A real bug caught before wasting a GPU run:** first attempt fed the network raw mesh vertices
(dense, ~10,235 points). It **catastrophically collapsed** — predicted almost the entire point
cloud as one single leg. Diagnosed directly (not guessed): tested the same network on a
correctly-sampled training-distribution specimen (2048 uniformly-resampled surface points,
exactly matching how it was trained) and got a sane 0.80 accuracy with a normal class
distribution. The network's fixed-radius neighborhood queries are tuned to the point *density*
`sample_points_from_meshes` produces, not raw vertex density — an I/O-style mismatch, same
family of bug this project's own conventions exist to catch. Fixed by resampling 2048 points the
same way training did. After the fix: 23/30 constraints used per specimen on average (up from
6/30), raw pose error 33.4° mean — right in line with the other real-scan-derived candidates
(PCA-coherent 32.9°, cluster 33.3°, tipdir 31.3°).

**CAVEAT confirmed 2026-08-26, before either job below landed — read this before trusting either
result:** the centroid-based constraint construction has a confirmed, quantified geometric bias.
Measured directly (centroid of each segment's true-labeled points vs. the exact joint position
`solve_chain_ik_multi` is told to match, straight from each specimen's own mesh via the joint
regressor, no H0-fit dependency): the centroid is **2.03x the bone's own length away** from the
target for the trochanter segment, 0.29-0.43x for femur/tibia/tarsus, and only well-behaved for
the tip (0.10x, where "furthest point" and "centroid" nearly coincide anyway). A raw-pose-error
near-tie between the network (33.42 deg) and a GT-label oracle (32.46 deg) — both using this same
biased centroid conversion — is therefore NOT evidence that "the network is near GT-quality and
IK-conversion is a generic ceiling." It's equally, probably more, consistent with **both hitting
the same fixable bug**, which would mask any real difference in correspondence quality between
them. **The two D1 fits below (`Network_correspondence`, `GTLabel_IK_oracle`) are centroid-method
results only** — useful as a "before" baseline once a corrected (furthest-point-per-segment)
conversion is built and tested, but not yet usable as evidence about whether the network's
correspondence predictions are actually good. Do not write either one up as a settled answer to
"does the network add anything." See the follow-up furthest-point work below for the real test.

**Status: the actual D1 fit (does this candidate produce a better final mesh?) is queued on the
cluster right now.** Once it lands, it gets scored against the B3 bar that was written down
*before* any of this network's numbers existed:
- Real-effect floor: leg_acc ≥ 0.886, seg_acc ≥ 0.834, AND both sign-test and Wilcoxon p<0.05
  across all 12 specimens.
- Strong-effect tier: leg_acc ≥ 0.904, seg_acc ≥ 0.847, same significance requirement.
- Anything short of that, or not significant by both tests even if the mean looks good, counts
  as a negative result and gets reported as one — this project has been burned twice already by
  reading too much into an aggregate mean that didn't survive a sign test (see Part 1a, and the
  retracted "targets hard cases" read of the group oracle).

---

## Part 5: the real diagnosis — DensePose-style's ceiling is dimensionality, not labels (2026-08-26, same day)

Before trusting either queued D1 fit, two things were checked that changed the whole picture.

### The centroid-conversion bug (confirmed, then a proposed fix falsified)
Measured directly (centroid of each segment's true-labeled points vs. the exact joint position
`solve_chain_ik_multi` is told to match, straight from each specimen's own mesh): the centroid is
**2.03x the bone's own length away** from the target for the trochanter segment, 0.29-0.43x for
femur/tibia/tarsus. A raw-pose-error near-tie between the network (33.42 deg) and a GT-label
oracle (32.46 deg) — both using this biased conversion — is NOT evidence the network is near
GT-quality; it's equally consistent with both hitting the same bug. **Tried the obvious fix**
(furthest-point-along-chain per segment, reusing `estimate_leg_tip`'s existing logic) — **it made
every segment worse**, not better (tr: 2.03x->3.85x). Falsified, reported as the negative result
it is, not swept aside.

### The real finding: `chain_pos` is degenerate, and fixing it wouldn't be enough anyway
Checked directly: every vertex in a segment class shares the EXACT SAME `chain_pos` value (all 275
`l1_r_tr` vertices = 0.2, all 153 `l1_r_fe` vertices = 0.4). This has been true since B1's
inception — the "within-segment position regression" head was never actually predicting a
within-segment position; it was predicting a constant redundant with the classification. This
also retroactively undercuts A3's original optimistic R2 (0.71-0.84): that number could be almost
entirely explained by class-separability alone, not genuine positional information, since the
target IT predicted was itself just a proxy for class.

**Re-ran the check properly, controlling for class membership** (B2's own trained sa1/sa2, real
synth_clean specimens, genuine continuous target = graph-distance-from-coxa on the rest mesh, not
the degenerate label): R2 on the raw target was high (0.88-0.92) but almost entirely explained by
between-class variance (residual is only 6-8% of total variance). **R2 on the class-controlled
residual: 0.10 (sa1), -1.50 (sa2, worse than predicting the mean).** The trained backbone carries
essentially no linearly-readable within-segment position signal, because nothing ever asked it to.

**But the deeper problem survives even a hypothetically-fixed label:** a leg segment is a tube — a
2D surface, not a 1D curve. Checked directly: within a narrow chain_pos band (21 real vertices,
matched to within 0.02), true 3D spread reached 0.046 (**11-18% of the segment's own length**) —
real, circumferential ambiguity a single scalar cannot resolve, however well it's labeled. Checked
for a free fix (authored UV/texture coordinates on the mesh, which would hand DensePose-style a
real circumferential coordinate for free): **none exist**, in the model dict or in any exported
`.obj` (`grep -c "^vt "` = 0). Hand-deriving one would be real, nontrivial work (the leg's
cross-section isn't a clean circle and changes shape near joints).

**Verdict on DensePose-style (Variant 1, what got built):** closed, for a specific, evidenced
reason — not "didn't quite work." A 1-D position value structurally cannot uniquely address a 2-D
tube surface, independent of label quality or training. Two independent fix attempts (better
label, better point-selection statistic) were the wrong kind of fix for a dimensionality problem.

### The path forward: CSE-style (Variant 2), and one cheap encouraging sign
PHASE10_DESIGN always named CSE (one continuous per-point embedding, matched by nearest-neighbor
to template-vertex embeddings, no hand-designed coordinate system at all) as the alternative to
DensePose's exact failure mode. Before committing GPU time to building it: checked whether the
EXISTING, already-trained backbone (no retraining) shows any structure a CSE head could exploit.
**Exactly what was measured, stated precisely (this number will be used to justify real GPU spend
next time, so precision matters more than "small-n" alone conveys):** took the 44 points from a
dense 20000-point sample of ONE specimen (`synth_000`) that landed, via nearest-true-vertex
correspondence, inside ONE circumferential band (21 template vertices of `l1_r`'s `fe` segment,
chain_pos matched within 0.02 of the band's median — the same band used for the earlier spread
check). For every pair among those 44 points (946 pairs, but only 44 independent points —
pairwise-distance correlation does NOT have 946 independent degrees of freedom, effective n is
much closer to 44), computed:
  - `true_d`: straight-line 3D Euclidean distance (posed mesh, NOT geodesic) between the TRUE
    corresponding vertices — not a pure circumferential coordinate; since the band is narrow in
    chain_pos, most of this distance should reflect circumferential separation, but that is not
    independently verified, some residual along-chain spread within the +/-0.02 window is possible.
  - `feat_d`: Euclidean distance between each point's frozen `sa1` feature, APPROXIMATED as its
    nearest FPS-centroid's feature (sa1 pools onto only 512 centroids from the 20000-point input,
    so this is a coarse proxy for a per-point embedding, not the network's actual per-point
    feature — distinct points can and do collapse onto the same centroid's feature).
  - Pearson correlation(`true_d`, `feat_d`) = **0.547**.

**What this does and does not establish:** a positive pairwise-distance correlation is suggestive
that SOME circumferential signal survives into frozen `sa1` features, but it is NOT the same
property CSE actually needs at inference (nearest-neighbor retrieval accuracy — "is a query
point's closest embedding actually its true nearest vertex" — a ranking property, not a linear
correlation). **Scope, stated explicitly rather than left as "small":** n=44 points, 1 band, 1
segment (`fe`), 1 leg (`l1_r`), 1 specimen (`synth_000`), 1 backbone layer (`sa1`, via the coarse
centroid-proxy above), zero replication across other segments/specimens/bands. Untested: whether
this correlation holds at `sa2`, on other segments (especially `tr`, the worst-offending segment
in every check so far), on other specimens, or whether it survives as an actual retrieval-accuracy
number rather than a distance correlation. Treat this as "worth a follow-up feasibility check
across more segments/specimens before the retrain," not as validation of the CSE approach itself.

**Not yet done:** the actual CSE embedding head + metric-learning loss + retrain. This is
identified as the concrete next step, not started this session (a real GPU commitment on an
already-over-quota account, for a component untested at this resolution) — see "what's next"
below.

### On the two D1 fits queued earlier (`Network_correspondence`, `GTLabel_IK_oracle`)
Left running (compute already sunk, not wasted) but **their conclusion value changed**: they can
no longer be read as "does the network's correspondence help the fitter" — both use the biased
centroid conversion, so at best they show what the CENTROID METHOD achieves, not what the
network's own correspondence quality is capable of. Useful as a "before" baseline once the CSE
path exists, not as an answer to the original question.

---

## What this all adds up to, plainly

**Works:**
- Correspondence-as-target is real and validated (dense oracle: 12/12 specimens, both
  significance tests, survived every falsification thrown at it) — this is untouched by
  everything in Part 5, since `Dense_GT_oracle` used true vertex indices directly, no
  representation/dimensionality machinery involved at all.
- The architectural fix (per-point features instead of collapsed global vector) trains cleanly
  and does carry real, usable structure — just not through the head it was given (see the 0.547
  frozen-feature circumferential correlation, Part 5).
- Feeding the network anything other than a matched-distribution point sample breaks it
  outright (a real gotcha to remember if this network is ever pointed at bench50 real scans).

**Doesn't work / closed, with a specific reason each time:**
- Hand-built coxa-only geometric estimators, individually or pooled (Part 1a).
- A learned pose-subspace prior — not enough real ground-truth data, permanently (Part 1b).
- Cross-estimator disagreement as a confidence signal (Part 2).
- Centroid-based IK-conversion (2.03x bone-length bias on the worst segment) and the obvious fix
  for it, furthest-point-per-segment (made every segment worse, not better) — Part 5.
- DensePose-style output representation (discrete class + 1D within-segment position): closed for
  a specific, structural reason — a 1D scalar cannot uniquely address a 2D tube surface,
  independent of label quality (confirmed: `chain_pos` was degenerate, AND fixing it wouldn't
  have been enough — real circumferential 3D spread of 11-18% of segment length persists even
  with a perfect continuous label). No free fix available either (no authored UV coordinates
  anywhere in the model). This also means A3's original R2=0.71-0.84 was likely an artifact of
  the degenerate label (near-fully explained by class membership, not genuine position info) —
  a pillar of the original case for this architecture direction didn't hold up on re-examination.

**Open / the actual next step:**
- CSE-style (one continuous per-point embedding, matched by nearest-neighbor to template-vertex
  embeddings, no hand-designed coordinate system) is the identified fix — it's what the
  literature built specifically for DensePose's exact failure mode, and a cheap, narrowly-scoped
  check (frozen, never-retrained `sa1` features correlate with true 3D distance at r=0.547 among
  44 points in ONE circumferential band of ONE segment of ONE specimen — see Part 5 for exactly
  what this does/doesn't establish) suggests there's real structure worth checking further, not
  yet grounds for confidence on its own. **Before any retrain**: repeat this same check across
  more segments (especially `tr`, the worst offender everywhere else in this investigation) and
  specimens, and as an actual nearest-neighbor retrieval accuracy, not just a distance
  correlation. **Not yet built at all**: the embedding head, the metric-learning loss, and the
  retrain itself. Real GPU commitment on an already-over-quota account for an untested
  component — the right next session's starting point, not something to rush into at the tail of
  this one.
- The two centroid-method D1 fits (`Network_correspondence`, `GTLabel_IK_oracle`) will still
  finish and land — useful as a documented "before" data point for the centroid method
  specifically, not as an answer to whether the network's correspondence quality is good, since
  both are confounded by the same conversion bug.
