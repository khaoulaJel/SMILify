# Talk script, synced to the videos in SMILify_CS_final.pptx (slides 1-19)

Written for: Khaoula, to speak. About 15 min, about 2,100 spoken words at 140 wpm.
Format per slide: **VIDEO** (what plays, in order, with the cue you speak on), **SAY** (spoken text), **ANCHOR** (the literature the sentence rests on).
Every number below is one that is printed on your slides or in a repo file I opened. Nothing is invented.

Videos on slides 14-15 are a build sequence (points, 2048, 128 centroids, 512, 2048 again). Check in slideshow mode which clip starts on which click and shift my cues by one if needed.

---

## 1. Title (0:20)
**SAY**
"Good morning. I'm Khaoula Jellal. This is my internship at Forschungszentrum Jülich, on registering a body model to 3D ant scans in SMILify.
The sentence I'd like you to keep: **before we trust morphometric measurements, we have to trust correspondence.**"

## 2. The problem (0:05)
"Let me define the problem."

## 3. Dense correspondence under articulation (0:55)
**VIDEO (14 s):** a scan appears with five coloured landmark dots and lines from the thorax. A second ant, in a different pose, fades in beside it.
**SAY**
"I put coloured dots on this ant's joints. Now a second specimen, different pose, different proportions. Where do the same dots go?
Nothing in the scan tells me. A scan is just a surface. It has no vertex identity.
The tempting answer is: take the nearest point. But nearest is geometry, and anatomy is identity. A femur can lie closer to the wrong leg than to its own counterpart.
So: **geometric proximity is not anatomical correspondence.** The rest of the talk is about that sentence."
**ANCHOR** In morphometrics this is homology (Klingenberg 2008). In vision it is why FAUST exists (Bogo 2014): ground-truth correspondence on real scans is hard to get.

## 4. The representation (0:05)
"To make correspondence mean something, we need a shared representation."

## 5. From SMPL to a body-plan-agnostic model (0:55)
**VIDEO (2.5 s, loops):** the ant mesh with its skeleton walking.
**SAY**
"SMPL, then SMAL, showed that a body is not thousands of free vertices. It is a few shape parameters plus joint angles, and a template mesh deformed by them.
SMILify keeps that principle but drops the assumption of a human or a quadruped. Here the template is an ant: a fixed mesh, a full articulated skeleton, and a compact shape space.
Shape plus pose gives a mesh, and the mesh always has the same vertex order. That gives me a coordinate system. **Vertex number i is, by construction, the same anatomical place on every specimen.**"
**ANCHOR** Loper 2015 (SMPL), Zuffi 2017 (SMAL). Fix to your draft: SMPL has 6,890 vertices, not "ten thousand". Say "thousands".

## 6. One shared model coordinate system (0:35)
**VIDEO (22 s), cue on each caption:**
1. *MESH*: say "one fixed mesh"
2. *55 JOINTS*: yellow joints appear. Say "driven by a full skeleton of joints"
3. *SHAPE, 25 SHAPE PCs*: the body changes proportions. Say "and a compact set of shape directions"
4. *POSE*: three poses of one ant. Say "pose moves the joints"
5. *SHAPE SPACE*: three specimens. Say "so every specimen is a point in the same space"
**SAY** (over the clips)
"Mesh. Joints. Shape directions. Pose. Different ants, same coordinate system.
That's why a width or a length becomes comparable across a population. But notice I said *by construction*. Whether a fit actually respects that identity is the real question."

## 7. Registration as an inverse problem (0:55)
**No video. Point at the equation.**
**SAY**
"We see only the scan. We want shape, pose, scale and a small residual deformation, so we search for the model that explains the surface.
Three things make this hard. It's **non-convex**: many local minima. It's **under-constrained**: different parameters give similar surfaces. And it's **correspondence-dependent**: the loss picks the nearest target point, not the right one.
So an optimizer can land on a convincing surface with the wrong anatomy."
**ANCHOR** Same structure as SMPLify (Bogo 2016), which is known to be initialization-sensitive and to fall into local minima. That is why SPIN (Kolotouros 2019) uses a network to initialize the optimizer. I do the same for scans.

## 8. The pipeline (0:50)
**VIDEO (14 s), left to right, one stage per click:**
Raw scan, then Preprocessing, then Anatomical structure, then Initialization, then Correspondence, then Differentiable fitting, then Fitted model. Each stage lists its tools underneath.
**SAY**
"One stage after another. Raw scan. Preprocessing: alpha-wrap and winding-number repair. Anatomical structure: shape diameter function, skeletonization, bilateral symmetry. Initialization: a PointNet-style network that regresses the starting leg pose from the scan. Correspondence: a separate PointNet++ network that gives per-point identity. Then differentiable fitting, with robust weighting that I investigated. And finally the traits.
I did not treat this as a black box. For each stage I asked what information it adds to the inverse problem."

## 9. The bottleneck (0:25)
"Here is the contradiction. **The geometry fits. The anatomy does not.** The surface can be close to the scan while the skeleton inside is in the wrong place. Everything that follows tries to explain that."

## 10. Injecting anatomical structure (1:10)
**Three clicks, in this order (checked in the slide's animation sequence):**
1. *Surface field* (left, 10 s): the ant coloured by local thickness, thick body in dark blue, thin legs and antennae in purple to orange; a second specimen shows the same pattern; the thin structures are then grouped into appendages.
2. *Anatomical hierarchy* (middle, 8 s): one black root, six branches, each branch fills with six joints.
3. *Proximity is not anatomy* (right, 8 s): three rows of the same point cloud. Row "Truth": every point labelled with its leg. Row "Proximity": labels from nearest-structure assignment, with legs wrongly merged or left grey where they run close.
**SAY**
"The question here is: before the optimizer starts, can we inject anatomical structure into the geometry? I tried two sources of structure, and then show why it matters.
**Click 1.** First, a signal that survives articulation: local thickness, the shape diameter function. Every surface point gets the width of the volume around it. The body is thick, legs and antennae are thin. A second specimen shows the same pattern, because thickness barely changes when limbs move. That already separates the bulk of the body from the appendages. It is an anatomical prior, not a correspondence solver.
**Click 2.** Second, the structure of the body itself. A leg is not a bag of points, it is a chain: one body, six legs, each six joints deep, from coxa to pretarsus. So a point is described by which leg it belongs to and where along that chain it sits. That is the coordinate a skeleton-driven model needs.
**Click 3.** And here is why it matters. On top, the truth: every point labelled with its leg. Below, what proximity alone gives you: where legs run close together or cross, the nearest structure is the wrong leg, so labels merge or drop out. Proximity is geometry. Anatomy is identity. The structure I just built is what lets us ask for the second one."
**ANCHOR** Shapira, Shamir, Cohen-Or 2008: the SDF is pose-oblivious and similar across analogous parts. Do not say it solves correspondence.

## 11. What does the optimizer see? (1:05)
**VIDEOS, left to right:**
- *Measured (7 s):* grey ant. Cue: "distance to the scan, and surface normals".
- *Robust objective (9 s):* a leg's points coloured by residual, with a histogram under it whose tail shrinks. Cue: "big residuals are down-weighted".
- *Staged freedom (8 s):* Body, then Legs, then Joints, then Deform, then Coarse, then Fine highlight in turn while the model locks onto the cloud. Cue: "one thing at a time".
**SAY**
"Here is what the optimizer sees: distance to the scan, surface normals, smoothness, symmetry, joint limits.
Two details. Robust weighting means a few bad scan points shouldn't drag the whole fit: large residuals are down-weighted. I investigated this as a mechanism. And the optimization is staged: body, then legs, then joints, then deformation, then fine detail. Each stage unlocks what the optimizer may explain.
But the key point is what it does **not** see. **No term sees vertex identity**, or the true anatomical location of a point. Chamfer asks 'is there a target point near me?', never 'is that *my* point?'. Everything that follows is about that missing information."
**ANCHOR** Geman-McClure / robust annealing: Barron 2019 (general robust loss, GM is α = -2). Graduated non-convexity: Yang 2020.

## 12. Counterfactual diagnosis (1:05)
**VIDEO (17.5 s), five bars rise on cue:**
1. *Baseline*: "identity is not recovered: only a tiny fraction of vertices land on their true counterpart"
2. *+ correct anatomical region*: "even a perfect part decomposition is capped low: most error is inside a part"
3. *+ near-exact correspondence*: "segment accuracy improves, but the articulation problem remains"
4. *+ expert joint information*: "joint error drops clearly: the missing information is the joints"
5. *+ joints everywhere but one region*: "the withheld region shows no detectable improvement"
**SAY** (let the slide carry the values; do not read them)
"Instead of asking whether a trick helps, I asked what the system is missing. I gave it the answer to one sub-question at a time and measured what error remains.
Baseline: almost no vertex lands on its true counterpart.
Give it the correct *part*: the ceiling is still low, so most of the error is *inside* a part, not between parts.
Give it near-exact correspondence: it improves, but the articulation problem remains.
Give it expert joints: joint error drops clearly. The missing information is the skeleton.
Last bar: give joints everywhere but one region. That region shows no detectable improvement. **Anatomical information is local; the objective does not spread it.**
So the next step cannot be a global trick. It has to be dense, per-point identity."
**ANCHOR** Oracle ablation is the standard way to bound a component (as in the diagnostic studies of pose estimators). Fitzpatrick 2009: fit error is uncorrelated with target error, so you need an independent reference.

## 13. From a global descriptor to per-point features (0:40)
**VIDEO (8.5 s):** blue points, label "points", then a grey "shared MLP" column appears with an arrow.
**SAY**
"This is where learning comes in. PointNet applies the same small network to every point and pools with a symmetric function, so the result doesn't depend on point order.
That gives one vector for the whole object. For initialization that's what you want: one starting pose per scan.
But for correspondence one vector is not enough. I need an answer for every single point."
**ANCHOR** Qi et al. 2017 (PointNet).

## 14. Per-point features: the PointNet++ encoder (0:55)
**Video (10 s), in order:** input points; sa1 (many centres, balls of three sizes); sa2 (fewer centres, bigger balls); sa3 (group everything into one vector); bars: linear probe on frozen features reads leg identity and chain position.
**SAY**
"So instead of squeezing the whole scan into one vector at once, PointNet++ does it gradually, and, crucially, it keeps every level.
We start from the scan as a set of points. First level: pick many centres, and around each one draw balls of three different sizes. A small PointNet summarizes each ball, so every centre now carries a feature for its neighbourhood at three scales.
Second level: fewer centres, bigger balls. The view goes from a piece of a leg to a whole leg.
Last level: group everything into a single vector. That is exactly the global descriptor from the previous slide, now just the top of a pyramid.
And here is why the levels below it matter. If I freeze the network and train only a linear probe on these features, leg identity and position along the chain can already be read out at the middle levels. So the anatomy is in the features, below the pooled vector, and that is what I want to keep.
But those features live on a few coarse centres, not on every point."

## 15. Dense per-point representation (0:35)
**Video (9 s), in order:** coarse centres coloured by leg; the next level; every point, with the caption "3-NN inverse-distance interpolation".
**SAY**
"So the decoder reverses the pyramid. Start from the coarse centres, coloured here by leg. Move up one level, and then up to every point.
For each finer point, take its three nearest coarser centres and average their features, weighted by inverse distance, so the closer centre counts more. Skip connections bring back the fine detail from the encoder, and a small network refines the result.
Now every point of the scan carries a feature that already knows which leg it belongs to and roughly where along it.
But 'which leg, roughly where' is still not an identity. For correspondence I need: which exact place. That is the next slide."

## 16. Learning vertex identity (1:45)
**Three clicks, in this order (checked in the slide's animation sequence):**
1. *Why not a scalar* (left, 8 s): the unrolled leg as a flat rectangle. "position only" is a left-to-right gradient; "position + angle" adds a bottom-to-top gradient, with a colour key.
2. *Contrastive* (middle, 4 s): unit sphere drawn as a circle. Large outlined circles are vertex keys, small dots are point embeddings; lines pull each dot onto its own key.
3. *Identity handoff* (right, 7 s): the scan coloured by embedding; the template coloured by vertex ("same colour = same vertex"); then Points, Network, State, Fitter with a loop back.
**SAY**
"A feature that says 'leg three' isn't enough. I want it to say *which place on leg three*. Three steps.
**Click 1.** Why not just give each point one number along the leg? Here the leg is unrolled into a flat sheet. Colour it by position along the axis only, and every point at the same position gets the same colour, however far round the leg it sits. But a leg is a tube, a two-dimensional surface, and one number cannot address a two-dimensional surface. Add the angle around the tube, and every point gets its own colour. Axis alone is not enough; you need axis and angle.
**Click 2.** Now the learning. Every template vertex gets a learned key, the large outlined circles, on a unit sphere. Every scan point is mapped to a query, the small dots. Training pulls each query toward the key of its true vertex and pushes it away from all the others. That is a contrastive, InfoNCE-style loss. Watch the dots travel until each one sits on its own key.
**Click 3.** At test time this gives every scan point an embedding, shown here as colour. Same colour means same vertex on the template. So correspondence becomes nearest key: match by colour, not by nearest surface. Every vertex has its own key, so the result is a map over the whole template. I keep only matches that are cycle-consistent, from scan point to vertex and back to the same point, which needs no ground truth. And that identity is handed to the fitter: it is the information the counterfactual slide showed was missing.
Fed to the fitter as direct vertex identity, it recovers a large share of what ground-truth labels give on synthetic data, improves both part-level and leg-level accuracy, and on real scans, where no ground truth exists, it improves the fit on nearly every specimen."

## 17. Validation (1:15)
**VIDEOS, left to right, one per panel:**
- *Round trip (10 s):* ant model, then a "generated target" point cloud, then the "fit". Caption "vertex i must land on vertex i". A curve (noise added to a perfect match vs % of vertices correct) is drawn, and an orange point appears: "fit 4.8%".
- *Expert joints (11 s):* skeleton on an ant; bars grow per region: coxa 12, leg proximal 13, body axis 24, mandible 43, antenna 55, leg distal 56 (median joint error, % body length). Badges: "11 expert-annotated specimens", "12 strategies x 3 seeds = 396 fits", "pre-registered, frozen before analysis".
- *Head width (7.5 s):* scatter, SMILify vs reference in mm, points on the diagonal. Badges: "20 workers, mean error 4.1%", "scale-free ratio: concordance 0.86".
**SAY**
"Three tests, each with a different question.
**First, synthetic identity.** I generate a target from the model itself, so I know every vertex's true identity, and ask whether the fit recovers it. The blue curve calibrates what noise does to even a perfect match. The orange fit sits at the very bottom of it. So the model can *represent* the target, but the fitting procedure does not necessarily *recover* the generating correspondence. That is the problem, measured with exact ground truth.
**Second, expert anatomy.** Specimens whose joints were annotated by hand, as an independent reference. Joints near the body are accurate. Mandibles, antennae and distal legs are not. Reliability depends on the structure. This benchmark was pre-registered and frozen before analysis.
**Third, measurement.** Head width against the reference measurements supplied with the dataset, for a set of workers: the points sit on the diagonal, and the scale-free ratio agrees well too.
Geometry, anatomy, measurement. Three different questions."
**ANCHOR** Pre-registration is the standard fix for analysis flexibility. Bland-Altman / concordance (Lin 1989) is the standard way to compare against a reference method.
**Correction to your draft:** panel 1 is the round trip with exact ground truth, not "robustness to increasing corruption".

## 18. What the work gave SMILify (0:35)
**VIDEO (14 s):** the pipeline from slide 8 builds again, each box listing its components.
**SAY**
"It's one stack, raw scan to measurement, and these are components I built and tested. Preprocessing and repair. Anatomical structure. A learned initialization. Learned correspondence. Robust differentiable fitting. And validation.
What's reusable is not one algorithm. It's a set of components and tests that let us ask where a registration fails and why."

## 19. Beyond ants (0:30)
**No video. Point at the three boxes.**
**SAY**
"The specimen changes, but the computational problem remains: recover structured 3D state from incomplete observations, and validate the correspondence before trusting the parameters.
Define the anatomy, learn shape and pose from your own data, validate before use.
Thank you. I'm happy to take questions."

## Timing
| Slides | Min |
|---|---|
| 1-2 | 0:25 |
| 3 | 0:55 |
| 4-6 | 1:35 |
| 7-9 | 2:10 |
| 10-12 | 3:15 |
| 13-16 | 4:00 |
| 17 | 1:15 |
| 18-19 | 1:05 |
| **Total** | **14:40** |

You have about 45 seconds spare for pauses. If you run long, cut slide 6's narration to one sentence and slide 11 to two terms.

## If you get lost on a slide (one sentence each)
- 3: nearest is not same.
- 5: shape plus pose gives a mesh with fixed vertex order.
- 7: three reasons the fit can be wrong.
- 12: give the answer to one question at a time and see what's left.
- 13: PointNet gives one vector (initialization); 14: PointNet++ gives local-plus-context features per point (correspondence).
- 15: copy features back from the nearest coarse points.
- 16: each vertex has a key; each point finds its nearest key.
- 17: exact truth, expert truth, a trait against the dataset's reference.

## Likely questions
1. **"Why PointNet++ and not a transformer or DiffusionNet?"** Scans are noisy, partial, fragmented, with no reliable connectivity. DiffusionNet and functional maps are intrinsic and need a clean mesh; they also cannot separate left from right without orientation tricks (Donati 2022). PointNet++ works on raw points.
2. **"Why not just run NICP or LoopReg?"** Both are trained on humans with huge MoCap data. There is no ant dataset. I use the model itself to generate training pairs with exact identity.
3. **"What is new here?"** The diagnosis: oracle ablations that locate the missing information, plus the validation stack. The embedding head is an application of known ideas to a new body plan.
4. **"Is PointNet++ your initializer?"** No. The initializer (`LegPoseRegressor`) is PointNet-style: shared MLP, max-pool, a 6D rotation head per leg joint, trained on synthetic scans. PointNet++ is the separate segmentation/correspondence network, which needs per-point features that a single pooled vector cannot give.
4b. **"Is learned correspondence deployed as a dense map?"** Yes, via the CSE head: a learned key per template vertex, nearest-key retrieval, cycle-consistency filter, fed to the fitter as direct vertex identity (`diagnostics/anatomical_pose_init/LAB_RECORD_correspondence_20260827.md`). Independent 48-specimen synthetic corpus: segment accuracy +0.031 (37/48, significant on all three tests), leg accuracy +0.058 (34/48, sign p = 0.0055; the old "leg accuracy never moves" null was an n = 12 artefact). The earlier 12-specimen figures (+0.050, "92% of the oracle gain") were superseded; quote only the 48-specimen ones. Real corpus: improves fit quality (not a correctness claim, there is no ground truth) on 44 of 50 specimens, mean +0.067. This is in the repo, not yet in the written report. The earlier within-part descriptor (W-series) failed as a dense map precisely because it had no template key table; the CSE design adds one.
4c. **"Why 16 dimensions?"** A hyperparameter; compact enough to train a key per vertex.
5. **"Why a contrastive loss and not regression?"** Symmetric regions (around a tube) make a regression target ambiguous; a contrastive loss gives a distribution instead of an average (SurfEmb).

## Sources
- [PointNet++, Qi et al. 2017](https://arxiv.org/abs/1706.02413)
- [Continuous Surface Embeddings, Neverova et al. 2020](https://arxiv.org/abs/2011.12438)
- [SurfEmb, Haugaard & Buch](https://arxiv.org/abs/2111.13489)
- [CoE, Zeng et al. 2025](https://arxiv.org/abs/2412.05557)
- [LoopReg, Bhatnagar et al. 2020](https://arxiv.org/abs/2010.12447)
- [SPIN, Kolotouros et al. 2019](https://arxiv.org/abs/1909.12828)
- [A General and Adaptive Robust Loss, Barron 2019](https://arxiv.org/abs/1701.03077)
- [Deep Orientation-Aware Functional Maps, Donati et al. 2022](https://arxiv.org/abs/2204.13453)
- [3D Menagerie / SMAL, Zuffi et al. 2017](https://arxiv.org/abs/1611.07700)
- [FAUST, Bogo et al. 2014](https://openaccess.thecvf.com/content_cvpr_2014/html/Bogo_FAUST_Dataset_and_2014_CVPR_paper.html)
- [Shape Diameter Function, Shapira et al. 2008](https://faculty.runi.ac.il/arik/site/mesh-partition.asp)
