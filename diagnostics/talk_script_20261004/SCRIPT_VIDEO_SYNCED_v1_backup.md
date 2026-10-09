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
"One stage after another. Raw scan. Preprocessing: alpha-wrap and winding-number repair. Anatomical structure: shape diameter function, skeletonization, bilateral symmetry. Initialization: a PointNet++ regressor. Correspondence: learned embeddings. Differentiable fitting with robust annealing. Finally the traits.
I did not treat this as a black box. For each stage I asked what information it adds to the inverse problem."

## 9. The bottleneck (0:25)
"Here is the contradiction. **The geometry fits. The anatomy does not.** The surface can be close to the scan while the skeleton inside is in the wrong place. Everything that follows tries to explain that."

## 10. Injecting anatomical structure (1:00)
**VIDEOS, in this order:**
- *Hierarchy clip (8 s):* black body root, six coloured branches, then each branch fills with six dots. Cue: "one body, six legs, six joints deep each".
- *Truth clip (8 s):* point cloud with each leg a different colour. Cue: "every scan point gets a leg and a position along it".
- *Surface field clip (10 s):* ant coloured blue (thick) to orange (thin). Cue: "thickness".
**SAY**
"Pure geometry is ambiguous, so I added structure at two levels.
First, the kinematic tree. A leg is a chain, coxa to pretarsus, so a point is assigned *a leg and a position along it*, not just a nearest point.
Second, the **shape diameter function**. It measures local thickness: thin where the legs and antennae are, thick in the body. It barely changes with pose, so it is a cheap anatomical prior that survives articulation."
**ANCHOR** Shapira, Shamir, Cohen-Or 2008: the SDF is pose-oblivious and similar across analogous parts. Do not say it solves correspondence.

## 11. What does the optimizer see? (1:05)
**VIDEOS, left to right:**
- *Measured (7 s):* grey ant. Cue: "distance to the scan, and surface normals".
- *Robust objective (9 s):* a leg's points coloured by residual, with a histogram under it whose tail shrinks. Cue: "big residuals are down-weighted".
- *Staged freedom (8 s):* Body, then Legs, then Joints, then Deform, then Coarse, then Fine highlight in turn while the model locks onto the cloud. Cue: "one thing at a time".
**SAY**
"These are the terms: distance, normals, smoothness, symmetry, joint limits.
The histogram shows why the loss is robust. A few bad scan points shouldn't drag the whole fit, so large residuals are down-weighted and annealed from loose to strict.
And the optimization is staged: body first, then legs, then joints, then deformation, then fine detail. Each stage unlocks what the optimizer may explain.
Look at what is absent: **no term sees vertex identity.** Chamfer asks 'is there a target point near me?', never 'is that *my* point?'."
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
"This is where learning comes in. PointNet applies the same small MLP to every point, then pools with a symmetric function such as max. The result doesn't depend on point order.
That gives *one vector for the whole object*, which is good for classification, and useless for 'which vertex is this point?'. For correspondence I need an answer per point."
**ANCHOR** Qi et al. 2017 (PointNet).

## 14. Per-point features: the PointNet++ encoder (0:50)
**VIDEO (10 s):** 2,048 blue points labelled "input", then groups of points form.
**SAY**
"So I use PointNet++. It applies PointNet **hierarchically**. Farthest-point sampling picks centres, a ball query groups the neighbours, and a mini-PointNet summarizes each ball. Repeat at larger radii.
Mine uses multi-scale grouping: many fine centres at several radii, then fewer coarser ones, then a global level. The network sees the leg locally and the body globally.
Two things matter for this problem. It's order-invariant, so a scan has no vertex order. And it's local, so a bent leg still looks like a leg."
**ANCHOR** Qi et al. 2017 (PointNet++, multi-scale grouping). This is also what point-cloud-native correspondence work builds on (CoE, Zeng et al., 3DV 2025).

## 15. Dense per-point representation (0:40)
**VIDEO (9 s):** 128 coarse centroids, coloured by leg. Then 512. Then all 2,048, with the caption "3-NN inverse-distance interpolation".
**SAY**
"After the encoder, features only exist at a few coarse centroids. That is too coarse to match vertices.
So the decoder walks back up, level by level, to the full point set. For each point, take its **nearest** coarser centroids and average their features, weighted by inverse distance, then concatenate the skip features from the encoder and refine with a small MLP.
Now every single point has its own feature."
**ANCHOR** This is exactly the PointNet++ feature propagation rule (k = 3, inverse-distance weights, p = 2).

## 16. Learning vertex identity: the initialization (1:25)
**VIDEOS, in this order:**
- *Position vs angle (8 s):* a flat rectangle coloured by position only (left to right gradient). Then "position + angle" appears below, adding a vertical gradient. Cue: "a leg is a tube".
- *Unit sphere (4 s):* big outlined circles on a ring labelled "vertex key", small dots labelled "point embedding", lines pull the dots toward their key. Cue: "pull to the right key, push from the others".
- *Scan point (7 s):* the full scan coloured continuously by embedding. Cue: "similar colour, similar identity".
**SAY**
"A feature that says 'leg three' isn't enough. I want it to say *which place on leg three*.
There's a catch. A leg is a **tube**. Position along the axis gives a gradient left to right. But points around the circumference share that position. So the descriptor also needs an **angle**: the second gradient, bottom to top.
Then the learning problem. Each template vertex gets a learned **key**, a compact vector on the unit sphere. Each scan point is mapped to a **query** in the same space. Training pulls the query toward its true vertex key and pushes it away from all other keys. That's contrastive learning, an InfoNCE loss, with a learned temperature.
At test time correspondence is **nearest key**, not nearest surface.
And I don't trust every match. I keep only matches that are **cycle-consistent**: scan point to vertex and back to the same point.
Fed into the fit as vertex identity, this recovered **almost all of the improvement** that ground-truth labels give, without needing any ground truth."
**ANCHOR**
- Continuous Surface Embeddings (Neverova et al., NeurIPS 2020): per-pixel embedding plus per-vertex embedding, correspondence by nearest neighbour. Mine is the point-cloud version.
- SurfEmb (Haugaard & Buch): contrastive query/key because symmetric surface regions make a plain regression ambiguous. That is why I use InfoNCE, and why the angle is the hard part.
- Cycle consistency as confidence: the same idea as in learned matching (e.g. LightGlue); I found cosine similarity is a *worse* confidence than cycle distance.
- Closest competitor: LoopReg (Bhatnagar 2020). Say "they learn it end to end on humans; I probe and inject it into an optimizer on an animal rig".
**Do not say:** "the network solved correspondence". Say: "it carries real identity signal, and injecting it gets most of the oracle benefit".

## 17. Validation (1:15)
**VIDEOS, left to right, one per panel:**
- *Round trip (10 s):* ant model, then a "generated target" point cloud, then the "fit". Caption "vertex i must land on vertex i". A curve (noise added to a perfect match vs % of vertices correct) is drawn, and an orange point appears: "fit 4.8%".
- *Expert joints (11 s):* skeleton on an ant; bars grow per region: coxa 12, leg proximal 13, body axis 24, mandible 43, antenna 55, leg distal 56 (median joint error, % body length). Badges: "11 expert-annotated specimens", "12 strategies x 3 seeds = 396 fits", "pre-registered, frozen before analysis".
- *Head width (7.5 s):* scatter, SMILify vs reference in mm, points on the diagonal. Badges: "20 workers, mean error 4.1%", "scale-free ratio: concordance 0.86".
**SAY**
"Three questions, three tests.
**Round trip.** I generate a target from the model itself, so I know every vertex's identity, and ask whether the fit recovers it. The blue curve calibrates what noise does to even a perfect match. The orange fit sits at the very bottom of it: the model can *represent* the target but cannot *recover* its identity. That is the problem, measured with exact ground truth.
**Expert joints.** Specimens with joints annotated by hand. Proximal joints, near the body, are accurate. Mandibles, antennae and distal legs are not. Reliability depends on the structure. This benchmark was pre-registered and frozen before analysis.
**A trait.** Head width against an independent reference on a set of workers: the points sit on the diagonal, and the scale-free ratio agrees well too.
Geometry, anatomy, measurement. Three different questions."
**ANCHOR** Pre-registration is the standard fix for analysis flexibility. Bland-Altman / concordance (Lin 1989) is the standard way to compare against a reference method.
**Correction to your draft:** panel 1 is the round trip with exact ground truth, not "robustness to increasing corruption".

## 18. What the work gave SMILify (0:35)
**VIDEO (14 s):** the pipeline from slide 8 builds again, each box listing its components.
**SAY**
"It's one stack, raw scan to measurement. Preprocessing and repair. Anatomical structure. A learned initialization. Learned correspondence. Robust differentiable fitting. And validation.
What's reusable is not one algorithm. It's a set of components and tests that let us ask where a registration fails and why."

## 19. Beyond ants (0:35)
**No video. Point at the three boxes.**
**SAY**
"Nothing here is specific to ants. The recipe is three steps: **define the anatomy you need, learn shape and pose from your own data, and validate correspondence before you use any parameter as a measurement.**
The body plan changes. The problem, structured 3D reconstruction from incomplete observations, stays.
Thank you. I'm happy to take questions."

---

## Timing
| Slides | Min |
|---|---|
| 1-2 | 0:25 |
| 3 | 0:55 |
| 4-6 | 1:35 |
| 7-9 | 2:10 |
| 10-12 | 3:10 |
| 13-16 | 3:35 |
| 17 | 1:15 |
| 18-19 | 1:10 |
| **Total** | **14:15** |

You have about 45 seconds spare for pauses. If you run long, cut slide 6's narration to one sentence and slide 11 to two terms.

## If you get lost on a slide (one sentence each)
- 3: nearest is not same.
- 5: shape plus pose gives a mesh with fixed vertex order.
- 7: three reasons the fit can be wrong.
- 12: give the answer to one question at a time and see what's left.
- 14: PointNet applied at several scales.
- 15: copy features from the 3 nearest coarse points.
- 16: each vertex has a key; each point finds its nearest key.
- 17: exact truth, expert truth, independent trait.

## Likely questions
1. **"Why PointNet++ and not a transformer or DiffusionNet?"** Scans are noisy, partial, fragmented, with no reliable connectivity. DiffusionNet and functional maps are intrinsic and need a clean mesh; they also cannot separate left from right without orientation tricks (Donati 2022). PointNet++ works on raw points.
2. **"Why not just run NICP or LoopReg?"** Both are trained on humans with huge MoCap data. There is no ant dataset. I use the model itself to generate training pairs with exact identity.
3. **"What is new here?"** The diagnosis: oracle ablations that locate the missing information, plus the validation stack. The embedding head is an application of known ideas to a new body plan.
4. **"Why 16 dimensions?"** Enough to separate about 10k vertices on a sphere at the point-count we use; a larger table is under-trained. Don't oversell; say it is a hyperparameter.
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
