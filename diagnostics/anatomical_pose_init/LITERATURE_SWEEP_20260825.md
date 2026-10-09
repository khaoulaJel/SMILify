# Literature Sweep: Alternative Pose-Initialization Strategies (2026-08-25)

Requested by Fabian: a full, unbiased sweep of the research landscape around pose initialization
for articulated 3D body-model fitting, comparable in ambition to the existing `smil_pointnet.py`
trials (a hand-written end-to-end PointNet regressor predicting the *full* SMIL parameter set —
pose, shape, translation, global rotation — in one pass, unfinished/no trained checkpoint), to
inform a discussion about whether SMILify's current, narrower leg-pose-only initializer
(`train_leg_pose_regressor.py`, evaluated G1→G1d, reaching parity with zero-init — see
`HANDOFF_20260824.md`, `RESULTS_bench50_G1_vs_G3.md`) should be scoped up, and if so, how.

**Method**: three independent research agents ran in parallel, deliberately scoped to not know
about or anchor on each other's findings, each covering a different cluster of the field (full
end-to-end regression; learned correspondence + multi-hypothesis + kinematic-aware architectures;
category-agnostic/insect-specific/classical methods). Each was instructed to verify papers exist
via live search rather than recall, judge relevance against SMILify's actual constraints (novel
non-human/non-quadruped category, small synthetic training corpus, point-cloud-only input with no
image/texture, no ground truth on real specimens, six near-identical articulated leg chains), and
flag gaps honestly rather than stretch adjacent work to fit. 76 search/fetch operations total,
~60 distinct methods/papers surveyed. Items the agents could not independently verify against a
primary source are flagged explicitly in §7 — treat those as leads, not citations, until re-checked.

---

## 1. Executive summary

- **The central obstacle SMILify has found — six near-identical articulated leg chains creating
  correspondence ambiguity — has no solved precedent anywhere in the literature surveyed.** All
  three independent sweeps converged on this same gap without being told to look for it. The
  closest analogs (2-fold human left/right symmetry, 4–5-fold hand-finger self-similarity, rigid
  single-object rotational symmetry) are all meaningfully easier problems. This is worth stating
  to Fabian plainly: this is not a matter of finding the right existing method to import.
- **SMILify's current two-stage design (learned network seeds a gradient-descent optimizer,
  rather than the network replacing it) is the dominant pattern in current SOTA, not a
  compromise.** SPIN, PyMAF, 3D-CODED, IPNet/LoopReg, and a 2025 point-cloud self-improving-loop
  paper all pair fast regression with optimization or fitting-loop refinement. The open question
  is *scope* (leg-only vs. full-body) and *architecture* (independent-joint regression vs.
  chain-aware), not the two-stage philosophy itself.
- **Going further toward `smil_pointnet.py`'s full end-to-end ambition is not well-supported by
  the evidence right now.** Every full-parameter regressor that generalizes well in the
  literature (HMR2.0, AniMer, MagicPony, Farm3D) depends on either massive training data or a
  pretrained foundation-model backbone (DINO, diffusion models) trained on natural
  images/humans/common animals. No such backbone or comparable data volume exists for ants.
  This is a real, evidence-backed argument against scaling up the network before addressing the
  symmetry problem, not just institutional caution.
- **The one architecturally closest peer found to `smil_pointnet.py`** — Jiang et al.
  (ICCV 2019), point-cloud → SMPL via PointNet++ — needed to add an explicit attention module
  just to resolve unordered-point-to-ordered-joint correspondence, for a human skeleton with far
  milder symmetry than six ant legs. That's direct evidence a plain PointNet-to-full-parameters
  regressor would need comparable (or stronger) correspondence-resolving machinery to work
  reliably on ants.
- **The forward-kinematics error-compounding principle is a known, general geometric fact in the
  field**, not something specific to any one paper: parent-joint (proximal) errors compound down
  a kinematic chain during forward kinematics, so they're expected to damage the final result
  more than distal errors, for architecture-independent geometric reasons. This directly
  corroborates the project's own D/E/F perturbation finding and is worth citing as independent
  confirmation in the basin-map writeup.
- **The "selection problem"** — picking the right pose among several plausible hypotheses without
  ground truth — is also an open problem in the wider field, not something unique to SMILify.
  Current answers (reprojection-error proxies, self-reported confidence heads) don't transfer
  cleanly, because SMILify's natural analogue proxy (Chamfer/surface fit) is exactly the signal
  already shown *not* to discriminate the correct leg-assignment basin from a wrong one.
- **Several concrete, learning-free or low-cost candidate techniques exist** that haven't been
  tried yet (§5) — worth prototyping before any larger network redesign.

---

## 2. The central finding: no solved precedent for repeated-limb correspondence ambiguity

Stated independently, in near-identical language, by all three sweeps:

> *"No paper was found that directly addresses feedforward regression under >2-fold
> near-identical articulated-chain correspondence ambiguity... nothing that is 'the hand-pose
> finger-ambiguity paper, but for 6 legs.'"* — end-to-end regression sweep

> *"None of the surveyed methods has a built-in solution to disambiguating near-identical
> repeated kinematic chains... the closest partial answers are candidate building blocks rather
> than drop-in solutions."* — correspondence/multi-hypothesis sweep

> *"None of the classical correspondence-based methods... have any inherent mechanism to break
> symmetry between the six near-identical leg chains — that ambiguity is a correspondence
> problem, not a training-data problem."* — category-agnostic/classical sweep

The nearest analogs found, and why each falls short:

| Analog | Symmetry order | Why it's easier than ants |
|---|---|---|
| Human left/right limb ambiguity | 2-fold | A single binary choice; solved mostly by soft limb-length-symmetry constraints on an otherwise-good reconstruction, not by resolving *which* points belong to *which* limb from scratch. |
| Hand pose (finger self-similarity) | 4–5-fold | Fingers have a fixed, non-permutable spatial ordering around the palm and very different length distributions — much stronger distinguishing priors than six legs mounted in bilateral pairs along a body axis. Mitigated via hierarchical per-finger sub-networks and strong positional context. |
| Rigid-object rotational symmetry (e.g., symmetric bottles) | Continuous/discrete, but single rigid body | Solved via symmetry-aware encodings (e.g., SymCode) or learned correspondence *distributions* rather than one-to-one matches — the "one-to-many correspondence" framing is the closest conceptual match found, but developed for one rigid part, not six independently articulated ones. |

**Implication**: this is a genuinely open sub-problem, not an off-the-shelf import. Any strategy
adopted will need to be adapted or newly designed around this specific structure, and that's a
legitimate, reportable finding in its own right — not a failure of the search.

---

## 3. What the field validates about the current approach

Three separate lines of evidence converge on the same architectural verdict:

1. **Two-stage (fast regression → optimizer/fitting-loop refinement) is the dominant modern
   pattern**, across both image-based (SPIN, PyMAF) and point-cloud-based (3D-CODED, IPNet,
   LoopReg, and a 2025 self-improving-loop paper for partial point clouds) human-body regression.
   SMILify's leg-pose-network-feeds-D1-optimizer design matches this pattern already.
2. **HybrIK's twist-and-swing decomposition** — predict 3D joint *positions* (a more robust,
   lower-ambiguity target) via the network, then solve joint *rotations* analytically via inverse
   kinematics from those positions relative to the parent joint, rather than regressing rotations
   directly — is architecturally close to what a chain-respecting redesign of the current leg-pose
   network could look like, and is one of the most consistently-cited ideas across two of the
   three sweeps.
3. **Full end-to-end everything-at-once regression is a much harder target than initialization**,
   confirmed by literature rather than just intuition: every full-parameter regressor achieving
   strong generalization leans on resources (massive data or pretrained foundation-model
   backbones) that don't exist for ants. `smil_pointnet.py`'s ambition is sound in principle, but
   the field's evidence suggests it would need either a much larger synthetic corpus, a
   correspondence-resolving mechanism the current architecture lacks, or both, before it could be
   expected to work — consistent with the fact it was left without a trained checkpoint.

---

## 4. Concrete candidate directions, roughly ranked by cost and evidential support

**Learning-free, essentially free to try (no training, no architecture change):**

- **Per-leg inverse kinematics** (Kim et al. 2015, arthropod leg tracking): if leg-tip and
  body-attachment points can be identified geometrically on a scan, solve each leg chain's joint
  angles analytically via IK given known segment lengths and joint-limit priors from the insect
  biomechanics literature. Sidesteps blind angle regression and any learned correspondence step
  entirely — worth prototyping as a baseline before further network investment.
- **L1-medial skeleton extraction** (Huang et al. 2013) as a raw geometric first pass — genuinely
  learning-free, could be run directly on each scan as a structural cross-check or coarse
  initializer, then matched to the known rig topology.
- **PCA / global-registration coarse pre-alignment** — trivial, cheap, already generically useful
  for the rigid global-orientation piece (the deferred G2/G4 hypothesis), though it does nothing
  for per-leg articulation.
- **CPD/GLTP articulated point-set registration** — a real classical option, but flagged with the
  same caveat as everything else here: it has no built-in symmetry-breaking mechanism, so it
  would likely need to be paired with one of the ideas below to avoid the same wrong-leg-basin
  failure mode.

**Architecture ideas for the existing network (moderate cost — retraining, not a new system):**

- **HybrIK-style position→analytic-rotation decomposition** (see §3) — predict positions, solve
  rotations by construction-respecting IK, rather than the current architecture's likely
  independent-per-joint rotation regression.
- **Explicit skeleton-graph encoding** (Pose Anything, ECCV 2024) — encode the known kinematic
  topology directly as a graph structure in the network rather than treating each leg joint as an
  independent prediction target, specifically to help break symmetry between structurally
  identical chains.
- **Correspondence-resolving attention module** (Jiang et al., ICCV 2019) — the closest existing
  peer to `smil_pointnet.py` found this necessary even for milder human symmetry; a similar
  explicit step (resolve "which points belong to which leg" before regressing angles) is a
  plausible missing piece in the current pipeline.
- **SE(3)/rotation-equivariant construction** (ArtEq, 2023; the point-cloud equivariance work
  cited in the end-to-end sweep) — removes global-rotation ambiguity from the learning problem by
  construction, a cheap way to remove one whole axis of variation a small-data network otherwise
  has to learn to be robust to.

**Bigger, longer-horizon directions (real research investment):**

- **TANet** (CVPR 2024) and **ArtEq** (2023) — the two existing systems that come closest to
  covering all four of SMILify's actual constraints (known template, unlabeled target, kinematic
  structure, small corpus) simultaneously, though neither was built for 6-fold leg symmetry.
  Worth a close primary-source read before deciding whether to adapt either.
- **Self-supervised correspondence-field training** (LoopReg, IPNet) — a demonstrated path to
  training a point-cloud-to-parameter regressor with *no* hand-labeled pose data at all, using
  surface self-consistency as the only training signal. Relevant if the small labeled corpus
  remains a bottleneck.
- **Multi-hypothesis output with learned selection** (GFPose, D3DP/JPMA, ManiPose) — matches the
  project's own Phase 6 idea (multi-hypothesis + selector) already sketched in
  `CHAIN_OF_THOUGHT_20260824_WSL.md`. The field's current best answer to "select without ground
  truth" (reprojection-error proxies) doesn't map directly onto SMILify, since the natural
  analogue proxy (Chamfer) is already known to be unreliable for this — this is flagged as an
  open problem in the field too, not a solved import, and would need a genuinely new selection
  signal to work here.
- **Synthetic-data diversity augmentation** (AniMer's diffusion-generated training set) — the
  general strategy (multiply a small real/synthetic corpus with generated diversity) is
  transferable in principle via SMILify's own existing synthetic-corpus generator
  (`diagnostics/moonshot/make_synth_corpus.py`), even though AniMer's specific mechanism
  (natural-image diffusion models) has no ant equivalent.

**Not recommended given current constraints:**

- Anything depending on a pretrained 2D vision backbone or natural-image diffusion prior
  (MagicPony, Farm3D, most recent high-generalization human regressors) — no ant-domain
  equivalent exists, and point-cloud-only input rules out the RGB/texture cues these lean on.
- Multi-view video-based insect tracking methods (DeepFly3D, Anipose, DeepLabCut-3D, SLEAP) —
  confirmed to be a rich, mature field, but built entirely around a different input modality
  (multi-view video of live/tethered specimens over time) than SMILify's single static scan.

---

## 5. Full catalogue — every finding, with what it specifically means for SMILify

~60 methods, organized by category. Every row states the mechanism and then, separately, what
that finding implies for SMILify's actual situation (novel category, small synthetic corpus,
point-cloud-only input, no real-specimen ground truth, six near-identical leg chains) — not just
a generic summary of the paper.

### 5.1 End-to-end regression — human, image-based

| Method (year) | Mechanism | What it means for SMILify |
|---|---|---|
| **HMR** (2018) | One feedforward CNN regresses SMPL pose+shape+camera in a single pass; adversarial prior keeps output plausible. | The "one-shot everything" template `smil_pointnet.py` follows — but it leans on a large pretrained image backbone and huge human datasets. SMILify has neither. This is the baseline case for *why* full end-to-end hasn't worked yet, not a method to copy. |
| **GraphCMR** (2019) | Graph-CNN regresses mesh vertices directly, then fits SMPL to that mesh afterward. | Shows an alternative target (regress geometry, then derive pose) instead of regressing joint angles directly. Not directly portable (still image/backbone-dependent) but worth noting as a reframe if the leg-pose network's direct-angle-regression target turns out to be the wrong one. |
| **SPIN** (2019) | Network gives an initial SMPL estimate → an optimizer refines it *inside the training loop* → the refined fit re-supervises the network. Self-improving loop, no 3D ground truth needed. | This is close to a template for what SMILify could do next: instead of training the leg-pose network once against synthetic ground truth and stopping, feed its predictions through the actual D1 optimizer during training and use the *optimizer's own output* as a better training target. Directly addresses the "predicting close-to-GT pose isn't the same as predicting a good optimizer seed" problem this investigation already identified. |
| **VIBE** (2020) | Video-based regression with a motion discriminator. | Not applicable — SMILify fits single static scans, not video sequences. |
| **PyMAF / PyMAF-X** (2021/2022) | Iterative "pyramidal alignment feedback" — regress, re-project, correct, repeat a few times, still within one feedforward network (no external optimizer). | A middle ground between one-shot regression and full optimization: a few internal self-correction steps. Relevant as a cheaper alternative to a full D1 optimizer pass if the leg-pose network is ever asked to self-refine without leaving the network. |
| **PARE** (2021) | An auxiliary 2D part-segmentation branch gives soft per-body-part attention, improving robustness to occlusion. | The "per-part attention" idea is conceptually attractive for a 6-leg body — an explicit per-leg attention or segmentation head is a plausible way to help the network keep legs distinct — but PARE's actual signal is RGB occlusion cues, which don't exist in a point-cloud-only pipeline. The *idea* transfers, the *mechanism* doesn't. |
| **ProHMR** (2021) | A conditional normalizing flow models a full *distribution* over plausible poses, not one point estimate — explicitly embraces ambiguity, with exact-likelihood evaluation. | Directly relevant in spirit: if leg correspondence really is multi-modal (which the basin-map experiment now running is testing), a distributional output is the literature's answer, not a single best guess. The exact-likelihood property would also let SMILify rank leg-pose candidates by learned plausibility — a genuinely new selection signal, distinct from Chamfer (which is already known not to discriminate the correct basin). Worth citing even though not directly portable without a flow-based rebuild. |
| **PIXIE** (2021) | Fuses separate body/face/hand "expert" networks via a confidence-weighted moderator. | Whole-body multi-part fusion, not symmetry-specific. Low relevance — SMILify's problem isn't fusing different body regions, it's disambiguating six copies of the same region. |
| **ExPose** (2020) | One-shot regression of body+face+hands to SMPL-X from a single RGB image. | Same backbone/data dependency as HMR. Not a new lever. |
| **ICON** (2022) | Uses an existing SMPL estimate to guide implicit-surface detail inference. | A downstream detail-refinement stage, not a pose initializer — not relevant to the initialization question specifically. |
| **CLIFF** (2022) | Fixes a known crop-relative-vs-full-image bias in the HMR family by feeding crop location into the regressor. | Fixes an artifact specific to image-cropping pipelines. Irrelevant to point-cloud input — included only as an SOTA image-baseline reference point, not an idea to adopt. |
| **HybrIK / HybrIK-X** (2021/2025) | Twist-and-swing decomposition: network predicts 3D joint positions + a 1-DOF "twist" angle; the rest of each joint's rotation ("swing") is solved **analytically** via forward-kinematics geometry from those positions, not regressed. | The single most actionable idea in this whole sweep for the current leg-pose network. Right now the network (as far as this investigation's docs describe it) likely predicts joint rotations directly and independently. HybrIK's pattern — predict *positions* (a more robust target, less prone to the symmetric-confusion failure mode) and *derive* rotations analytically per the known chain — respects the kinematic structure by construction instead of hoping FK-consistency emerges from independent per-joint predictions. Concretely testable as a redesign of `train_leg_pose_regressor.py`'s output head. |
| **4DHumans / HMR2.0** (2023) | Transformer-based one-shot SMPL regression trained on very large-scale, diverse human data; strong generalization even to unusual/occluded poses. | Evidence about *what it costs* to get a one-shot regressor to generalize well: a big transformer plus massive diverse training data. SMILify has a 12–5000-specimen synthetic corpus and no transformer-scale data budget. This is direct evidence *against* expecting a from-scratch full-parameter regressor (à la `smil_pointnet.py`) to generalize well on the current corpus size — not an argument to abandon the idea, but a reason to expect it needs either much more synthetic data or a smaller, more constrained target (which is exactly why the current work narrowed to leg-pose-only). |
| **TokenHMR** (2024) | Discretizes/tokenizes the pose representation instead of regressing continuous values, improving pose plausibility. | Discretizing the leg-pose output space to only anatomically valid states is a plausible way to constrain a small-data network and reduce the "wrong but structurally-plausible" failure mode by construction, rather than by adding a separate prior/regularizer after the fact. Worth considering for a future architecture revision. |
| **WHAM** (2024) | World-grounded video regression with a motion prior. | Video/world-frame-specific. Not applicable. |
| **NLF** (2024) | Predicts arbitrary body-point locations via a continuous localization field, generalizing across different body models/datasets. | The model-agnostic design principle (useful for a project juggling multiple rigged models, e.g. ant vs. mouse templates) is appealing, but it needs large-scale pretraining data SMILify doesn't have. Note the idea, don't expect to reuse the weights or approach as-is. |
| **GenHMR / ADHMR / ProPose** (2023–2025, generative/diffusion HMR variants) | Diffusion or flow-based generative models over pose space, explicitly targeting ambiguity; ADHMR further aligns diffusion output via preference optimization. | Reinforces that current human-pose research treats "ambiguous pose from ambiguous input" as a distribution-modeling problem, not a single-regression problem. None targets multi-limb correspondence specifically, but this is useful framing to cite when discussing why a single-point-estimate leg-pose network might be structurally the wrong kind of output for this problem. |
| **PHD** (2025) | Two-stage: calibrate personal shape once, then a diffusion transformer iteratively refines *pose* (as 3D keypoints, not a scan point cloud) as a prior guiding fitting. | Input modality (video/2D keypoints) doesn't match SMILify's (point-cloud scan) despite the superficially similar name. Relevant only for the general pattern — a learned diffusion prior used as a regularizer during fitting — which is conceptually adjacent to how the network's prediction is already used to seed the D1 optimizer. |

### 5.2 End-to-end regression — human, point-cloud-based (closest input-modality match)

| Method (year) | Mechanism | What it means for SMILify |
|---|---|---|
| **3D-CODED** (2018) | A feedforward encoder gives an initial global shape descriptor from a point cloud/mesh, then **gradient descent through the decoder** (test-time optimization of the latent code) refines the reconstruction; correspondences fall out implicitly from a fixed template's deformation. | Direct precedent for SMILify's actual philosophy: it is *not* pure one-shot regression — it explicitly combines a fast feedforward init with per-instance test-time refinement, the same two-stage pattern the current leg-pose-network-plus-D1-optimizer design already uses, just applied to a learned latent shape code instead of leg-pose parameters. Validates the existing architecture choice at a conceptual level. |
| **Skeleton-Aware 3D Human Shape Reconstruction from Point Clouds** (Jiang et al., 2019) | PointNet++ → graph-aggregation → an **attention module** that maps unordered point features onto an *ordered* set of skeleton-joint features → SMPL parameter regression. Fine-tuned per test point cloud with an unsupervised loss. | The single closest architectural peer to `smil_pointnet.py` found anywhere in this sweep — same input (raw point cloud), same PointNet-family backbone, same one-shot full-parameter output. Its key lesson: raw PointNet features were *not* sufficient on their own — an explicit attention step was needed to resolve the unordered-point-to-ordered-joint correspondence problem, even for a human skeleton with only 2-fold symmetry at worst. This is strong evidence that `smil_pointnet.py`'s current architecture (as described) is missing a comparable correspondence-resolving mechanism, and that the gap would be *worse*, not better, for six near-identical legs. Read this one closely before any redesign decision. |
| **LoopReg** (2020) | Self-supervised: one network predicts a per-point implicit correspondence to a canonical model, a second regresses pose/shape, and the two supervise each other in a loop — no ground-truth pose or correspondence labels needed, trained on raw unregistered scans. | Directly relevant to SMILify's small-corpus constraint: demonstrates training a point-cloud-to-parameter regressor using only surface self-consistency as signal, no hand-labeled pose. A plausible path if the project wants to reduce dependence on synthetic ground-truth generation, or to eventually incorporate real (unlabeled) bench50-style scans into training. |
| **IPNet** (2020) | Predicts implicit occupancy reconstruction plus sparse correspondences to SMPL from partial/sparse input, then fits SMPL to the implicit surface. | Second data point (with LoopReg) that "predict correspondences, then fit" outperforms "predict parameters directly" for point-cloud human recovery in the literature. Directly informs the architecture debate: an explicit correspondence-prediction stage before pose regression is a recurring, validated pattern — not a one-off idea — and is a serious alternative to what `smil_pointnet.py` currently attempts (parameters, no correspondence stage). |
| **NASA** (2020) | A neural occupancy function conditioned on a *given* pose parameter — assumes pose is already known. | Not a pose regressor at all (it's a shape-representation technique). Included for completeness; not applicable to the initialization question. |
| **Generalizing Neural Human Fitting to Unseen Poses with Articulated SE(3) Equivariance** (2023, = "ArtEq") | Builds SE(3)-equivariance into the point-cloud network so predictions transform consistently under arbitrary global rotation, specifically to improve generalization to unseen/out-of-distribution poses. Local SO(3)-invariant part detection + equivariant pose head, no optimization-refinement step needed, ~44% error reduction over prior point-cloud-to-SMPL methods on their benchmark. | Directly actionable: baking rotation-equivariance into the network is a cheap way to remove an entire axis of variation (global orientation) that a small-data network would otherwise have to learn to be robust to by seeing many examples. Very plausibly transfers to an ant point-cloud network. Also relevant to the deferred global-orientation (G2/G4) hypothesis — equivariance could sidestep needing to solve that separately at all. One of the two systems (with TANet) that come closest to covering SMILify's actual constraint set; the part-based local-frame construction is worth reading closely, though it's untested on 6-fold-symmetric structures. |
| **FPCR-Net** (2025) | End-to-end front-point-cloud → SMPL regression, using SO(3)-equivariant separable convolutions on the front scan plus a network-predicted *back* point cloud, explicitly to handle rotational ambiguity from partial/single-view scans. Reports ~43–45% error reduction over direct-regression baselines. | The most recent point-cloud-to-parametric-model regressor found, and its problem setup (partial/single-view point cloud → full parametric model, one-shot) is nearly identical to SMILify's. Good current-SOTA reference point for what's achievable — but tested only on humans, with far less severe symmetry than a 6-legged body, so the reported error reduction shouldn't be assumed to transfer directly. |
| **Self-Supervised Human Mesh Recovery from Partial Point Cloud via a Self-Improving Loop** (2025) | SPIN-style self-improving loop adapted specifically to partial point-cloud input; no ground-truth pose needed. | Confirms "network init + optimizer/fitting-loop refine, mutually bootstrapping during training" is now a standard pattern for point-cloud body regression specifically (not just images). Reinforces the SPIN-style recommendation in §5.1: closing the loop between the leg-pose network and the D1 optimizer *during training*, rather than training the network once in isolation, is the natural next architectural step suggested by the field, more so than building a wholly separate full-body regressor. |

### 5.3 End-to-end regression — animal / quadruped

| Method (year) | Mechanism | What it means for SMILify |
|---|---|---|
| **SMALify / SMAL** (2017) | Pure per-instance optimization (gradient descent against 2D keypoints/silhouette), no learned regressor at all. | This is the ancestor of SMILify's own D1 optimizer, not an alternative to it — establishes the baseline the whole investigation is trying to improve on with initialization. |
| **SMALR** (2018) | Multi-image optimization refining shape in a canonical pose across several views, still no learned regressor. | Optimization-only, no new lever for initialization. |
| **SMALST / "Three-D Safari"** (2019) | Feedforward network regresses pose+shape+texture directly from pixels for zebras, trained via a differentiable renderer with self-supervised photometric + silhouette losses (no 3D ground truth, no keypoints needed at test time). | Demonstrates per-species feedforward regression is achievable with a *moderate*, not massive, corpus, by substituting self-supervised rendering losses for 3D ground truth. Relevant strategy in principle for a small ant dataset — but it fundamentally needs RGB texture/photometric cues that SMILify's point-cloud-only pipeline does not have. The self-supervision *idea* is worth keeping, the *mechanism* isn't portable as-is. |
| **WLDO ("Who Left the Dogs Out?")** (2020) | Feedforward network gives an initial pose/shape estimate from an image; an **EM loop over the whole training set** jointly refines a learned shape prior and per-image fits. | The EM-in-the-loop strategy — using population-level joint refinement to bootstrap a richer prior from a modest, noisy, in-the-wild dataset — is a strong candidate template if SMILify ever wants to build a fuller shape/pose prior for ants from its small corpus, rather than treating each specimen's fit as fully independent. Distinct from, and complementary to, the SPIN-style single-instance refinement idea. |
| **BARC** (2022/2023) | Feedforward image→SMAL-derived pose/shape, with a novel breed-similarity loss (same-breed dogs pulled together in shape space) to compensate for missing paired 3D ground truth. | The core trick — exploit a coarse category label as weak supervision when 3D ground truth is scarce — could map onto SMILify *if* ant specimens carry any coarse sub-category/caste/species label usable the same way; otherwise not directly portable. Worth checking whether the bench50/synth corpora have any such metadata already. |
| **hSMAL** (2021) | Horse-specific SMAL adaptation + optimization-based fitting for motion analysis. | Minor relevance — mainly evidence that the SMAL family already extends to new species via model modification (which SMILify has itself done for ants/mice), not a regression-methodology contribution. |
| **AniMer** (2025) | Transformer-based one-shot regressor across *multiple quadruped families simultaneously*, using a "family-aware contrastive learning" scheme to separate anatomically distinct quadruped families within one backbone; trained partly on a diffusion-generated synthetic dataset (~10k images) to expand data diversity. | The most recent (2025) full end-to-end animal-mesh regressor found. Its family-aware mechanism disambiguates *between different animal body plans* — a different problem from disambiguating *repeated identical limbs within one body plan* — so it does not solve SMILify's core issue. Its synthetic-data-augmentation trick (generate more training diversity rather than collect more real data) is the transferable idea: SMILify's own synthetic-corpus generator (`make_synth_corpus.py`) already does something structurally similar and could plausibly be pushed further in this direction. |
| **CoP3D** (2022/2023, dataset) | A large crowd-sourced cat/dog video dataset (4,200 pets, 322GB) plus a NeRF-based tracker for dynamic view synthesis. | Not a parametric regressor — included mainly to illustrate the *scale* of data typically used for animal 3D work, which underscores how much smaller SMILify's ant corpus is by comparison. A scale-of-the-problem data point, not a method to adopt. |
| **MagicPony** (2023) | Single-image feedforward network predicts shape/articulation/viewpoint/texture/lighting with **zero 3D supervision**, using features distilled from a pretrained self-supervised vision transformer (DINO). | Strong precedent that zero-3D-supervision training is achievable — but the entire mechanism depends on DINO's pretrained semantic correspondence knowledge, learned from massive natural-image datasets. No equivalent pretrained backbone exists for ants, and point-cloud-only input has no RGB signal to feed such a backbone anyway. This dependency is the single biggest reason MagicPony's specific approach is not portable to SMILify, even though the *goal* (train without 3D ground truth) is shared. |
| **Farm3D** (2023) | Same category-specific reconstruction goal as MagicPony, but bootstrapped entirely from a pretrained 2D image diffusion model instead of real image collections. | Same core limitation as MagicPony — depends on a large pretrained 2D generative prior with no ant-domain equivalent. Currently inapplicable, for the same reason. |
| **4D-Animal** (2025/2026) | Reconstructs animatable 3D animals from casual video. | Video-based; not examined in depth. Flagged as existing, likely shares the foundation-model dependency of the other recent animal-reconstruction work above — not a near-term lever. |
| **Animal 3D reconstruction survey** (2025, arXiv 2508.16062) | Recent survey of deep-learning animal 3D reconstruction methods by input modality/representation/training regime. | Confirmed RGB image/video-based scope; **could not confirm** whether it covers point-cloud-native methods or multi-limb symmetry ambiguity specifically — flagged in §7, worth a full read as a bibliography source but not yet as evidence for or against any specific claim here. |

### 5.4 Learned point cloud correspondence / registration

| Method (year) | Mechanism | What it means for SMILify |
|---|---|---|
| **GeoTransformer** (2022) | Encodes pairwise distances + triplet angles for transformation-invariant superpoint features; attention-based matching without RANSAC. | Rigid registration only, built for scene-scale scans. Would need per-part rigid decomposition (one instance of the network per leg segment, essentially) to handle an articulated body at all — a substantial adaptation, not a drop-in fit. |
| **D3Feat** (2020) | Joint dense keypoint detection + description from a single KPConv backbone. | Same rigid-only limitation as GeoTransformer. Useful only as a feature-extraction backbone if SMILify wanted dense per-point descriptors as an input representation — not a template-aware or articulation-aware method itself. |
| **REGTR** (2022) | End-to-end transformer directly regresses each point's corresponding location in the other cloud via attention, no explicit matching/RANSAC step; also predicts overlap probability. | The overlap-probability output is conceptually useful for partial scans (a real ant scan may be incomplete), but it's rigid-registration only — no notion of a kinematic template, so still needs per-part adaptation to be usable. |
| **Lepard** (2022) | Disentangles feature space from 3D position; handles both rigid *and* deformable (smooth non-rigid) scenes via deformable-ICP; introduced 4DMatch/4DLoMatch non-rigid correspondence benchmarks. | The closest of the classic-registration family to SMILify's actual problem, since it handles deformation at all — but "deformable" here means smooth surface warps (cloth-like), not a discrete kinematic tree with hard joint-angle constraints. Matching quality on genuinely articulated, symmetric multi-limb structures is untested; worth a closer look but likely needs real adaptation, not direct application. |
| **TANet** (CVPR 2024) | Learns explicit-structure templates plus a "template-assistance" module establishing correspondence from multiple perspectives, unsupervised, **evaluated on animal datasets** alongside human data. | Highest structural relevance in this whole thread: it's explicitly template-to-target correspondence — a known parametric template, an unlabeled scan — which is SMILify's actual setup, not an adaptation of one. Not evaluated on symmetric multi-limb topology or trained from a corpus as small as SMILify's, which is exactly the open question. Recommended for a deep read before any correspondence-based redesign decision — this is the single strongest correspondence-thread candidate found. |
| **Deep Functional Maps family** (Spatially/Spectrally Consistent DFM 2023; Non-Rigid Registration via DFM Prior 2023; CoE 2024; RINO 2024) | Spectral-basis correspondence via Laplace-Beltrami eigenfunctions, made "deep" by learning descriptor functions or synchronizing maps across a shape collection. | A mature line of dense non-rigid correspondence work, strong for near-isometric deformation. Important caveat: large articulated pose changes and *topologically repeating* limbs (six legs = near-symmetric spectral signatures) are a known failure mode for spectral methods — i.e., this family would plausibly hit the *same* symmetric-leg confusion problem the current learned regressor already hit, for a structurally similar reason (both rely on signals that don't distinguish repeated near-identical parts). Informative as a caution, not a promising direction on its own. |
| **3D-CODED / CPAE** (2018/2021) | 3D-CODED deforms a fixed template to match a target shape, yielding correspondence implicitly via the deformation field; CPAE maps arbitrary point clouds to a canonical primitive so same-location points across instances are inferred as corresponding. | The closest classical precedent to "known template, unlabeled target, learn the deformation/correspondence" — essentially SMILify's problem restated for shape instead of pose. Built for near-rigid or single-object-category shapes without a hard kinematic-chain constraint; would need the deformation field explicitly constrained to the SMIL joint hierarchy to be a fair comparison for SMILify's use case. |
| **ArtEq** (2023) | See §5.2 — listed here too since it straddles correspondence and regression. | Same entry as above; cross-referenced because it's simultaneously one of the closest correspondence-aware *and* regression-based candidates found. |

### 5.5 Multi-hypothesis / probabilistic / ambiguity-aware pose estimation

| Method (year) | Mechanism | What it means for SMILify |
|---|---|---|
| **VPoser** (2019, part of SMPLify-X) | A VAE trained on motion-capture poses regularizes a latent code toward a Normal distribution; used as a differentiable pose-plausibility prior/regularizer *during* optimization, not as a hypothesis generator. | Not multi-hypothesis, but directly relevant as an alternative lever: a learned low-dimensional "is this leg configuration plausible" prior, trained on a corpus of valid leg configurations, could be added to the D1 optimizer's loss as a regularizer, independent of the initialization question entirely. Needs a pose corpus large enough to train the prior — the same small-corpus constraint as everything else here, but a much lower-dimensional target than a full initializer. |
| **Pose-NDF** (2022) | Models the pose manifold as the zero-level-set of a neural distance field over joint-rotation space; arbitrary poses can be projected onto the manifold via gradient descent, preserving pose-space distances better than a VAE's Gaussian collapse. | Same category as VPoser but geometry-preserving. Could serve the same "plausibility regularizer inside the optimizer loss" role, potentially more faithfully. Worth comparing against VPoser's simpler VAE approach if this direction is pursued. |
| **GFPose** (2023) | Learns the pose prior as a gradient (score) field; Langevin-dynamics-style sampling naturally yields multiple hypotheses from the same input. | The "multiple plausible completions from ambiguous input" framing matches SMILify's occluded/self-similar-leg problem well — but the selection-among-samples problem is left unsolved here too, same as everywhere else in this thread. Confirms the selection problem is a field-wide gap, not something SMILify uniquely failed to close. |
| **DiffPose / D3DP (+ JPMA)** (2023, ICCV family) | A diffusion model denoises from a random pose distribution conditioned on 2D keypoints, producing many hypotheses; **JPMA** explicitly tackles selection — reprojects each hypothesis to 2D and picks the best hypothesis *per joint* by reprojection error (a ground-truth-free proxy), then recombines joint-wise. | The single highest-value item in this thread for the selection problem specifically. The pattern — score/select *per-joint* using a cheap, GT-free, differentiable proxy, rather than treating whole-pose selection as one decision — is directly portable in spirit: SMILify could analogously score per-*limb* hypotheses via a chamfer/surface-consistency proxy rather than one global score. Real caveat: their proxy (2D reprojection against a known camera) has no exact SMILify analogue, since surface Chamfer is exactly the metric already shown *not* to discriminate the correct leg-assignment basin. Whether *any* cheap proxy would discriminate correctly here is an open question this idea raises but doesn't answer. |
| **ScoreHMR** (2024) | A diffusion model over body-model parameters, with inference-time denoising *guided* by a task-specific score/gradient toward the observation — effectively replaces optimization with guided sampling, without retraining per task. | An alternative to "regress an init, then run D1" entirely: train a diffusion prior over SMIL leg poses once, then use guided denoising toward the point-cloud observation instead of (or alongside) gradient-descent fitting. Same small-corpus constraint as VPoser/Pose-NDF for training the prior. A genuinely different architecture family worth flagging as an option, not just an incremental tweak. |
| **ProHMR** (2021) | See §5.1 — listed here too as a multi-hypothesis method. | Cross-referenced; its exact-likelihood property (candidates rankable by learned likelihood, not just sampled) is the specific reason it appears in both threads. |
| **HuMoR** (2021) | A conditional VAE over pose *transitions* (not static poses) — captures that some states (e.g. mid-motion) are nearly deterministic while others (e.g. idle) are highly ambiguous; used as a motion prior for robust optimization from occluded/noisy observations. | SMILify fits static poses, not motion sequences, so the temporal-transition framing doesn't map directly. The transferable idea is the *reframing*: rather than one global "how good is the initializer" number, characterize ambiguity per-joint or per-limb — which is exactly what the basin-map experiment currently running is doing empirically, independent of HuMoR's specific mechanism. Useful as corroborating framing for that experiment's design, not as a method to adopt. |
| **ManiPose** (2024) | Outputs multiple candidate poses *each with its own estimated plausibility score*, via a discriminative (non-generative, non-sampling) architecture. | Directly relevant to the selection problem: a network that outputs a hypothesis *and* a self-assessed confidence, trained discriminatively rather than generatively, may suit a small synthetic corpus better than a full generative model (VAE/diffusion/flow) would. Could let SMILify's optimizer or a downstream check decide whether to trust the learned init or fall back to zero-init on a per-specimen basis — a concrete, testable idea. |

### 5.6 Kinematic-tree-aware / hierarchical / iterative-refinement architectures

| Method (year) | Mechanism | What it means for SMILify |
|---|---|---|
| **HybrIK / HybrIK-X** | See §5.1 — the central architectural recommendation of this whole sweep, cross-referenced here as the anchor of this category. | Restated: predict positions, derive rotations analytically per the known chain. This is the most concrete, actionable, literature-backed next step for `train_leg_pose_regressor.py`'s architecture found anywhere in the sweep. |
| **RLE (Residual Log-Likelihood Estimation)** (2021) | Learns the *residual* between a flow-reparameterized output and a simple base distribution, giving a regression-based network calibrated per-joint uncertainty essentially for free (no extra test-time cost). | A cheap way to get per-joint confidence out of the *existing* leg-pose network architecture without switching to a full multi-hypothesis rebuild — bridges directly to the §5.5 selection-problem discussion. Low-cost to try relative to a full architecture change. |
| **Iterative Error Feedback (IEF)** (2016) / cascaded pose regression (DeepPose, 2014) | Feed the current pose estimate back in alongside the input, predict a correction, apply, repeat — self-correcting refinement instead of one-shot regression. | Older, foundational rather than SOTA — the ancestor of PyMAF/HybrIK-style refinement. Relevant as one candidate explanation for why a one-shot PointNet regression underperforms: it never checks its own prediction against the observed surface before committing to it. Worth naming as historical context for why the field moved toward refinement loops, not as something to implement directly. |
| **Skeletal Graph Neural Networks for Hard 3D Pose Estimation** (Zeng et al., 2021) | A graph-convolution regressor that explicitly encodes the kinematic-tree topology as graph edges, rather than predicting joints independently or fully-connected, targeting specifically the "hard" (ambiguous/occluded) joints. **Unverified — see §7.** | If confirmed, directly relevant: a graph neural network over the SMIL/SMAL kinematic tree (coxa→trochanter→femur→tibia→tarsus edges, six such chains) is a natural architecture to explicitly encode "child depends on parent," instead of the current network's likely flat, independent per-joint prediction. A plausible next architecture to prototype alongside HybrIK-style decomposition. Read the primary source before committing engineering time to this, since the mechanism above is reconstructed from search snippets only. |
| **Coarse-to-fine / FK-error-compounding literature** (general, cross-checked across multiple independent secondary sources) | Recurring pattern: resolve root/torso pose first (well-constrained), then progressively estimate distal joints conditioned on the resolved proximal ones — because "an error in a parent joint propagates through all its children during forward kinematics," compounding down the chain. | This is the literature's explanation for *why* proximal errors should hurt more than distal errors — a structural, geometry-driven fact about forward kinematics itself, not an architecture-specific empirical quirk. **This directly and independently corroborates the project's own D/E/F perturbation finding** (proximal errors more damaging than distal). Worth citing in the basin-map writeup as confirmation from the wider field — though cite the general FK-error-compounding principle, not any single paper, since it's closer to a geometric truism recurring across sources than one citable empirical result. |
| **GAIP** (2024) | Multi-stage graph-convolution pose regressors for sparse-IMU-to-full-pose regression, motivated by "existing methods often ignore the spatial prior of skeletal topology." **Unverified — see §7.** | Analogous problem shape to SMILify's: sparse/partial observation (IMUs ≈ partial surface points) → full articulated pose, with a topology-aware GNN proposed as the fix over topology-agnostic regression. Reasonable precedent for the same architectural direction as the Zeng et al. entry above, but treat as a lead pending primary-source verification. |
| **Learnable SMPLify** (2025) | Learns a differentiable/neural replacement for the SMPLify optimization loop itself — an entirely learned inverse-kinematics solve instead of gradient descent against a loss. | Useful for framing the design-space boundary in the discussion with Fabian: SMILify's current approach (learned init + optimizer) sits *between* one-shot regression (ArtEq, Learnable SMPLify itself) and pure optimization (SMILify's original zero-init baseline). Very recent (2025) and not yet battle-tested — name it as the far end of the spectrum, not as something ready to adopt. |

### 5.7 Category-agnostic / novel-category articulated shape+pose (no large pretrained prior)

| Method (year) | Mechanism | What it means for SMILify |
|---|---|---|
| **LASSIE** (2022) | Learns articulated 3D shape from a sparse ensemble (~20–30) of 2D images of one animal class, discovering parts/skeleton via a category-level 3D "part" prior optimized per-instance — no 3D training data. Still needs a hand-specified rough skeleton and 2D images. | Closest "small-corpus, no-pretrained-prior" precedent found — but the input modality (2D image ensemble) and reliance on 2D silhouette/feature losses don't transfer directly to single 3D scan fitting. Relevant as evidence that small-corpus, no-pretrained-backbone learning is achievable at all, not as a method to reuse. |
| **Hi-LASSIE** (2023) | Extends LASSIE to auto-discover the class skeleton from a reference image (no manual skeleton needed), then optimizes per-image articulation from 20–30 images. | Its "discover a generic skeleton, then re-pose per-instance via optimization" pattern is architecturally similar to what SMILify already does (hand-rigged template + optimizer) — validates the general two-stage strategy again, from yet another independent line of work. Still 2D-image-driven, not scan-driven, so not directly portable. |
| **MagicPony** (2023) | See §5.3. | Cross-referenced; the DINO-pretrained-backbone dependency is the recurring reason this family doesn't transfer to ants. |
| **Farm3D** (2023) | See §5.3. | Cross-referenced; same diffusion-prior dependency issue. |
| **BANMo** (2022) | Builds an animatable neural 3D model from many casual videos of *one individual* — per-subject optimization, not a cross-instance category prior. | Least relevant of this group — closer to per-specimen video-based NeRF fitting than to template-based pose initialization from a single static scan. |
| **3D-Fauna** (2024) | Learns one shared model across many animal species (a semantic "part bank") so a new species can be handled with few images by borrowing structure from related species already in the bank. | Directly tests the "novel category, few examples" regime SMILify is in — but leans on cross-species transfer from a large existing mammal/bird corpus, which has no arthropod analogue to borrow from. The part-bank idea itself (share structure across nearly-identical repeated parts) is conceptually relevant to ants' six similar legs, even though the specific fauna bank wouldn't apply. |
| **ANCSH** (2020) | Category-level articulated *object* pose: PointNet++ predicts a per-part Normalized Coordinate Space + joint parameters from a single depth point cloud. Needs substantial per-category labeled training data (many instances, part/joint annotations). | Solves exactly SMILify's structural problem (known kinematic structure, unknown instance pose, single point cloud input) — but needs a real per-category training set with part/joint annotations, which SMILify's small synthetic corpus would need heavy adaptation to provide (though note: SMILify's synthetic generator *does* produce exact ground-truth joint annotations for free, unlike real-world object datasets — this may be more feasible here than the "needs substantial data" caveat suggests for other domains). Strong architectural reference for "predict per-part canonical pose directly from a single point cloud." |
| **CAPTRA** (2021) | Extends ANCSH-style ideas to temporal tracking with an end-to-end differentiable pipeline. | Same training-data trade-off as ANCSH; the tracking/temporal framing is less relevant since SMILify fits static single scans, not sequences. |
| **SCAPO** (2026, very recent) | Self-supervised (no ground-truth pose/segmentation labels) category-level articulated pose from a single RGB-D observation, using an SE(3)-equivariant autoencoder for a canonical space + joint-aware blend-skinning, trained via cycle-reconstruction across a set of instances. | The most directly relevant Thread find overall: removes label supervision (matches SMILify's constraint) while keeping single-observation inference. Caveat: still needs a *set* of same-category instances for self-supervised training — a single hand-rigged ant template with no comparison population wouldn't straightforwardly provide that, though SMILify's synthetic corpus of many specimens might. Flagged in §7 as very recent (weeks old) — worth a close read, not yet a citation to build on. |
| **Pose Anything** (2024) | 1-shot 2D keypoint localization for arbitrary categories via a Graph Transformer Decoder that explicitly encodes the keypoint-graph structure, evaluated on a 100-category benchmark; needs one annotated support image. | 2D-image/keypoint task, not 3D shape+pose, and still needs one annotated example — but the mechanism (explicitly encode the known skeleton-graph topology to disambiguate structurally similar keypoints) is the single most transferable *idea* found for ants' six near-identical legs, independent of the modality mismatch. Cross-referenced with the "explicit skeleton-graph encoding" recommendation in §4/§5.6. |

### 5.8 Insect / arthropod-specific 3D pose and leg-tracking

| Method (year) | Modality | What it means for SMILify |
|---|---|---|
| **DeepFly3D** (2019) | 7-camera multi-view video of tethered Drosophila; 2D CNN detection + pictorial-structures error correction + active learning; 38 landmarks. | Multi-view video, not a single static scan — a genuinely different problem, not a source of a directly-portable method. The one usable piece of methodology: its pictorial-structures error-correction step uses known skeletal-chain geometric constraints to auto-detect and fix implausible 2D detections before triangulation — a "geometric plausibility check" idea that could inform sanity-checking an initial pose guess against known joint-limit constraints. |
| **Anipose** (2021) | Multi-view video built on DeepLabCut 2D output: calibration + spatiotemporal-regularized filtering + triangulation. | Same modality gap as DeepFly3D. Its regularization ideas assume a temporal sequence SMILify doesn't have for a single static specimen scan — not portable. |
| **DeepLabCut(-3D)** (2018) | 2D markerless pose via transfer learning (~200 labeled frames sufficient); 3D extension triangulates across calibrated multi-view 2D detections. | Foundational tool of this subfield but strictly image/video-based, 2D-then-triangulate. No single-3D-scan analogue exists in this line of work at all. |
| **LEAP → SLEAP** (2019/2022) | 2D (and multi-view-derived 3D) multi-animal pose tracking, fast inference. | Same video/2D-input characterization. Not applicable to static-scan geometry directly. |
| **Kim et al., "Tracking the joints of arthropod legs using multiple images and inverse kinematics"** (2015) | Multi-view images of small arthropods; uses an **IK solver** to recover intermediate joint angles of a leg chain from known end-effector (tip) positions, given known chain topology/segment lengths. | The single most directly transferable finding in this whole thread: if a scan's leg-tip and body-attachment positions can be identified geometrically, an anatomically-constrained IK solve per leg chain — using known segment lengths and joint limits — produces a pose guess consistent with real anatomy, independent of any learned correspondence step and immune to the leg-vs-leg confusion problem (since each leg is solved from its own known tip/root, not matched via learned features). Genuinely learning-free and directly prototypable now, as noted in §4. |
| **Lanternfly self-righting study** (2023/2024) | Multi-view high-speed video combined with a rigged articulated 3D "anchor" model (6 two-segment legs) fitted to tracked keypoints, to reconstruct leg kinematics during self-righting. | The closest methodological analogue to SMILify's actual pipeline found in this thread — a rigged articulated template posed to match observed geometry — but still driven by tracked multi-view video keypoints over time, not a single static 3D scan. Worth reading for how they handled the per-leg IK/optimization against the rig, even though the input modality differs. |
| **Insect leg joint-range biomechanics literature** (e.g. trochanter–femur rotation-range studies) | Anatomical/biomechanical measurement of joint DOF and angular ranges per leg segment across species — not a pose-estimation method. | Not an initialization strategy by itself, but a direct source of concrete joint-limit priors that could regularize any optimization- or IK-based initializer (see Kim et al. above), reducing the anatomically-implausible-but-surface-plausible failure mode this whole investigation exists to fix. Low effort to incorporate, worth doing regardless of which broader direction is chosen. |

### 5.9 Classical / non-learned skeleton extraction and initialization

| Method (year) | Mechanism | Learning-free? | What it means for SMILify |
|---|---|---|---|
| **RigNet** (2020) | DGCNN network predicts joint locations (via mean-shift clustering) + skinning weights from a *rest-pose* mesh, trained on a large corpus of pre-rigged character models. | No — supervised, needs a large rigged-shape training corpus. | Confirmed via search: RigNet predicts a rest-pose skeleton + skinning weights only — it does **not** fit or estimate a pose against a target/posed scan. Since SMILify already has a hand-authored rig, RigNet's actual role (skeleton discovery) is moot here; it doesn't solve the initialization problem at all, despite superficial topical overlap (this project has separately drawn on RigNet-style ideas for point-to-limb assignment, a different sub-problem). |
| **Point2Skeleton** (2021) | Unsupervised (no skeleton labels, but trained on a shape dataset via reconstruction loss) network predicts skeletal points + connectivity from a raw point cloud, grounded in medial-axis-transform theory. | Partially — no label supervision, but still needs training on a shape collection. | Could give a coarse topological skeleton estimate of a scanned ant as a sanity check or coarse initializer, but the discovered skeleton isn't guaranteed to align with the hand-authored joint hierarchy (no semantic/joint-limit awareness) — more a structural cross-check than a direct pose initializer. |
| **L1-medial skeleton** (2013) | Fully classical iterative local L1-median optimization directly on a raw, noisy, incomplete point cloud — no training data of any kind. | Yes — genuinely learning-free. | The strongest zero-training safety net in this whole sweep: could run directly on each ant scan to extract a raw curve-skeleton (leg centerlines, etc.) as a purely geometric first pass, independent of any learned prior, then matched to the known rig topology. Cheap to prototype, listed first among the "free to try" recommendations in §4. |
| **Coherent Point Drift (CPD)** (2010) + articulated extension **GLTP** (2015) | Classical GMM-based probabilistic non-rigid point-set registration; GLTP adds Local Linear Embedding to handle articulated (not just smooth non-rigid) deformation, applied to human pose from 3D sensor data. | Yes — per-scan optimization, no training corpus needed. | Directly applicable: register the rigged template's rest-pose point set to a scan via GLTP-style articulated CPD to get an initial pose with no learned network at all. Explicit caveat: GMM correspondence has no built-in notion of "this is the front-left vs. front-right leg," so it's exactly as vulnerable to the six-leg confusion problem as any other correspondence method here — would likely need combining with a symmetry-breaking prior (e.g. an approximate global-orientation estimate, or the graph-topology-encoding idea from §5.7) to be usable on its own. |
| **PCA-based initial alignment** | Classical: align the template's principal axes/centroid to the scan's, as a coarse global rotation/translation/scale pre-step before local refinement. | Yes. | Trivial, learning-free, already generically useful as a coarse global-orientation initializer — directly relevant to the deferred G2/G4 global-orientation hypothesis specifically. Does nothing for per-leg articulation on its own; frame it as a cheap pre-step, not a solution to the core failure mode. |
| **Global registration (RANSAC-based feature matching)** (e.g. Open3D's pipeline) | Classical feature-based correspondence + RANSAC for coarse rigid alignment without manual initialization, typically the step before ICP. | Yes. | Same role as PCA alignment — a learning-free bootstrap for rigid/coarse alignment. Applying it per-rigid-part (e.g. per leg segment) to handle articulation would inherit the same part-symmetry ambiguity as CPD above. |

---

## 6. Genuine gaps worth stating to Fabian directly (not glossed over)

- No insect-specific single-scan pose *initialization* method exists in the literature — confirmed
  absent across multiple independent query framings, not a search-term artifact.
- No method combines all four of SMILify's actual constraints (known articulated template,
  unlabeled point-cloud target, explicit kinematic-chain structure, small training corpus)
  simultaneously — TANet and ArtEq are the closest, each missing a different piece.
- No method anywhere addresses >2-fold near-identical articulated-chain correspondence ambiguity
  — this is the crux, and it means whatever direction is chosen next will likely require genuine
  adaptation or novel design, not a straightforward literature import.
- Two items surfaced late in the sweep are recent enough (2025–2026) to warrant extra scrutiny
  before relying on them: **SCAPO** (self-supervised category-level single-observation pose,
  arXiv 2606.01940) and the animal-3D-reconstruction survey (arXiv 2508.16062, scope not fully
  confirmed). Worth a primary-source read, not a citation yet.

---

## 7. Flagged as unverified — re-check primary source before citing precisely

- Zeng et al., "Skeletal Graph Neural Networks for Hard 3D Pose Estimation" (ICCV 2021) — PDF
  fetch blocked during the sweep; mechanism reconstructed from search snippets only.
- GAIP (skeleton-aware GNN for sparse-sensor pose, ACM TOMM 2024) — same caveat, snippet-only.
- ProHMR's exact mechanism details — cross-referenced against search snippets and training
  knowledge, not independently re-confirmed against the primary paper in this sweep.
- SCAPO (arXiv 2606.01940, 2026) — confirmed to exist via arXiv abstract/PDF, but only weeks old
  at time of writing; treat conclusions about it as provisional.
- The 2025 animal-3D-reconstruction survey (arXiv 2508.16062) — abstract confirms RGB
  image/video-based scope; could not confirm from the fetched abstract alone whether it covers
  point-cloud-native methods or multi-limb symmetry ambiguity specifically.
