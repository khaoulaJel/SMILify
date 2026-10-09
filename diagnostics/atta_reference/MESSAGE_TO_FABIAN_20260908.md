# Draft message to Fabian — a pose-invariance test for SMILify morphometrics, and the WOLO connection

*Draft for review before sending — not sent yet.*

---

Hi Fabian,

Following up on the head-width replication — it turned into a broader methodological check that I
think is worth showing you, plus a real connection to WOLO.

**The question I actually tested.** Not "can SMILify reproduce every measurement" — the useful
question is: **for a given measurement, can we tell in advance whether it's trustworthy, or does
that have to be discovered by luck?** I built a four-axis validation framework to answer that
directly rather than assume it, and ran it over the GLAD trait protocol already in the repo.

**Method.** For each of the 20 fitted Atta specimens, I recomputed every candidate measurement
twice: once in the specimen's real fitted pose (as posed by the CT scan), and once with joint
rotation zeroed but everything else about that specimen (its own `betas`, per-joint
scale/translation, shape) left exactly as fitted — same specimen, same size, different
articulation. The gap between those two numbers is a direct, per-specimen measure of how much a
given measurement is contaminated by pose rather than reflecting the specimen's actual anatomy.

| trait | B2 pose Δ (mean) | B3 exponent ~body length | B1 anatomical status |
|---|--:|--:|---|
| **HW** head width | **0.115%** | **1.205** [1.154, 1.257] ✓ allometric | Type III part extent |
| HL head length | 0.321% | 1.122 ✓ | adjudicated (landmark moved 9.3% diag) |
| FL hind femur | 0.528% | 1.190 ✓ | rig joint, not surface |
| TBL total body length | 0.678% | 1.014 ✗ isometric | derived |
| ML mandible length | 1.653% | 1.130 ✓ | adjudicated (moved 12.6%) |
| WL Weber's length | 1.926% | 1.010 ✗ isometric | adjudicated (moved 7.8%) |
| PetL / GL | 2.27% / 2.33% | 1.040 ✗ / 0.887 ✓ | **no expert landmark exists** |
| **SL** scape length | **6.081%** | 1.225 ✓ | adjudicated (moved **14.7%**) |

The B3 positive control reproduces your published head-width exponent exactly (reference HW vs
reference BL → 1.2352, R²=0.9868), and the model-derived HW lands at 1.205 [1.154, 1.257] —
overlapping it, using a *different* head-width definition than the replication used. That is
convergent validation rather than a re-run.

**A correction to an earlier claim of mine, before anything else.** I previously found leg segments
essentially uncorrelated with body length (R²≈0.02) and concluded they carried no size signal.
**That was a coordinate-frame artefact, not biology.** `load_meshes` normalises every specimen
independently, so model-unit traits have absolute size stripped — real specimens span a 3.2× size
range, but model-unit body length spans only 1.55×. Regressing one compressed quantity against
another produced the near-zero R². Against the clean physical reference, hind femur scales at
**1.190 [1.088, 1.292]**, excluding isometry. Legs *do* carry size signal. What survives is legs'
higher **pose** sensitivity, which is measured independently of scale.

Three things worth your attention:

1. **Read the B3 null as 1.0, not 0.** Physical scale is restored per specimen via
   `BL_mm / BL_model`; because that factor contains body length, a trait carrying no independent
   size information lands at exponent ≈1.0. Deviation from 1.0 is the signal — R² is inflated by
   the shared factor and is not evidence on its own.
2. **`trait_extract.py` still computes traits at unadjudicated landmarks.**
   `landmark_template_indices.json` is rule-computed and explicitly `source:"verify"`. The
   expert-recalibrated file moves those exact vertices by 7.8–14.7% of the body diagonal. And no
   expert annotation exists for `petiole_*` or `gaster_apex_mid`, so PetL/GL/TBL are unvalidatable
   on that axis even in principle.
3. **Body length itself is not perfectly pose-invariant** (WL 1.93%, and `b_t`–`b_a_5` runs through
   the articulating waist). Worth knowing given it is the normaliser for everything else.

**One negative worth having.** I tested whether bilateral asymmetry — which needs *no* ground truth
and so could run on all 757 specimens — works as a per-specimen confidence score. It does not:
within a trait, per specimen, it fails to predict pose contamination (all four correlations
non-significant, p ≥ 0.10, n=20). It ranks correctly *across* traits, so it is a usable trait-level
screen and nothing more. Claiming it as a per-specimen trust score would have been exactly the
proxy-without-mechanism error this project has hit repeatedly.

**One thing I tried and abandoned, worth naming so nobody rediscovers it the hard way.** My first
instinct was to try to define a clean "pose-free" version of each specimen by zeroing not just
rotation but the per-joint scale/translation parameters too, keeping only `betas` — the rigging
equivalent of a T-pose. That doesn't work on this representation: doing so collapsed body length to
near-constant across all 20 specimens (<0.1% variation), which means this model's per-specimen size
signal lives almost entirely in the free per-joint scale/translation parameters, not in `betas`.
There's no clean "shape only, independent of size" quantity to fall back on here — trying to treat
`betas` as pure morphology and reconstruct a canonical specimen from it for morphometrics would
silently throw away most of the real biological variation. The measurement validation matrix above
is the more honest approach: work with the fitted mesh as measured, and test which anatomical
quantities on it happen to be pose-invariant, rather than trying to engineer pose-invariance into
the representation itself.

**The WOLO connection — stated conservatively, and I actually ran the check rather than just
propose it.** WOLO asks whether 2D body proportions predict mass in this species, because *Atta*
workers are strongly polymorphic. Twenty specimens isn't enough to build a competing predictive
model, and this isn't one. What I tested: does combining the two pose-invariant 3D proportions
above with body length explain mass any better than body length alone?

`log(mass) = a + b1·log(body_length) + b2·log(head_width) + b3·log(mandible_span)`, all in mm
(scaled from model units using each specimen's own body-length scale factor):

- Each predictor **alone** already explains ~99.3-99.4% of log(mass) variance (log-log slopes near
  3, consistent with roughly isometric mass~volume scaling) — unsurprising, any decent size proxy
  tracks mass this well here.
- **Combined, R²=0.996 — barely better than any single predictor**, and only `body_length` is
  individually significant (p=0.010). `head_width`'s coefficient is ~0 and not significant
  (p=0.92); `mandible_span` isn't either (p=0.32), both with large standard errors.

Honest reading: at n=20, the three measurements are too collinear with each other (all roughly
proportional to overall size) to tell whether head width or mandible span add anything to mass
prediction beyond body length alone — the data can't currently distinguish that. Not a negative
result exactly, just an underpowered one; more specimens (see the ask below) would let this
question actually be answered rather than left ambiguous. What IS established: the individual
scaling relationships themselves are real and independently checked — head width at 1.205
[1.154, 1.257], overlapping your published 1.2352, and mandible length at 1.130, both excluding
isometry. (An earlier version of this note reported mandible span at 0.616 — "negative allometry".
That figure came from the same normalised-frame regression as the leg claim above and is
withdrawn for the same reason.) It's specifically their *combined, independent* contribution to a
mass model that n=20 can't resolve.

**What I'd want from you, if it exists.** The 20-specimen set spans 1.1–47.1mg, a good range, but if
there's a larger mass-labeled *Atta* set anywhere — the WOLO training data itself, for instance —
I'd want to check whether these relationships hold from minor to major, or bend at the extremes of
the polymorphism. (Checked what's already reachable from here first: the 757-specimen `worker_ALT`
corpus has only 3 *Atta* specimens — it's the mixed 172-genus set from the genus-classification
work, not a polymorphic *Atta* series, so this isn't buildable from what I already have access to.)

**A second, more concrete ask — and I think this one is a quick answer for you.** I hit a hard
blocker trying to do allometry on the 757-specimen `worker_ALT` corpus: **I can't establish what
physical unit those meshes are in.** The numbers:

- Raw `.obj` coordinates carry a per-specimen scale, but against approximate known worker body
  lengths for 13 species the implied scale factor spans **7×**, correlating at only r≈0.43.
  *Megaponera analis* (~12mm) reads *smaller* than *Tetramorium polymorphum* (~3mm).
- Not a pose artefact — bbox diagonal is inflated by leg splay, so I redid it with mesh
  volume^(1/3), which legs barely affect. Same answer (r=0.49, 7.5× spread).
- The antscan per-specimen JSON records `voxel_size`, so I assumed the meshes are in **voxel
  units** and that this was the whole story. It isn't: **247 of 279 specimens share the same
  2.44 µm voxel**, so it varies for only ~12% and can't produce a 7× spread. Applying it lifts the
  correlation to r≈0.71 but leaves the spread at 7.2×, and *within* the single 2.44 µm group the
  implied scale is still off by 2–3× in **both** directions.
- `prepare_antscan_data_for_mesh_fitting.py` has no rescaling step — it imports the STL, cleans,
  decimates, exports — so it isn't introduced there either.

So the question is simply: **what unit are the antscan STLs in, and is that unit comparable across
scans?** If the meshes are in voxels, is `voxel_size` the correct and complete conversion, or is
there a per-scan reconstruction scaling upstream that I'm not seeing? I'd rather ask than reverse-
engineer it from body lengths I half-remember from the literature — that reference is too weak to
diagnose a 7× discrepancy, which is why I've stopped rather than guessed.

Concretely this decides an estimand, not just a unit: with trustworthy absolute scale I can do
proper corpus allometry; without it I've frozen the analysis down to within-genus relationships and
scale-free shape ratios (HW/TBL, HL/TBL, HW/HL), which are defensible but a strictly smaller claim.

**One more thing, no data needed for this one.** replicAnt generates WOLO's synthetic training
images from 3D models digitized with scAnt and rigged in Blender — the same upstream pipeline this
registration work sits downstream of. The correspondence-accuracy work this summer has been about
is directly relevant to the fidelity of any future synthetic training data generated that way.

Let me know if the mass-labeled data exists somewhere I haven't checked.
