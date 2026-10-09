# GLAD trait landmark protocol — SMILify

**Version 0.1 — 2026-09-02. Pre-registration draft. Not yet annotated against.**

This document has two consumers and one rule.

- **The annotator** (expert) places these points on specimen meshes.
- **The code** (`trait_extract.py`) reads the same definitions as template vertex indices.
- **The rule:** if the two drift apart, the resulting error is *convention mismatch, not accuracy*.
  This is the `b_h` failure from `RESULTS_G1_20260831.md`, where a landmark with no agreed
  definition returned annotator offsets spanning 1.2%–49.5% and had to be dropped from every
  alignment. One document, two views of it, no second definition anywhere.

Traits follow the GlobalAnts / GLAD schema (Parr et al. 2017, *Insect Conserv. Divers.* 10:5–20)
and AntWiki measurement conventions. **Where the literature carries more than one convention, this
document does not silently pick one — it marks the choice as OPEN and requires the expert to
record which was used.**

---

## 1. The methodological problem this document exists to solve

**Every published ant measurement is defined in a 2D view.** "Maximum head width in full-face
view." "The diagonal of the mesosoma in profile." Those definitions assume a specimen under a
microscope, rotated by hand until a standard aspect is presented, and measured with an ocular
micrometer in that projection.

We measure on a 3D mesh. There is no view. So every trait needs an explicit **3D
operationalisation**, and that operationalisation is a *choice* — one that changes the number.
"Maximum head width in full-face view" could mean:

- (a) the maximum 3D Euclidean distance between any two head vertices in the mediolateral direction;
- (b) the maximum extent of the head after projection onto the coronal plane;
- (c) the width measured in the plane containing the eyes.

These are not equal, and they diverge most on exactly the aberrant forms (*Cephalotes*,
*Odontomachus*) that the benchmark set was chosen to span.

**This is a genuine contribution, not a chore.** Nobody has written down the 3D operationalisation
of the standard ant measurement set. It has to be pre-registered before any number is computed,
and it belongs in the paper's methods.

**Decision adopted here, for every trait: (a), the direct 3D distance between two defined points,
with no projection.** Reasons: it is the only option that is pose-invariant without first
constructing a canonical view; it is what a caliper on a physical specimen approximates; and it
requires no additional convention about how the view plane is defined. Where a trait is a maximum
over a region rather than two fixed points (Type III below), the maximum is taken over the 3D
distance within that region, constrained to the mediolateral direction where the definition says
"width".

**Consequence to state in the paper:** our values will differ slightly and systematically from
microscope measurements of the same species, because a projected width is always ≤ a 3D width.
This is a *constant, correctable* offset of exactly the kind B3 tests for — and it is a reason to
expect the bias term to be non-zero even for a perfect fit.

---

## 2. Template and axis convention

Model: `3D_model_prep/SMIL_OmniAnt.pkl` — **10,229 vertices, 20,454 faces.**

Axis convention, **verified from joint positions, not assumed**:

| axis | direction | evidence |
|---|---|---|
| **+X** | anterior | mandibular apex x = 0.607; gaster x = −0.30 |
| **Y** | mediolateral, midline at y = 0 | head spans y = ±0.163 symmetrically |
| **+Z** | dorsal | hind femur joint z = 0.083 sits above hind tibia z = −0.066 |

**The template is exactly bilaterally mirror-symmetric.** All 10,229 vertices have an exact mirror
partner (median mirror distance 0.0; 100% within 1e-3). So:

- **Click one side only.** Left-side indices are derived by mirror map, never placed by hand.
- **223 vertices lie exactly on the midline** (|y| < 1e-4). Every "midpoint" landmark must come
  from this set — which reduces those searches from 10,229 candidates to 223.

---

## 3. Landmark table

**Bookstein type** governs how error is reported and must never be averaged across types:

- **Type I** — anatomically discrete junctions. Lowest error.
- **Type II** — curvature extrema. Intermediate.
- **Type III** — constructed extremal points, sliding along one axis by definition.
  Direction-dependent error, structurally larger. **Reported on its own axis in the certificate.**

`candidate` = the automated geometric proposal from `landmark_candidates.py`. It is **a starting
point for verification, not a definition.** Confidence says how much scrutiny each needs.

| # | landmark | type | definition (GLAD / AntWiki) | candidate vtx | confidence |
|---|---|---|---|---:|---|
| 1 | `clypeal_margin_ant_mid` | I | Anteriormost point of the median clypeal margin, on the midline. Excludes mandibles and labrum. Where the margin is concave, the midpoint of a transverse line spanning the apices of the projecting portions. | **773** | medium |
| 2 | `cephalic_margin_post_mid` | II | Midpoint of the posterior cephalic (occipital) margin. Where concave, same midpoint-of-transverse-line rule. | **511** | medium |
| 3 | `mandibular_apex_r` | I | Apical tooth of the right mandible — the point furthest from the mandibular articulation. | **1662** | **high risk** |
| 4 | `antennal_insertion_r` | I | Centre of the right torulus / antennal socket, where the scape articulates with the head. | **585** | medium |
| 5 | `scape_apex_r` | I | Distal end of the right scape. **SL excludes the basal condyle and neck**, so the proximal endpoint is the articulation, not the basal bulb. | **1800** | medium |
| 6 | `wl_anterior_r` | I | Point at which the pronotum meets the cervical shield (Weber 1938), in profile. **Right side, not midline — see the same-side rule.** | — | **none — place by eye** |
| 7 | `wl_posterior_r` | I | Posterior basal angle of the metapleuron, in profile. Lateral sclerite, \|y\| ≈ 0.09. | — | **none — place by eye** |
| 8 | `petiole_ant_mid` | II | Anterior margin of the petiole, on the **ventral** midline outline. | **2699** | medium |
| 9 | `petiole_post_mid` | II | Posterior margin of the petiole, same **ventral** outline. | **5158** | medium |
| 10 | `gaster_apex_mid` | II | Posteriormost midline point of the gaster. | **5193** | medium |

Hind femur proximal and distal endpoints are **already rig joints** (`l_3_fe_r`, `l_3_ti_r`) and
need no placement — see the caveat in §6.

### Type III regions (a rule over a vertex set, not a point)

| region | rule | vertices | confidence |
|---|---|---:|---|
| `head_region` | maximum mediolateral extent within the head part → **HW** | 2173 | high |
| `pronotum_region` | maximum mediolateral extent → **PW** | 469 | **LOW — placeholder** |
| `eye_region_r` | maximum diameter → **EL**, **HW2** | **0** | **ABSENT** |

---

## 4. OPEN conventions — the expert must choose and record

These are genuine divergences in the published literature. Picking one silently is how convention
mismatch enters, so each is recorded as a field in the annotation file.

**O1. HW — across the eyes, or excluding them? — RESOLVED 2026-09-04: ACROSS THE EYES.**
The fitted side cannot exclude them (no eye geometry is separable, 0 vertices), so excluding them
in the truth would measure convention mismatch rather than accuracy. Mandibles, antennae and scapes
are excluded from HW throughout. Original discussion follows. GlobalAnts/Parr states maximum head width across
the eyes; several taxonomic conventions measure maximum width *excluding* the eyes, and others
measure behind the eyes. On genera with strongly protruding eyes (*Gigantiops* — the largest-eyed
specimen in the benchmark set) these differ substantially. **Must be recorded per annotation.**

**O2. ML — from which proximal point?** Mandible length is variously measured from the mandibular
apex to the anterior clypeal margin, or along the outer margin from the mandibular base. The
closing plan uses the former. **Confirm.**

**O3. HW2 / EP.** The plan defines HW2 as head width behind the eyes and EP = HW2 − HW1. This
formulation should be checked against its source before use; it is not the AntWiki standard
wording. Currently **unverified** and blocked on O4 in any case.

**O4. PetL on a two-segment waist.** Myrmicinae (*Pheidole*, *Cephalotes*, *Acromyrmex*,
*Crematogaster*, *Cyphomyrmex*, *Aphaenogaster*) have petiole **and** postpetiole; Formicinae,
Dolichoderinae and most Ponerinae have petiole only. The template encodes a two-segment waist
(`b_a_1`, 72 verts; `b_a_2`, 127 verts). **On a one-segment-waist specimen, the template's
`b_a_2` corresponds to anterior gaster, not to a postpetiole.** Landmarks 8 and 9 are therefore
comparable across all specimens, but any postpetiole measurement is not. Recommendation: measure
petiole only, and record waist segment count per specimen.

---

## 5. What is NOT computable, stated plainly

**EL and HW2 cannot be computed at all.** There is no eye geometry separable from the head part —
the eye region returns **0 vertices**. Two options: hand-paint an eye vertex group on the template
(the model is textured, so the eye boundary is visible), or drop both traits from the certificate
and say why. **Recommendation: drop them for now**; they are the two traits with the weakest link
to the paper's argument, and painting an eye boundary introduces a Type III region with no
published 3D convention.

**WL is at risk, and it is the most important trait in the set** — every other trait is
conventionally normalised by it. Weber's length needs two points inside the mesosoma, and
**neither the pronotum, the cervical shield, nor the metapleuron exists as a modelled structure**:
`PART_DEFS` treats the whole mesosoma as one undifferentiated 1,379-vertex blob (`n == "b_t"`).

The automated proposal for landmark 7 **fails verification and is rejected**: it lands at
y = −0.0072, essentially on the midline, whereas the metapleuron is a lateral sclerite at
|y| ≈ 0.09. The extremal rule found a posteroventral *midline* point, which is not the anatomy.
Landmarks 6 and 7 must both be placed by eye against the profile view.

**PW is a placeholder.** The `pronotum_region` is currently just the anterior third of `b_t` by x.
It needs hand-painting.

**`b_a_5` carries zero dominant vertices** — the terminal gaster joint drives no geometry, the same
pathology as the four wing joints already documented in `part_groups.py`. Gaster surface is carried
by `b_a_3` and `b_a_4`. No trait should reference `b_a_5`.

---

## 6. Trait definitions, as computed

| trait | computed as | landmarks | status |
|---|---|---|---|
| **HL** head length | distance(1, 2) | 1, 2 | ready |
| **HW** head width | max mediolateral extent in `head_region` | III | ready, pending O1 |
| **ML** mandible length | distance(3, 1) | 3, 1 | ready, pending O2 |
| **SL** scape length | distance(4, 5) | 4, 5 | ready |
| **WL** Weber's length | distance(6, 7) | 6, 7 | **blocked** — §5 |
| **PW** pronotum width | max mediolateral extent in `pronotum_region` | III | **blocked** — §5 |
| **FL** hind femur length | distance(J`l_3_fe_r`, J`l_3_ti_r`) | joints | ready, **caveat below** |
| **PetL** petiole length | distance(8, 9) | 8, 9 | ready, pending O4 |
| **TBL** total body length | ML + HL + WL + PetL + gaster | 1,2,3,6,7,8,9,10 | blocked on WL |
| **EL** eye length | — | — | **not computable** |
| **HW2 / EP** | — | — | **not computable** |

**FL caveat.** Using the rig joints makes FL a *joint-to-joint* distance — a rotation-pivot
measurement, which is the exact quantity class that G2 found unreliable (`leg_prox` real R = 0.032
against synthetic 0.756). GLAD FL is a surface length on the femur. These are different
quantities, and FL should be flagged in the certificate as measured on the rig rather than the
surface until surface endpoints are placed.

---

## 7. Standing risks

**R1 — mandibular apex correspondence (landmark 3).** The template is a *mean* mandible. The
benchmark set deliberately includes *Odontomachus* (trap-jaw, `mandible_slenderness` max 5.36) and
*Aenictus* (`mandible_index` min 0.49). A single template vertex index transfers to every specimen
*through the fit*, so if correspondence on the mandible fails on those forms, ML is garbage on
exactly the specimens that matter most. This is testable and should be checked first.

**R2 — the certificate inherits fit error and landmark error together.** A trait's error against
expert truth confounds (a) the fit misplacing the surface and (b) our template index being the
wrong vertex. Only R1-style visual verification separates them, and it must happen before scoring.

**R3 — Type III traits have no observer floor of the same kind.** A sliding landmark's
repeatability is direction-dependent; the expert-vs-annotator distance along the sliding axis is
not comparable to a Type I distance. Report separately.

---

## 8. Artifacts

- `landmark_candidates.py` — generates the proposals above.
- `landmark_candidates.json` — machine-readable, consumed by the extraction layer.
- This document — the definition of record. **Vertex indices are provisional until visually
  verified in Blender; nothing downstream should be run before that.**

---

# Appendix A — placement guide

Written for someone who is not a myrmecologist. For each landmark: what the structure is, where it
sits on this template in template coordinates, how to confirm you are on it, and what it is most
easily confused with.

**Orientation first.** In Blender, the template faces **+X**. Numpad 1 = front view looks down −Y
(this is the ant's *profile*). Numpad 3 = side view looks down +X, i.e. straight at the ant's face
(**full-face view**). Numpad 7 = top (**dorsal view**). Turn on the N-panel: for a selected vertex
in Edit Mode it shows the coordinate, so you can compare against the numbers below.

**Reference boxes**, so you always know which structure you are inside:

| structure | x (ant.→post.) | y (lateral) | z (dorsal) |
|---|---|---|---|
| head capsule | 0.274 … 0.573 | ±0.163 | −0.069 … 0.183 |
| mesosoma | −0.154 … 0.291 | ±0.098 | −0.054 … 0.178 |
| right mandible | 0.484 … 0.607 | −0.118 … −0.036 | −0.139 … −0.006 |
| right scape | 0.550 … 0.576 | −0.308 … −0.061 | 0.031 … 0.105 |
| petiole | −0.231 … −0.145 | ±0.035 | 0.006 … 0.088 |
| postpetiole | −0.329 … −0.208 | ±0.060 | 0.018 … 0.113 |
| gaster | −0.696 … −0.303 | ±0.134 | −0.156 … 0.126 |

Useful joint anchors: hind coxa `l_3_co_r` = (−0.121, −0.042, 0.038); mid coxa `l_2_co_r` =
(−0.025, −0.048, 0.021); neck `b_t`/`b_h` ≈ (0.28, −0.01, 0.05); scape articulation `an_1_r` =
(0.521, −0.060, 0.004).

**Note the scape runs sideways, not forward** (x barely changes, 0.550→0.576; y sweeps
−0.061→−0.308). The template's antennae are held out laterally.

**THE SAME-SURFACE RULE — read this before placing any pair.** A trait named "length" or "width"
must be measured between points that differ along *that* axis and not much along the others.
Picking each endpoint by an independent extremal rule does not guarantee this, and it has already
produced two artefacts here:

- **WL** paired a midline anterior point with a lateral posterior one, so the diagonal picked up
  mesosoma *width*. Fixed by constraining both endpoints to the same lateral band.
- **PetL** paired a dorsal anterior point with a ventral posterior one, so 60% of the measured
  "length" was height. Fixed by pinning both to the ventral outline.

**But not every off-axis component is an artefact.** HL runs from the ventral-anterior clypeal
margin to the dorsal-posterior occipital margin, and ML from the clypeal margin down to the
mandibular apex — in both cases the large vertical component is real anatomy, because the head is
pitched relative to the body axis and the mandibles hang beneath it. The test is not "is the line
diagonal" but **"is the diagonal the anatomy, or the sampling?"** When in doubt, check whether the
two endpoints were chosen by *independent* extremal rules — that is the signature of the artefact.

---

### 1. `clypeal_margin_ant_mid` — candidate v773 at (0.573, 0.000, 0.007)

**The structure.** The clypeus is the sclerite plate on the front of the head, sitting *between*
the antennal sockets above and the mandibles below. Its anterior edge is the anterior clypeal
margin. You want the midline point of that edge — the frontmost point of the head capsule proper.

**How to confirm.** In full-face view it is dead centre horizontally, immediately above the gap
between the two mandibles. In profile it is the anteriormost point of the head *excluding* the
mandibles.

**Confusable with.** The mandibles (which project further forward — note they reach x = 0.607,
beyond the head's 0.573) and the labrum, a small flap tucked behind the clypeus above the mandible
bases. If the point is at negative z it is probably on a mandible, not the clypeus.

**Where the margin is concave** (some genera have a notched clypeus): use the midpoint of a
straight line spanning the two projecting corners, not the bottom of the notch.

---

### 2. `cephalic_margin_post_mid` — candidate v511 at (0.296, 0.000, 0.102)

**The structure.** The posterior (occipital) margin is the rear edge of the head capsule. In
full-face view it is the back outline of the head. You want its midpoint.

**How to confirm.** On the midline, at the posterior end of the head part. In profile it should
sit on the *outline* where the dorsal surface of the head turns down toward the neck.

**Confusable with — and this is the likely error.** The **occipital foramen**, the hole through
which the neck passes into the mesosoma. That opening is posterior and ventral. The margin you
want is the visible rear edge in full-face view, not a point inside the foramen. If your point is
at low z and buried between head and mesosoma, you are in the foramen.

Note the head starts at x = 0.274 and the mesosoma reaches x = 0.291 — **they overlap**, so the
neck region is genuinely ambiguous by coordinate alone. Judge it visually.

---

### 3. `mandibular_apex_r` — candidate v1662 at (0.602, −0.038, −0.136)

**The structure.** The apical tooth — the sharp tip of the right mandible, the most distal point
of the jaw.

**How to confirm.** It is the frontmost point of the whole animal (x = 0.607 is the mandible's
maximum, ahead of the head's 0.573) and it is *ventral* (negative z). In full-face view the
mandibles cross below the clypeus.

**Confusable with.** The maxillary and labial palps (small finger-like mouthparts behind the
mandibles) and the left mandible if the two cross. Check y is negative — right side.

**This is the highest-risk landmark in the set.** The template is an averaged mandible. On
*Odontomachus* (trap-jaw, mandible slenderness 5.36) and *Aenictus* the real mandible is a
radically different shape, so even a perfectly placed template index may not land on the apex
after the fit.

---

### 4. `antennal_insertion_r` — candidate v585 at (0.550, −0.072, 0.013)

**The structure.** The torulus — the circular socket on the front of the head where the antenna
articulates. You want its centre.

**How to confirm.** In full-face view, one of the two sockets flanking the midline above the
clypeus. It should be very close to the scape articulation joint at (0.521, −0.060, 0.004).

**Confusable with.** The frontal lobes — flanges that in many ants partly cover the sockets. Aim
for the socket centre, not the lobe edge.

---

### 5. `scape_apex_r` — candidate v1800 at (0.575, −0.308, 0.096)

**The structure.** Ant antennae are *geniculate* — elbowed. The scape is the long first segment
running from the head to the bend; everything past the bend is the funiculus. **The scape apex is
the elbow, not the tip of the antenna.**

**How to confirm.** Follow the antenna outward from the socket and stop at the **first bend**.
Verified against the rig: the elbow is 0.260 from the scape articulation, the far tip of the whole
antenna is 0.460 — **using the tip would overestimate SL by 77% on every specimen.** The two
segments' vertex sets meet at a sharp boundary, `an_1` ending at y = −0.308 / x = 0.576 and `an_2`
beginning at y = −0.306 / x = 0.576. Candidate v1800 sits on that ring and is correct.

**Do not snap to the `an_2_r` joint.** Its rig pivot is at z = 0.162, *above* the surface of both
segments (which span z 0.031–0.105 and −0.047–0.095) — a rotation pivot floating off the mesh.
Navigate by the vertex boundary instead.

**Also note `an_3_r` carries only 3 vertices** — effectively degenerate, the same pathology as
`b_a_5` (zero verts) and the wing joints. The template models the funiculus as two segments rather
than the real ~10–12 antennomeres, and the distal one is nearly empty. Nothing should depend on it.

**Convention that matters.** GLAD scape length **excludes the basal condyle and neck** — the small
bulb and constriction at the proximal end. So the proximal endpoint is landmark 4 (the
articulation), not the outer edge of the bulb. Getting this wrong inflates SL on every specimen.

---

### 6. `wl_anterior_r` — UNPLACED, no usable candidate

**The structure.** Weber's length starts at the point where the **pronotum** meets the **cervical
shield**. The pronotum is the first and largest dorsal plate of the mesosoma, immediately behind
the head. The cervical shield / neck is the narrow connection between head and pronotum. You want
the anterior limit of the pronotum in profile, *excluding* the neck.

**How to find it.** Profile view (Numpad 1) — WL is *the diagonal of the mesosoma in profile*,
so it is the side view, not dorsal and not posterior. The diagonal runs front-upper to back-lower.

**THE SAME-SIDE RULE.** Place this on the **right side**, at a lateral position comparable to
`wl_posterior_r` — *not* on the midline. The classic measurement is read off a profile image where
both endpoints lie in the same lateral plane; measuring 3D from a midline point to a lateral one
adds a sideways component the real measurement never has. The magnitude is small (~1%), but **WL
is the normaliser for every other trait**, so a lateral component leaks mesosoma *width* into every
normalised trait as a shape contaminant. Verified on the template: with both endpoints in the same
lateral band the 3D and profile distances agree to four decimals (0.3659 vs 0.3659).

**Where the anterior margin sits.** The mesosoma's profile silhouette reaches furthest forward at
z ≈ 0.09–0.10 and recedes both above and below:

| height z | outline reaches x |
|---|---|
| 0.12 … 0.18 | 0.269 |
| **0.08 … 0.12** | **0.291 — anteriormost** |
| 0.04 … 0.08 | 0.285 |
| 0.00 … 0.04 | 0.234 |
| −0.06 … 0.00 | 0.152 |

The neck articulation is at z = 0.038, *below* that bulge — the pronotum arches forward over the
top while the cervix attaches lower down.

**Confusable with.** The neck itself, and the pronotal collar in genera that have one. Also
remember head and mesosoma overlap in x, so the head may occlude this point — hide the head part
or use wireframe.

---

### 7. `wl_posterior_r` — UNPLACED, no usable candidate

**The structure.** The **metapleuron** is the lateral plate on the side of the rear mesosoma,
above and behind the hind leg's insertion; in most ants it carries the metapleural gland bulla.
Weber's length ends at its posterior basal angle — the posteroventral corner of that plate.

**How to find it.** Profile view, right side. It is **lateral** — expect |y| ≈ 0.08–0.098, *not*
the midline. Posterior and ventral within the mesosoma: expect x ≈ −0.14 to −0.15 (the mesosoma
ends at −0.154) and z ≈ −0.05 to 0.00. It sits just behind and below the hind coxa joint at
(−0.121, −0.042, 0.038).

**Why the automated proposal was rejected.** It returned (−0.105, −0.007, −0.048): the right
posteroventral region, but at y = −0.007, essentially on the midline. The extremal rule found the
mesosoma's ventral keel rather than a lateral plate. **If your point has |y| below about 0.05, you
are on the midline and it is wrong.**

**Confusable with.** The propodeum (the rearmost dorsal segment of the mesosoma, above the
metapleuron) and the propodeal lobes. Some authors end WL at the propodeal lobe instead — that is
convention O2-adjacent and should be recorded if you deviate.

---

### 8. `petiole_ant_mid` — candidate v2699 at (−0.153, 0.000, 0.006)

**The structure.** The petiole is the first waist segment, the isolated node between mesosoma and
gaster. You want the anterior margin on the midline, **on the ventral outline** — see the same-line
rule under landmark 9.

**How to confirm.** The petiole is a small, clearly separate lump spanning x −0.231…−0.145. The
mesosoma ends at −0.154, so the anterior margin is right at that junction. This point sits low
(z = 0.006), on the underside, not on the dorsal node.

---

### 9. `petiole_post_mid` — candidate v5158 at (−0.231, 0.000, 0.013)

**The structure.** The posterior margin of the same segment, where it meets the postpetiole (in
Myrmicinae) or the gaster (in everything else).

**THE SAME-LINE RULE — resolved.** The petiole's midline is only **8 vertices**, and they form two
distinct rows: a dorsal edge (x = −0.151, −0.171, −0.188) and a ventral edge (x = −0.153, −0.164,
−0.186, −0.209, −0.231). Picking "frontmost" and "backmost" independently lands them on *opposite*
rows, and the resulting line is a diagonal through the segment: 0.0994 measured, of which only
0.0794 is longitudinal — **60% of the "length" was height**.

Both points are therefore pinned to the **ventral outline**, the row that spans the segment's full
length, matching the standard reading of PetL as maximum petiole length in profile. This gives
0.0786 with only 8.6% vertical component. `petiole_ant_mid` was moved from v5144 (dorsal) to
**v2699** (ventral) for this reason.

**Recorded limit.** The two outlines genuinely disagree about where the petiole ends — ventral runs
to −0.231, dorsal stops at −0.188. With 72 vertices in the whole segment and 8 on the midline, PetL
is a low-resolution quantity on this template, and that belongs in the certificate as a caveat.

---

### 10. `gaster_apex_mid` — candidate v5193 at (−0.696, 0.000, −0.098)

**The structure.** The rearmost point of the gaster, on the midline — the tip of the abdomen.

**How to confirm.** It should be the most posterior point of the entire animal apart from the
trailing legs. Note it comes from `b_a_4`: `b_a_5` carries **zero** vertices, so do not go looking
for a terminal segment to click.

---

### Before you export

- Every `*_mid` landmark must have **y = 0.000** exactly. There are only 223 midline vertices; if
  the N-panel shows a non-zero y, you are off the midline and the left/right mirror will be wrong.
- Every `*_r` landmark must have **negative y** (right side). Positive y is the left side, and the
  code derives left from right by mirroring.
- Run "Check mesh" once more before exporting.
