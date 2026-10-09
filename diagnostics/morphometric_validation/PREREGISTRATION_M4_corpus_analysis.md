# Pre-registration — M4: large-corpus morphometric analysis (757 specimens, 172 genera)

> **STATUS 2026-09-10 — CLOSED.** M4-A/B/C/E complete on all 757; M4-D optional pending an external
> scale reference. Results and the hard stop: `M4_CONCLUSIONS.md`. B-i was found NOT EXECUTABLE
> (M4-B §1). This document is the frozen plan; it is not revised to match outcomes.

**Frozen analysis specification. Written BEFORE any biological association is inspected.**
Date: 2026-09-09. Folder: `diagnostics/morphometric_validation/`.

The point of freezing this is narrow and specific: the phenotype panel was selected on measurement
validity alone (`PHENOTYPE_PANEL.md`), and that selection is only meaningful if the biology is not
allowed to reach back and revise it. Every analysis M4 will run, and every way its result may be
read, is fixed here.

---

## 0. Inputs, and one blocker that changes the cost

**Panel (FROZEN, M3, not revisable by M4):**

| tier | traits | licence |
|---|---|---|
| **primary** | **HW**, **HL** | may carry a biological claim |
| secondary | *(none — deliberately empty)* | — |
| covariate | WL, TBL | normaliser / size axis only, never an endpoint |
| conditional | SL | reported only with its 6.08% pose Δ attached |
| **excluded** | FL, ML, GL, PetL | may not appear in any biological claim |

**Blocker 1 — the fits do not exist on this cluster, nor on the Drive.** `run_m1_fit_all.sh`'s 757-specimen run was
executed on JUWELS (`/p/scratch/cias-7/...`); only the meshes were synced to RWTH. M4 therefore
requires a fresh 757-specimen fitting run under the unmodified `D1_PROD.yaml` recipe
The mounted Google Drive was checked directly and does not have them either: 5453 files, zero
`.npz`, no `Stage_3*` anywhere. **Measured cost: 40 specimens fitted in 11m27s on one c25g GPU
⇒ ≈3.6 GPU-hours for 757** (an earlier 80-minute figure was an extrapolation and was wrong).
Not previously budgeted.

**Blocker 2 — `worker_ALT` has no trustworthy physical scale.** Established 2026-09-09, before
any M4 analysis: the raw `.obj` meshes carry a per-specimen scale, but that scale is **not a
consistent physical unit across specimens**. Against approximate known worker body lengths for
11 unambiguous species, the implied scale factor spans 182–1280 µm/mm (7×) and correlates with
known body length at only r=0.558 (physical scale would give ≈0.95). Within-species scale is far
tighter (median CV 0.14), so the inconsistency is *between* scans/batches, which is exactly the
axis a between-species allometry needs.

**Addendum 2026-09-09 — a candidate cause was found and does NOT explain it.** The Drive's
per-specimen antscan metadata (previously unused) records a `voxel_size` per scan: mesh coordinates
are in **voxel units**, and voxel size varies 1.22–6.11 µm across the corpus. This looked like the
whole explanation and is not. Of 279 metadata-matched specimens, **247 share the same 2.44 µm
voxel size** — it varies for only ~12%, so it cannot produce a 7× spread. Applying the correction
raises the correlation with approximate known body lengths from r=0.43 to r=0.71 but leaves the
spread at 7.2×, and within the single 2.44 µm group the implied scale is still wrong by 2–3× in
*both* directions. The preprocessing script was also checked (`prepare_antscan_data_for_mesh_
fitting.py`): it imports, cleans, decimates and exports with no rescaling step, so this is not a
processing bug either.

**The cause of Blocker 2 is therefore NOT established.** It is also not diagnosable with the
reference used so far — approximate body lengths recalled from the literature, which are
caste-dependent and too weak to carry this. Diagnosing it properly requires a real external size
reference. Until then §2's restrictions stand exactly as written; the voxel correction is
physically principled and should be applied, but it is **not** sufficient to unlock B-iii.

Checked against the obvious innocent explanation and it does not hold: bbox diagonal is inflated
by leg splay, so the test was repeated with a pose-robust proxy (mesh volume^(1/3), to which
outstretched legs contribute little). It gives the same verdict — r=0.492, implied-scale spread
7.5×, within-species CV 0.15 — so the inconsistency is in the scan scale itself, not in how the
specimens are posed. Two independent size proxies, same conclusion.

This is load-bearing for M4-B and is treated in §2, not worked around.

## The three nested questions

M4 is not "run biology because the methods are finished." It is the controlled final experiment
asking **what biological information survives all the methodological filters imposed so far** —
and it is the first experiment in this project whose primary object is biology rather than method.

The five analyses below are nested, not parallel. Each one's admissible sample is defined by the
one before it.

> **M4-1 — Does the frozen phenotype behave coherently on the corpus?** (§1)
> **M4-2 — Does validated morphology contain biological structure?** (§2–§4)
> **M4-3 — Does the interpretable phenotype agree with the high-dimensional one?** (§6)

## 0b. The estimands — what absolute scale costs us, stated as a constraint not a weakness

Because absolute per-specimen scale is unidentifiable on `worker_ALT` (§0 Blocker 2), M4's
estimands are restricted. This is a scientific constraint that is **reported**, not hidden or
worked around.

**Allowed**: within-genus relationships where scale is internally comparable; scale-free ratios;
shape proportions; variance decomposition; taxonomy; size-corrected analyses where the required
scale relationship is identifiable; comparison against scale-independent dense morphology; failure
analysis.

**Not allowed**: pooled absolute body-size allometry across `worker_ALT`; interpreting absolute
lengths as comparable across the corpus; using recalled literature body lengths as if they were
specimen-specific ground truth.

The project's endpoint was never "recover millimetres at all costs" — it was *recover meaningful
biological phenotype from imperfect 3D data*. The scale-free question is the defensible form of
exactly that.

## 1. M4-A — transfer, distributions, and the QC gate

**This runs FIRST and it gates everything after it.** The question is not "what does the biology
look like" but **what subset of the 757 actually satisfies the frozen panel's measurement
assumptions.** Treating every automated output as ground truth is the specific mistake this
ordering exists to prevent.

```
757 workers
     │
     ▼  frozen fitting/extraction pipeline (D1_PROD, unmodified)
     │
     ├── HW ── C2/C4 QC ──► valid ──────────────► phenotype
     │                  └─► failed ─────────────► failure class
     │
     └── HL ── AP-ordering QC (M3b C3)
                    ├── valid ──────────────────► phenotype
                    ├── AP inversion ───────────► failure class
                    └── extraction failure ─────► failure class
```

Reported: HW/WL and HL/WL distributions (median, IQR, 5–95%) by genus and pooled, singleton genera
named not binned; and the per-specimen QC table.

**M4-A carries its own biological result, and it may be the first one**: M3b found HW robust 40/40
while HL failed ~7.5% through head-axis inversion. If that rate holds at n=757, the finding is
*HW is robust to the dominant real-corpus head-orientation failure that compromises HL* — a
trait-specific reliability result, not a methods footnote. HW and HL are **not** interchangeable
primaries despite both clearing M3 on *Atta*.

Only measurements passing their QC gate enter §2–§6. Failure classes go to §5.

**The gate is not a filter — failure structure is itself a result.** Before any specimen is
excluded, M4-A tests whether the failure classes are **taxonomically structured**: is HL AP-inversion
distributed at random across the corpus, or concentrated in particular genera, subfamilies, or body
plans? Pre-registered because the temptation runs the other way: a QC step that quietly removes a
biologically coherent subgroup would bias every downstream taxonomic analysis while looking like
routine hygiene.

- If inversion is **random** w.r.t. taxonomy: exclusion is safe, and the admissible set is
  unbiased for §2–§6.
- If inversion is **concentrated** in a subgroup: that concentration is reported as a scientific
  finding in its own right — a body plan the fitter systematically mis-orients — and every
  downstream analysis must state that the subgroup is under-represented in its admissible sample.
  It may **not** be silently dropped.

Either way the admissible set is **frozen at the end of M4-A** and not revisited by §2–§6.

## 2. M4-B — scale-free and within-genus morphology

On the M4-A-valid subset only.

- **B-i.** Within-genus allometry for genera with n ≥ 5, where scale is internally comparable.
  Per genus with its own CI; never pooled into a corpus exponent.
- **B-ii.** Scale-free ratios — HW/TBL, HL/TBL, **HW/HL** — compared between genera. Scale-free by
  construction, so they survive Blocker 2 intact. These are **shape** claims and must be worded as
  such, never as allometric ones.
- **B-iii (PROHIBITED).** A pooled corpus-wide absolute allometric exponent. If an external
  per-specimen size reference is ever obtained, B-iii requires a *new* pre-registration, not an
  amendment to this one.

AntWeb was checked as a source for that reference and is not one: specimen records carry collection
data and images, not measurements. Recorded so it is not re-attempted.
## 3. M4-C — genus variance structure

**Variance decomposition is the primary analysis; classification is a secondary demonstration.**
The question is not "can we classify genus" but **how much of the recovered morphological variation
is structured by genus, versus within-genus variation** — which is much closer to the biological
meaning of a morphometric phenotype. A mixed model per trait
(`trait_ratio ~ 1 + (1|subfamily/genus/species)`) partitioning variance into between-subfamily,
between-genus, between-species, and residual (within-species + measurement error) components, with
the residual floor anchored on M3's reproducibility estimate.

Classification accuracy is deliberately **not** the headline, and must not be run first. The Z9–Z13 series already
established the classification framing saturates at 16.3% top-1 / 7.21× lift and that its
headroom sits in the classifier rather than the morphometrics — a further accuracy number would
re-answer a question already answered. Variance decomposition answers the question actually open:
*how much of the taxonomic signal do two validated traits carry, and at what level of the
hierarchy does it live?*

## 4. M4-D — explicit size correction

Every M4-C result is reported **twice**: uncorrected, and with the size axis (WL) explicitly
removed. Size correction is stated as a modelling choice with consequences, not applied silently —
size *is* a biological trait, and removing it removes real signal along with the confound. Where
the two versions disagree, the disagreement is the reported result.

## 4b. Scan-quality metadata — available for only 37% of the corpus

The Drive carries per-specimen antscan metadata (`voxel_size`, `processed_hole_count`,
`processed_mesh_smoothness`, `processed_face_size_cov`, caste, tribe, locality) at
`antscan_data/<specimen>/<specimen>.json`. All 784 of the 785 folders were retrieved
(`/hpcwork/nao48500/antscan_meta`, distilled to `/hpcwork/nao48500/antscan_voxel.json`, keyed by
the JSON's own filename).

**Coverage is 279/757 (37%) and this is permanent, not a download gap** — 478 `worker_ALT`
specimens have no `antscan_data` folder. Any M4 analysis conditioning on scan quality is therefore
restricted to that 37% subset and must say so; a scan-quality covariate may NOT be silently applied
corpus-wide, and specimens lacking metadata may NOT be dropped from M4-A/B/C, since missingness is
not random with respect to genus.

## 5. The failure map

A per-specimen, per-trait table of which extractions failed and how, carried forward from M3b's
C1–C4 checks applied to all 757. Reported as a first-class result, not an appendix: **which
phenotypes cannot be recovered, and why, is the deliverable** — equal in standing to the traits
that succeed. Failure classes: mesh/scan pathology, fit failure, extraction failure.

## 6. M4-E — the bridge: does the interpretable phenotype agree with the dense one?

The endpoint the rest of M4 exists to reach, and the one that turns two disconnected results
("dense shape carries genus signal", "HW is a valid measurement") into a single claim.

> Do HW, HL and the scale-free morphometric axes correspond to biologically structured directions
> in the dense 3D phenotype?

Establishing that builds the chain **3D shape → interpretable morphometrics → biological
structure**, i.e. that the validated scalar measurements capture identifiable components of the
much richer 3D phenotype, rather than sitting beside it.

**The claim a positive result licenses, fixed now.** The dense-shape reference signal is itself
modest, not overwhelming ground truth: the Z-series reached top-1 0.179 and nest-mate agreement
18.9% feeding `betas` alongside traits ([[project-z11-z12-classifier-push]]). So a positive bridge
supports exactly:

> *Validated interpretable morphometrics recover a measurable component of biological structure
> already present in the dense 3D representation.*

and **not** "the scalar traits explain the biological structure." Recorded before the result so the
wording is not chosen after seeing the effect size.

This subsumes the chamfer-vs-traits comparison as its negative control, with all three outcomes
pre-declared publishable:

- **traits beat chamfer** — validated morphometrics carry signal the fit-quality metric does not.
- **chamfer beats traits** — the dense shape carries genus signal the two-trait panel discards; a
  real limitation of scalar phenotyping, reported as such.
- **neither separates genera** — the corpus registration is at its noise floor for this question
  (consistent with Z6/Z7), and *that* is the finding.

**And if M4 finds weak biology, that is a legitimate result, not a failed experiment**: the
conclusion would be that the pipeline recovers certain measurements reliably while those validated
measurements do not capture substantial taxonomic structure at the tested scale. That is a strong
result precisely because it separates **measurement validity from biological informativeness** —
which are not the same thing, and which this project is the first to hold apart.

**None of the three licenses a panel change.** Recording that here is what makes the
third outcome reportable rather than a trigger to go looking for a fourth.

## 7. Do-NOT list — binding

1. **Do not revise the phenotype panel** on the basis of any M4 result. A trait that looks
   interesting here has not passed accuracy × reproducibility × biological validity.
2. **Do not promote an excluded trait** (FL, ML, GL, PetL) into a biological claim, however clean
   its corpus distribution looks. Their exclusions rest on evidence M4 does not measure.
3. **Do not open a method branch to rescue a trait.** Trait failure is a result. This is the
   standing project constraint (`project-direction-measurement-validity-not-fit-quality`), and M4
   is exactly where the temptation peaks.
4. **Do not report a pooled corpus allometric exponent** (§2, B-iii).
5. **Do not tune the fitting recipe.** M4 runs `D1_PROD.yaml` unmodified. A recipe change
   invalidates the M1/M2/M3 validation chain the panel rests on.
6. **Do not substitute classification accuracy for variance decomposition** in M4-C (§3).
7. **Do not drop the negative results.** §5's failure map and §6's third outcome are deliverables.
8. **Do not run biology before the QC gate.** §1 defines the admissible sample; §2–§6 run on it.
   No analysis may treat an un-gated automated output as ground truth.
9. **Do not lead with pooled genus classification.** §3 is variance decomposition first,
   classification second.
