# Pre-registration — Y1: is pose the cause of the worker registration failure?

**Written and committed BEFORE the runs.** Fixed before any Y1 number was seen.

Date: 2026-08-30. Series letter `Y` (A–H, F, R, W, X in use). Folder: `diagnostics/pose_causality/`.

---

## 1. The claim under test, and why it now matters more than anything else

REPORT.md §6.4.1 establishes the single sharpest fact in the investigation:

| corpus | gen@10 / spread | gen@20 | gen@40 |
|---|---:|---:|---:|
| worker registrations | **0.99–1.00** | 0.98 | — |
| ALL_ANTS_CLEAN | **0.5383** | 0.4999 | 0.4586 |
| synthetic, exact correspondence | *0.42* | *0.00* | — |

**The pipeline registers clean scans and only surface-fits workers.** §6.1 attributes the
difference to **pose** — ethanol-preserved workers are contracted and folded, the reference corpus
posed and extended.

**That attribution has never been tested causally.** It is inference from an observed corpus
difference, and two things now make it decisive:

1. Every alternative we could act on has been closed: correspondence descriptors (W1–W5), pose
   initialisation (C14p/E1), candidate selection (F2/F4/F5), scale bounds (X1).
2. If pose *is* the cause, pose-normalisation of workers is the project. If it is not, the central
   failure has been attributed to the wrong thing and the shape-prior route is blocked for a reason
   nobody has identified.

> **H: pose magnitude alone, holding shape variation and correspondence machinery constant, degrades
> the learnable shape structure recovered by the pipeline.**

## 2. The trap this design avoids

The gen/spread metric operates on **rest-space shaped geometry** — `v_template + deform_verts +
shapedirs · betas` — i.e. *pose removed analytically*. Run on **ground-truth** parameters, pose
therefore **cannot** affect it, by construction. A design that fed GT geometry to the metric would
measure an identity and report a guaranteed null.

The causal chain must run **through the fitter**:

> pose magnitude ↑ → the fit degrades → correspondence in the *fitted* parameters degrades → the
> shape space built from those fits loses learnable structure.

So Y1 **fits** every corpus and builds the shape space from the **fits**, never from ground truth.
Ground truth is used only as the instrument's positive control (§5.1).

## 3. Design — a pure pose ladder with shape held exactly constant

`generate_synth_large.py::sample_batch` seeds once, then draws `betas` **before** `joint_rot`.
With the same seed and different `pose_scale`, this yields **identical betas** and pose that is
identically *directed* and merely rescaled. That is a clean one-factor ladder, verified by assertion
rather than assumed (§5.2).

- **Corpora**: `pose_scale ∈ {0.05, 0.15, 0.25, 0.50, 0.75}`, **n = 50** each, one shared seed.
  0.15 ≈ 13.8°/joint, the value `generate_synth_5000_lowpose.slurm` verified against real bench50
  specimens; 0.50 is the generator default.
- **Fitting**: identical recipe on every corpus — `D1_PROD.yaml` **as shipped** (it contains
  `w_scale: 0.052`; that is production, and X1 showed the term does not touch anterior anatomy, so
  it is held constant here rather than varied).
- **Metric**: leave-one-out PCA reconstruction of rest-space shaped geometry, `gen@k / spread`,
  k ∈ {1, 5, 10, 20, 40}, matching §6.4's definition (`spread` = k=0 = predicting the mean).

## 4. Pre-registered bar

Endpoint: **gen@20 / spread as a function of pose magnitude**, across the five corpora.

- **CONFIRMED** — gen@20/spread increases monotonically with pose magnitude across the ladder,
  **and** spans from **≤0.60** at `pose_scale = 0.05` to **≥0.90** at `0.75`. That range is chosen
  to bracket the two published anchors it would have to explain: ALL_ANTS_CLEAN at 0.50 and workers
  at 0.98.
- **PARTIAL** — a significant monotone increase that does not span that range. Pose contributes but
  does not account for the worker failure.
- **REFUTED** — no monotone relationship. **Pose is not the cause**, the §6.1 attribution is wrong,
  and the worker failure must be re-attributed (scan quality, topology defects, occlusion — none of
  which this design tests).

Monotonicity is judged on the ordered five points; Spearman ρ across corpora is reported with it.

## 5. Mechanism checks — the first three void the run

**5.1 Instrument validation (VOIDS).** The same metric, run on **ground-truth** rest-space geometry
for the same specimens, must recover the corpus's known dimensionality: since GT shape is exactly
`v_template + shapedirs · betas`, the data is **exactly `n_betas`-dimensional by construction**, so
**gen@k / spread must fall to ≤ 0.10 by k = n_betas**. If the implementation does not reach the
floor on data whose answer is known analytically, it is the wrong instrument and no Y1 number may
be read.

*Amended before any Y1 data existed, and the reason recorded rather than silently applied.* The
first draft of this check cited §6.4's published reference directly (gen@10 ≈ 0.42, gen@20 = 0.00).
Those figures were measured on synthetic data that was **exactly 13-dimensional**; this corpus is
built on `OmniAnt_25PCs` and is 25-dimensional, so a 20-mode reconstruction cannot be expected to
reach zero here and the published numbers do not transfer. The dimension-aware form above is the
correct instrument check, and it is strictly harder to pass by accident than a fixed threshold.

**5.2 One-factor ladder (VOIDS).** Assert `betas` are **byte-identical** across all five corpora,
and that `joint_rot` differs only by the scalar `pose_scale`. If shape varies across the ladder, the
experiment is confounded at source.

**5.3 Pose magnitude actually differs (VOIDS).** Report mean degrees/joint per corpus; must be
strictly increasing across the ladder. A ladder that does not vary its own factor tests nothing.

**5.4 Fit quality, reported alongside.** Chamfer and fscore per corpus. A collapse in gen/spread
accompanied by total fit failure at high pose is a *different* finding from a graceful degradation
of correspondence — the two must be distinguishable, so both are reported and neither is read
without the other.

## 6. What each outcome licenses — fixed now

- **CONFIRMED** → pose-normalisation of worker scans becomes the priority, and the shape-prior route
  is unblocked *conditional on* that normalisation. It does **not** license any particular
  normalisation method; that would be a separate pre-registration.
- **PARTIAL** → report the fraction attributable to pose and stop; the residual needs its own cause.
- **REFUTED** → a genuinely important negative: the project's central attribution is wrong, and the
  next step is identifying what actually distinguishes worker scans, not building anything.

## 7. Not claimed

This is synthetic. Real worker scans additionally carry scan noise, partiality, topology defects and
occlusion, **none of which vary in this ladder**. A CONFIRMED result shows pose is *sufficient* to
produce the degradation; it does not show pose is the *only* cause in real workers, and Y1 will not
claim that.
