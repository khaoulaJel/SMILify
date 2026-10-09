# Y1 — PARTIAL. Pose is a real cause, and it is quantitatively insufficient to explain the worker failure.

Bar fixed in `PREREGISTRATION_Y1_pose_causality.md` **before** the run. Job 3350661, 34 min. Pure
pose ladder, n=50 per rung, betas byte-identical across rungs, fits from `D1_PROD.yaml` as shipped.

---

## Endpoint

`gen@k / spread` computed from the **fitted** parameters (1.0 = no better than predicting the mean):

| pose_scale | °/joint | gen@10 | **gen@20** | gen@25 | gen@40 | vert-err to GT |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 4.57 | 0.6805 | **0.3691** | 0.2950 | 0.2874 | 0.08925 |
| 0.15 | 13.71 | 0.6494 | **0.4183** | 0.3499 | 0.3362 | 0.10320 |
| 0.25 | 22.86 | 0.6693 | **0.4821** | 0.4497 | 0.4334 | 0.13917 |
| 0.50 | 45.71 | 0.7607 | **0.6241** | 0.5839 | 0.5763 | 0.24555 |
| 0.75 | 68.57 | 0.7525 | **0.6276** | 0.6017 | 0.5948 | 0.30914 |

- **Monotone: yes**, on every rung. **Spearman ρ = 1.000, p = 1.4e-24.**
- **Spans ≤0.60 → ≥0.90: no.** Lowest 0.3691, highest **0.6276**.

**Verdict: PARTIAL.** Pose degrades the learnable shape structure recovered by the pipeline —
unambiguously, and with a perfectly ordered dose-response — but it **saturates around 0.63 and
cannot reach the worker level of 0.98–1.00**.

## What this settles, quantitatively

Set the two published anchors against the ladder:

| | gen@20/spread |
|---|---:|
| synthetic @ 13.7°/joint (≈ real bench50 pose) | **0.4183** |
| **ALL_ANTS_CLEAN (real, clean)** | **0.4999** |
| synthetic @ 68.6°/joint (far beyond real workers) | **0.6276** |
| **worker registrations (real)** | **0.98** |

Two readings fall out, and they point in opposite directions:

**1. The clean corpus is fully consistent with pose alone.** At a realistic pose magnitude
(13.7°/joint, the value verified against real bench50 specimens) the synthetic pipeline lands at
0.42, against ALL_ANTS_CLEAN's measured 0.50. Clean-scan registration behaves exactly as
pose-only degradation predicts. Nothing else needs invoking to explain it.

**2. The worker failure is not explicable by pose.** At **68.6°/joint** — a pose regime far more
extreme than any real ant, five times the realistic value — the pipeline still only reaches 0.63.
Real workers sit at **0.98**. The ladder cannot get there, and it has stopped climbing: the last
two rungs are 0.6241 and 0.6276, flat, while pose magnitude rose 50%.

**So §6.1's attribution is partially right and quantitatively insufficient.** Pose is a genuine
cause with a clean dose-response, and it accounts for the clean-corpus level. It does **not**
account for the worker failure, and no amount of additional pose explains the remaining gap from
0.63 to 0.98.

## The saturation is itself informative

Fit quality (mean vertex error to GT) keeps rising monotonically across the whole ladder — 0.089 →
0.103 → 0.139 → 0.246 → **0.309** — while `gen@20/spread` flattens after 0.50. The fits get steadily
worse; the *shape-structure* damage plateaus.

That distinguishes two things the pre-registration required be distinguishable (check 5.4): this is
**not** "the fit collapses and takes the shape space with it". Beyond ~45°/joint the fit degrades
further without destroying more learnable structure. Whatever destroys structure in real workers is
therefore **not** simply "a worse fit" — the ladder shows that worse fits alone stop mattering.

## Mechanism checks

**5.1 — instrument validation (VOIDING): PASS, and cleanly.** On ground-truth geometry, which is
exactly 25-dimensional by construction:

| gen@1 | gen@5 | gen@10 | gen@20 | **gen@25** | gen@40 |
|---:|---:|---:|---:|---:|---:|
| 0.9701 | 0.9040 | 0.7460 | 0.3989 | **0.0000** | 0.0000 |

Exactly 0.0000 at k = n_betas = 25, the analytically known floor. And the row is **byte-identical
across all five rungs** — confirming the design's central premise directly: the metric on ground
truth is pose-invariant by construction, which is precisely why the endpoint had to be computed
from the fits. Had this experiment been run on GT geometry it would have returned five identical
rows and a guaranteed null.

**5.2 — one-factor ladder (VOIDING): PASS.** `betas` byte-identical across all five corpora,
`joint_rot` exactly rescaled by `pose_scale`. Asserted at generation.

**5.3 — pose magnitude varies (VOIDING): PASS.** 4.57 → 13.71 → 22.86 → 45.71 → 68.57 °/joint,
strictly increasing.

**5.4 — fit quality: reported.** See the saturation section; it changes the reading rather than
merely accompanying it.

---

## What this licenses, per §6

**PARTIAL → report the fraction attributable to pose and stop; the residual needs its own cause.**

The fraction is now measurable rather than assumed. On a 0.42→0.98 gap between "clean" and
"worker", pose moves the pipeline from 0.42 to at most **0.63 even at absurd pose magnitudes** —
roughly **37% of the gap**, with the remaining ~63% unexplained and *not* reachable by more pose.

**What should NOT happen next:** a pose-normalisation project. That is what a CONFIRMED result
would have licensed, and Y1 is not that. Building a worker pose-normaliser would target a cause
that demonstrably cannot close the gap.

**What the residual candidates are** — none tested here, and Y1 deliberately varies none of them:
scan noise, partiality/occlusion, topology defects, and mesh-quality differences specific to
ethanol-preserved worker scans. §7 of the pre-registration stated this scope in advance. The
natural next arm is the same ladder with **scan degradation** as the varied factor instead of pose,
which would say whether the residual is reachable at all.

**Not claimed:** that pose is unimportant. The dose-response is perfectly ordered (ρ = 1.000) and
real. The claim is only that pose is *insufficient* — it explains the clean corpus and about a
third of the worker gap.

## Artifacts

- `y1_generate_corpora.py` — the pose ladder, with checks 5.2/5.3 asserted at generation.
- `y1_shape_space_analysis.py` — LOO gen/spread on rest-space shaped geometry; GT control + fits.
- `submit_Y1_20260830.sbatch`; `corpora/ps*/` (5 × 50 meshes + ground truth); runs at
  `diagnostics/moonshot/runs/Y1_ps*`.
- `out_Y1/y1_results.json`.
- `sbatch_logs/Y1_3350661.log` — gitignored (`*.log`); all numbers are in the JSON.
