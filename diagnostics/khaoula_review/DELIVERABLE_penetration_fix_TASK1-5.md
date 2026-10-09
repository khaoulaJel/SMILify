# Penetration/registration fix — TASK 1-5 deliverable

> **Looking for the standing recommendation or what shipped?** Read
> `PENETRATION_LOSS_SUMMARY.md` first — it's the consolidated finding across this doc and
> TASK6/7/8. This file (and TASK6/7/8) is the evidence trail behind it, not the deliverable
> itself.

bench50_clean (50 specimens), gentle proximity penetration loss (0.02/0.05) + `scheme:'all'`
vs `w_penetration=0` baseline. Full detail in the five linked docs; this is the summary.

## TASK 1 — does gentle + scheme:'all' help? (`TASK1_gentle_penetration_scheme_all_bench50.md`)

Pure re-analysis of existing runs, re-scored under the Hard Rule. On gaster-legs (the pair
called out as anatomically dominant): **yes**, count -20.8%, consistent across 3 seeds, no
collapse. On all 10 non-adjacent pairs the loss actually trains on: **no** — aggregate
+16.8% worse, the gaster-legs win is outweighed by new shallow contact elsewhere (mandible/
antenna/head vs. legs/thorax). Both true at once — contact gets shallower on the targeted
pair, more widespread everywhere else.

## TASK 2 — does a structural constraint (w_offset) fix it? (`TASK2_structural_constraint_offset_penalty.md`)

Same config + w_offset (5.0/2.0). Preserves the gaster-legs win (-20.2%, within noise of
TASK1) and **more than halves** the aggregate cost (+16.8% → +5.4%), at the cost of a worse
single-specimen F-score outlier (-0.1125 vs -0.0596). Real, partial fix — not a full one.

## TASK 3 — GWN disagreement diagnostic, shipped + verified (`TASK3_gwn_disagreement_baseline_vs_gentle.md`)

Diagnostic shipped (`diagnostics/moonshot/eval_gwn_disagreement.py`), independently
recomputed from raw per-specimen CSVs (matches logs to <5e-5). An unrelated ground-truth
metric (GWN inside/outside) **corroborates TASK1's finding from a second code path**:
13/15 non-trivial pair/directions disagree with ground truth more under gentle, concentrated
in the same `A_into_B` direction and the same pairs (mandible/antenna/head-thorax) where
TASK1's own count table showed the largest blowups. Confirms this is a real degradation in
signal quality, not a training-loss-specific artifact — and confirms (FINDINGS.md Step 6's
open question) that freeing pose via `scheme:'all'` does **not** resolve the underlying
proximity-test accuracy asymmetry, even though it does change the count outcome.

## TASK 4 — pair-scoped loss: does restricting training to gaster-legs alone make TASK 2's
partial fix a clean one? (`TASK4_pairscoped_gentle_penetration.md`)

Pre-registered follow-up, not the original spec's TASK 4 (torch-mesh-isect, still untried).
Same config as TASK 2 plus a new `penetration_train_pairs` option (`fitter_3d/trainer.py`)
restricting the w_penetration loss to gaster-legs only; evaluation still scores all 10 pairs.
Mixed result, not the clean win the hypothesis predicted: the all-pairs aggregate does turn
genuinely positive for the first time (+5.4% → **-13.6%**, PASS), but 2 of 4 pre-registered
criteria fail — mean F-score cost roughly quintuples (-0.0032 → **-0.0170**) and integrity
cost roughly quadruples (deform_mag/edge_logratio ~+6% → **~+21%**) — and the gaster-legs win
itself gets *weaker*, not stronger, when trained alone (-20.2% → **-16.8%**). A same-arm seed
check shows the effect size isn't stable to better than ~±10pp at this sample size. No
collapse in outlier renders. Reads as a squeezed-balloon effect: removing gradient pressure
from the other 9 pairs does eliminate their collateral damage, but concentrates deformation
(and its cost) onto gaster-legs rather than genuinely resolving more of it.

## TASK 5 — add Fabian's scale/trans barriers on top of TASK 4 (`TASK5_pairscoped_offset_scalecap.md`)

One pre-registered bench50 arm: TASK4's pair-scoped config plus `w_scale=0.052` (the one
validated value in the branch, T0.4, previously shipped only on the moonshot/hierarchical
pipeline, never run on this one) and `w_trans=1.0` (the only value ever tried anywhere,
explicitly uncalibrated). Newly wired into `fitter_3d/trainer.py`, ported verbatim from
`trainer_moonshot.py`. Result: **2 of 4 pre-registered criteria still fail** (mean F-score
−0.0195, worst −0.0890, both worse than the −0.01/−0.08 bar) — same two criteria TASK4
failed. The barriers did help on two axes that weren't required to pass: all-pairs aggregate
strengthened further (−13.6% → **−16.9%**, the best net win yet) and edge-distortion
integrity improved sharply (+21.1% → **+4.9%**, a 4.3x reduction) — but did not fix
mean/worst F-score, which is the criterion every arm across TASK1-5 has failed. No collapse.

## TASK 6 — GWN-based matching primitive: causal test, not another diagnostic (`TASK6_gwn_matching_primitive_causal_test.md`)

Answers what TASK3/FINDINGS.md Step 6 explicitly left open: does replacing the proximity test's
matching primitive with a GWN-based one actually fix the accuracy asymmetry, not just disagree
with it more visibly. 3 specimens (Acanthostichus, Acromyrmex — strongest original asymmetry
signal, Solenopsis — no-signal contrast case), gaster-legs-only GWN loss, `w_penetration=0`,
recovered from a JURECA cluster run (`fit3d_results_gwn_gentle_calibrated_longstage3/`, extended
to 3000 Stage_3 iterations after a 1000-iteration checkpoint looked still-declining rather than
plateaued). Result: **not a clean fix**. Only Acanthostichus wins on both count (−27.3% vs.
baseline) and F-score. Acromyrmex — the specimen the mechanism was expected to help most —
instead takes the worst F-score hit of the three (−0.078 vs. proximity) while its count is
*worse* than the existing proximity loss, not better. Solenopsis, the contrast case, gets
meaningfully worse on count under GWN (+22.4% vs. proximity) with no offsetting benefit. n=3, no
seeds — below this project's own established bar, not a final verdict — but a second real,
evidenced result (alongside the torch-mesh-isect addendum below) leaning against either
candidate matching-primitive fix being a clean win.

## TASK 7 — symmetric chamfer sampling: a SEPARATE, decoupled fix, and a partial win (`TASK7_symmetric_chamfer_sampling.md`)

Tests the OTHER of FINDINGS.md step 3's two independent confirmed causes — chamfer
vertex-density asymmetry (legs 2.36x denser than gaster) — via a new opt-in
`symmetric_chamfer_sampling` flag (`fitter_3d/trainer.py`, default `False`), porting
`trainer_moonshot.py`'s already-working area-weighted-both-sides chamfer sampling. Entirely
decoupled from penetration/matching-primitive (TASK6's subject) — a chamfer-term fix, not a
penetration-term one. Same isolation discipline throughout (TASK5's config held fixed, `w_sdf=0`
explicit). Progression: n=3 (same specimens as TASK6, not a clean fix, same shape as GWN's
result) → n=11 curated (deliberately sampled near known outcomes + untested corners in a
(density-ratio, baseline-concentration) map; the map's own "gate" hypothesis was directly
falsified by replication at n=11 — **though that falsification later needed a correction: one
of its two anchor points, Strumigenys_sp.appretiata's dramatic single-run win, turned out to be
seed noise once retested, not a stable result — see TASK7's own doc for the full retraction,
stated as prominently there as the original claim was**) → **n=10x3-seed, randomly sampled,
same discipline as FINDINGS.md's own reseeded rerun**: mean F-score improves robustly in every
seed (+0.028 to +0.058) but aggregate count does not reliably improve (mean +1.3%, sign-flipping
−7.8% to +11.0% across seeds) — does not clear TASK5's simultaneous bar. A short, zero-new-compute
follow-up (per-vertex penetration-classification flips between seeds, using already-saved fitted
vertices) found the near-tie/boundary mechanism real in 7/10 specimens but insufficient alone to
explain aggregate-count stability — and replicates a pattern FINDINGS.md's reseeded rerun already
found for the unrelated gentle-proximity arm, i.e. this looks like a property of the count metric
and fitting stochasticity generally, not something specific to any one loss term.
**Recommendation: ship the F-score result on its own terms now** — `symmetric_chamfer_sampling`
is a real, seed-validated, independently useful surface-fit improvement that doesn't need the
penetration-count question resolved first; it's available today as a zero-effect-by-default
opt-in flag for anyone who wants it.

## Verdict

No arm produces an unconditional win. Gentle + scheme:'all' alone trades one pair's
improvement for broader, GWN-confirmed degradation elsewhere (TASK1, TASK3). Adding
w_offset (TASK2) is a real, partial fix: keeps the gaster-legs win, cuts the aggregate
cost by more than half, but does not eliminate it and worsens the worst-case single-specimen
F-score. Restricting training scope further (TASK4) turns the aggregate net-positive but at
a larger, not smaller, per-pair fit/integrity cost and a weaker own-pair win. Adding the
scale/trans barriers on top (TASK5) improves the aggregate and edge-distortion further but
still does not clear the F-score bar. Replacing the matching primitive itself with a GWN-based
signal (TASK6), the fix Step 6 originally motivated, does not resolve the asymmetry cleanly
either — real wins on some specimens, real new costs on others, no consistent direction.
Symmetric chamfer sampling (TASK7) — targeting the OTHER of Step 3's two confirmed causes,
decoupled from penetration entirely — improves F-score robustly at seeded scale but does not
reliably move penetration count either, and a short follow-up found that count's seed-instability
is not specific to any one loss term (replicated across TASK7's fix and the original
gentle-proximity arm), pointing at the metric/fitting stochasticity itself.
**Standing recommendation from `REVIEW_penetration_update.md` — ship as an available config
option, not a default, gated by a required per-specimen check — holds for penetration-count
reduction specifically.** No arm across TASK 1-7 clears all Hard-Rule-style criteria at once on
the count axis; the mean/worst F-score bar in particular has failed on every soft-loss arm tried
whose PURPOSE was reducing count, GWN-trained ones included. Per the pre-registered decision tree
for TASK 5: treat soft proximity as optional and gated, not the solution; default stays
D1/no-penetration for the penetration-loss question. **Carve-out: TASK7's `symmetric_chamfer_sampling`
is a separate claim about the chamfer term, not the penetration term — it clears its own,
narrower bar (seed-stable F-score improvement) and is recommended as a standalone opt-in now,
independent of whether or when the count question resolves.** Part-filtered conical/BVH loss
(the original spec's TASK 4, and TASK5's own step 3) remains untried as a training signal and,
per the addendum below, is not currently a viable near-term alternative either — its reference
implementation does not run correctly on this project's environment, independent of the
product-decision gate it was already behind.

## Addendum (2026-08-24) — torch-mesh-isect compat probe: kernel non-functional on current stack

Before any further attempt to wire the BVH/conical loss into `trainer.py`, ran a local
compile-and-run smoke test (not import-only) against this project's actual environment:
`diagnostics/torch_mesh_isect_compat_PROBE.sh` / `torch_mesh_isect_compat_probe.py`, output
preserved at `diagnostics/torch_mesh_isect_compat_PROBE_out.txt`. Result: the maintained fork
(`EthanFifle/torch-mesh-intersection`) **compiles cleanly** after one local packaging fix (its
`setup.py` passed a relative `include` dir that silently resolved to nothing once torch's
ninja backend runs from the build temp dir instead of the source dir — patched to an absolute
path in the vendored copy under `custom_processing/external/torch-mesh-intersection/setup.py`).
But the compiled `bvh_cuda` extension **does not detect intersection at all** on this box: zero
collisions reported for a boundary-touching pair, for a hand-verified interior-piercing pair
(crossing segment computed analytically, strictly inside both triangles), and — decisively —
for two **exactly coincident** triangles (identical vertices, maximal possible overlap). The
last case rules out any geometric ambiguity; the kernel is non-functional as compiled here, not
just imprecise. Root cause not pursued further (out of scope for a compat check): this is an
~2019 CUDA kernel, and this box's installed stack (torch 2.5.1+cu124) is a larger gap from the
fork's own tested baseline (torch 2.0/CUDA 11.7) than this project's pinned `environment.yml`
(torch 2.3.1/CUDA 11.8) — noted as a plausible but **untested** confound, not chased further
since debugging an unmaintained, multiple-CUDA-generations-old kernel is an open-ended task this
decision was never scoped to absorb. **Standing status unchanged but now evidenced rather than
merely deprioritized**: part-filtered conical/BVH loss stays untried as a training signal, and
is not currently a viable near-term alternative to the GWN-based matching primitive without
separate, open-ended kernel-debugging work.

**Update (2026-08-24, later): a different fork works.** Investigating "Instant Self-Intersection
Repair for 3D Meshes" (SIGGRAPH 2025, `wonjongg/instant-mesh-intersection-repair` -- a post-hoc
mesh repair approach, not a training-time loss, potentially sidestepping this whole
training-instability thread) surfaced that it depends on its own maintained
`wonjongg/torch-mesh-isect` fork, distinct from the `EthanFifle` fork tested above. Same
compile-and-run rigor applied: after the same packaging fix (absolute include path) plus
bundling `helper_math.h` (one file, reused from the already-built EthanFifle fork's copy, since
this fork doesn't bundle it and still references `$CUDA_SAMPLES_INC`), it **compiles clean and
correctly detects real, non-degenerate intersections** -- a genuine interior-piercing pair
(PASS), correct absence when separated (PASS), and 6 triangles with 2 simultaneous real
colliding pairs, exact match, no false positives (PASS). It still fails ONLY the fully-degenerate
exact-coincident-triangle case (zero volume, identical position/orientation) -- a known blind
spot for face-orientation-dependent algorithms generally, and not a configuration that occurs in
a fitted SMIL mesh (fixed, non-degenerate template topology). **This reverses the standing
"non-functional on this stack" finding for torch-mesh-isect specifically** -- it was fork-
specific, not a property of the kernel/architecture/CUDA-version gap as first suspected. The BVH
detection primitive itself (`bvh_cuda`) is now confirmed working on this environment; wiring it
into an actual loss or repair pipeline (as either this trainer's penetration term or via the
SIGGRAPH paper's own post-hoc `repair_factory.py`) is a further step, not done here.

> **CORRECTION (2026-08-24, later still) — that "further step" was attempted and failed
> differently than expected. Read before citing the paragraph above as "BVH is ready to use".**
> `fitter_3d/bvh_penetration_loss.py` wired this in as an actual `w_penetration_bvh` training
> signal (TASK8, `TASK8_bvh_penetration_attempt.md`). Small-scale smoke tests and the weight
> calibration probe ran clean. The real 10-specimen seeded experiment **crashed with a CUDA
> illegal memory access on the very first real training iteration** -- not a scale problem
> (standalone detection at full 10-specimen scale was fine), not an iteration-0/ramp problem
> (the identical loss call in isolation at iteration 0 was fine), specific to combining this
> loss with the rest of a real training step's other active losses in one graph. GPU state was
> confirmed healthy afterward; not investigated further, per this project's "don't debug an
> unfamiliar kernel open-endedly" discipline. **The detection primitive works in isolation; it
> does not currently work as part of a real training step** -- a materially more serious finding
> than the packaging issue this update originally reported, and the reason this correction is
> written with the same weight as the claim it corrects, not appended quietly.
