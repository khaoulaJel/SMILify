# R11b — the crossing is real, but both methods fail. The foundation-feature route for laterality closes.

Job 3901561, 1m30s, c25g. 4 held-out `synth_power48` specimens, θ swept about one fixed seeded axis,
applied identically to both methods (rendering AND the geometric query).

---

## Result

| θ | DINO | geometric | winner | either ≥70% bar? |
|--:|--:|--:|---|---|
| 0° | 81.5% | **86.4%** | geometry | both |
| 30° | **76.8%** | 76.0% | DINO | both |
| 60° | **59.9%** | 48.7% | DINO | neither |
| 90° | **44.2%** | 31.4% | DINO | neither |
| 180° | 5.5% | 1.8% | — | neither (both far below chance) |

**Consistency check passes**: θ=0° reproduces R11 within noise (DINO 81.5 vs 82.6; geometry 86.4 vs
86.1), confirming the two experiments measure the same thing.

## Verdict — the predicted crossing happens, and it does not help

The hypothesis R11b was built to test was: *geometry inherits a canonical frame and will collapse
without it, while DINO's view-based features will hold.* **The first half is confirmed, the second
half is refuted.**

- The **ordering does reverse at ~30°**, exactly as predicted — geometry falls faster (86.4 → 31.4)
  than DINO (81.5 → 44.2).
- But **DINO does not hold.** It falls below 50% chance by θ=90°. At θ=180° both sit far *below*
  chance (5.5%, 1.8%), which is systematic side-inversion rather than noise — "DINO wins" there is
  arithmetically true and scientifically meaningless.
- **DINO clears the 70% bar only at θ ≤ 30°, which is exactly where geometry is equally good.**

> There is no regime in this sweep where semantic features deliver usable laterality that plain
> geometry cannot. That is the decisive negative R11's pre-registration said would "genuinely close
> the route", and it is now measured rather than assumed.

## The caveat that cuts the other way — stated, not buried

The 8 render cameras are **fixed**, so rotating the specimen forces genuinely non-canonical
viewpoints (an ant seen from below, or inverted). Real pinned-ant scans are roughly consistently
oriented; R7/R8's finding was that the bilateral frame cannot be recovered **from the geometry**,
not that specimens sit at arbitrary 90–180° attitudes.

So the realistic operating regime is probably θ ≈ 0–30° — precisely the region where DINO and
geometry are indistinguishable (86.4/81.5 and 76.0/76.8). The large-θ conditions are harsher than
deployment, and a fairer test of the semantic route would re-render with view selection adapted per
specimen rather than fixed cameras. That is a real remaining gap; it is not, however, a reason to
read the 0–30° region as a win, because there DINO does not beat geometry either.

## What this closes, and what it leaves open

**Closes**: "use pretrained semantic features to supply the laterality geometry cannot" — as a
drop-in, with fixed-view rendering and a linear probe, on this template. Five independent
instances now (W1, W2, T1.1, W5, R11/R11b) of a learned or semantic descriptor failing to beat plain
geometric proximity in this project.

**Leaves open**: (a) adapted/canonicalised view selection rather than fixed cameras; (b) geometry-
aware semantic correspondence (*Telling Left from Right*), which targets exactly this
orientation weakness — though R11b now shows the weakness is severe, not marginal; (c) whether
laterality is needed at all if the fitter is supplied *unlateralized* part identity plus a separate
side decision made once per specimen rather than per vertex.

**(c) is the cheapest and most promising of the three** and follows directly from the numbers: part
identity is recovered far better than side (R11: ~52% over 13 classes vs 82.6% side at θ=0, but side
is what collapses under rotation). A single per-specimen side decision has one binary degree of
freedom instead of ~5000, so it can be made by majority vote over vertices, or by any global cue —
and it converts a per-vertex problem the features fail at into a per-specimen one they may not.

## Limits

4 specimens × 5 angles; one fixed rotation axis (seeded, shared across all conditions); fixed camera
rig; linear probe. Synthetic, shared topology, exact labels — nothing here is measured on real scans.

## Artifacts

`r11b_rotation_sweep.py`, `out_R11/r11b_results.json`, `sbatch_logs/R11b_3901561.log`.
