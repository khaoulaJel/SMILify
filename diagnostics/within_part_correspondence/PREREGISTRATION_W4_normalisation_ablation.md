# Pre-registration — W4: which normalisation destroyed the signal?

**Written and committed BEFORE the run.** Fixed before any W4 number was seen.

Date: 2026-08-30. CPU-only; runs while W3b trains.

---

## 1. The question W1 left underdetermined

W1's headline was that anatomy-conditioned normalised part coordinates (`PART_FRAME`) are *worse*
than plain part-local rigid proximity on every segment (`co` −2.1%, `tr` −29.1%, `fe` −11.6%,
`ti` −46.0%), and the stated lesson was **"normalising a part to canonical shape destroys
correspondence-bearing signal."**

But `PART_FRAME` normalised **two** things at once — the axial extent *L* and the radial scale *R* —
so W1 cannot say which one did the damage, or whether both did. That matters directly: if only one
is harmful, the other is a free, cheap coordinate a future method should keep.

W4 decomposes it. One factor at a time, everything else identical.

## 2. Arms — a 2×2 over what is divided out

All arms use the **same** anatomy-oriented part frame as W1 (axis sign from the parent segment,
θ referenced to the body centroid). They differ *only* in which scale factors are removed:

| arm | axial | radial | note |
|---|---|---|---|
| `PF_none` | absolute | absolute | **mathematically identical to `XYZ_RIGID`** — see §4 |
| `PF_axial` | ÷ *L* | absolute | removes segment-length variation only |
| `PF_radial` | absolute | ÷ *R* | removes thickness variation only |
| `PF_both` | ÷ *L* | ÷ *R* | **= W1's `PART_FRAME`** |
| `PF_axial_only_coord` | ÷ *L*, radial dropped | — | axial coordinate alone |
| `XYZ_RIGID` | — | — | the baseline, computed independently |
| `RANDOM` | — | — | chance floor |

Same 12 held-out specimens (3988–3999), same 20 pairs, same 30 parts, same metric as W1:
error = ‖B[retrieved] − B[true]‖ / seglen_B, ceiling 0.

## 3. Registered predictions — falsifiable, stated before the run

**P1 (the strong one).** If W1's lesson is correct, error is **monotone in how much is normalised**:
`PF_none` ≤ {`PF_axial`, `PF_radial`} ≤ `PF_both` on every segment.
- Confirmed → W1's lesson holds and W4 says which factor costs more.
- **Violated** — i.e. some normalisation *improves* on `PF_none` — → **W1's stated lesson is too
  strong and must be narrowed.** That outcome is reported as a correction to W1, prominently, not
  as a footnote. W1's *endpoint* would stand either way (`PF_both` did lose); only its
  *explanation* would be wrong.

**P2.** Axial normalisation is more harmful than radial, because a segment's length varies more
across specimens than its thickness and carries more of the identifying proportion. Reported;
gates nothing.

## 4. Mechanism checks — the first voids the run

1. **Identity check (VOIDS).** `PF_none` is a rigid change of basis of the part into its own frame,
   so nearest-neighbour in it is *provably* the same as nearest-neighbour in `XYZ_RIGID`'s rigidly
   aligned coordinates. **`PF_none` must equal `XYZ_RIGID` to within floating point on every
   segment.** If it does not, the frame construction or the retrieval indexing is wrong and no W4
   number may be read. This is a genuine proof-backed no-op, not a plausibility check — which is
   why it is the voiding one.
2. **W1 reproduction.** `PF_both` must reproduce W1's published `PART_FRAME` numbers (0.2871 /
   0.1020 / 0.1108 / 0.1329 on `co`/`tr`/`fe`/`ti`) to within 1%. Same code path, same seed, same
   specimens — a discrepancy means something drifted between W1 and now.
3. **Chance control.** `RANDOM` must be the worst arm on every segment.

## 5. What this licenses

W4 is a **decomposition of an existing result**, not a new hypothesis about a method. No outcome
licenses training anything, and no outcome is a fitter claim. Its whole value is telling a future
method which anatomical coordinates are safe to use and which are not — and, if P1 is violated,
correcting a lesson that would otherwise propagate.

**Not claimed:** anything about real scans. Synthetic, exact correspondence, oracle part labels.
