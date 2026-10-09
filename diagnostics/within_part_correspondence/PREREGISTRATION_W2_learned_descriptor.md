# Pre-registration — W2: does the *learned* descriptor beat plain proximity within a correct part?

**Written and committed BEFORE the run.** Same discipline as W1: bars fixed here are not revised
after seeing a number.

Date: 2026-08-30. Folder: `diagnostics/within_part_correspondence/`.

---

## 1. Why this runs before any new training

W1 closed the *handcrafted* half of the descriptor question: local-PCA eigen-features land near
chance, and anatomy-conditioned normalised part coordinates are **worse** than plain part-local
rigid proximity on every segment. W1 explicitly did **not** close the *learned* half, and set the
bar a learned descriptor must clear.

A learned dense descriptor trained on exactly this corpus **already exists**: the C11 CSE head
(`out_C11_hardneg_20260826/best_model.pt`), a per-point 16-d embedding supervised by true vertex
identity with hard-negative mining. Measuring it on the W1 within-part protocol is **inference
only** — no training, no GPU budget beyond a forward pass.

Running this first is the point. If C11 already clears the W1 bar, a new architecture is not
needed. If it does not, W2 says *by how much it misses*, which is the only honest basis for
deciding whether a new descriptor is worth the spend. Training a network before measuring the one
already on disk would be the same mistake the record has logged five times.

## 2. Protocol — identical to W1, one arm added

Same 12 held-out specimens (corpus tail 3988–3999, inside C11's own `val_frac=0.05` validation
split, so genuinely unseen), same 20 pairs, same 30 parts, same metric:

> error(i) = ‖ B_verts[retrieved j] − B_verts[i] ‖ / seglen_B(s)

New arm **`CSE_C11`**: query embeddings for A's part vertices matched to query embeddings for B's
part vertices by cosine nearest neighbour. Query-to-query, **not** query-to-key: the key table is
per-template-vertex and identical on every specimen, so key-space matching would be trivially
perfect and would measure nothing. Both sides must be computed from each specimen's own posed
geometry, exactly as the other arms are.

## 3. Pre-registered bar

Primary comparison **`CSE_C11` vs `XYZ_RIGID`**, per segment, individually, against the numbers W1
established:

| seg | bar to beat (`XYZ_RIGID`) |
|---|---:|
| `co` | 0.2813 |
| `tr` | 0.0790 |
| `fe` | 0.0993 |
| `ti` | 0.0910 |

- **PASS** — `CSE_C11` median error is lower on **all four** segments individually, each with
  paired sign test **p < 0.05** *and* **≥ 15% relative** reduction. Licenses Phase 2 (stress the
  descriptor under noise/partiality), not a fitter claim.
- **PARTIAL** — direction holds on some but not all four, or on all four but under 15%.
- **FAIL** — no segment improves. A learned descriptor of this family does not beat proximity
  within a correct part, and the "inject semantic identity at the point level" line is constrained
  for learned descriptors as W1 constrained it for handcrafted ones.

`RANDOM`, `LOCAL_GEO`, `PART_FRAME` are re-reported unchanged from W1 as context.

## 4. Mechanism checks — mandatory; the first two void the run

1. **Checkpoint actually loaded (VOIDS the run).** The C11 file stores weights under
   `model_state_dict` — precisely the key that F1 looked up as `model`, matched **nothing** under
   `strict=False`, and produced a believable result from a randomly-initialised network. Required
   here: `load_state_dict(..., strict=True)` with **0 missing and 0 unexpected** keys, **plus** a
   named tensor compared element-wise against the value read straight from the raw file. A
   non-zero-weight check does not substitute — random weights are also non-zero.
2. **C11 retrieval reproduction (VOIDS the run).** Under C11's own training convention (2048
   sampled surface points, full-vocabulary query→key argmax), the held-out median rest-space 3D
   error must reproduce the published **0.02687** to within 20%. This validates that the input
   convention and preprocessing are the ones the network was trained under. If it does not
   reproduce, every W2 number is measuring a mis-fed network and none may be read.
3. **Density control.** The primary run feeds the full 10,235-vertex set, which is ~5× C11's
   training density (2048 points) and could shift PointNet++'s radius-based neighbourhoods. The
   run is therefore repeated with embeddings computed at the 2048-point training density and
   transferred to part vertices by nearest sampled point. **The verdict must not flip.** If it
   does, the density-matched version is the one that counts and the discrepancy is reported as the
   headline, not a footnote.
4. **Chance control.** `CSE_C11` with its embeddings randomly permuted across vertices must land at
   the `RANDOM` floor. If a shuffled embedding still retrieves well, the arm is measuring geometry
   leaking through the pipeline rather than the descriptor.

## 5. What each outcome licenses — fixed now

- **PASS** → Phase 2 only: stress the descriptor under pose variation, morphology variation, scan
  noise, partiality, topology defects. **Not** a fitter claim, and not a green light to train a new
  architecture — C11 passing would mean the existing one suffices.
- **PARTIAL/FAIL** → report the margin. A new descriptor is justified only if the miss is small
  enough that the H1 lesson (retain morphology; do not canonicalise) plausibly accounts for it —
  and that argument must be made explicitly against the measured gap, not asserted.

**Not claimed under any outcome:** anything about real scans, and anything about the fitter.
Retrieval gains have failed to reach the fitter four times (C11, C12, F2, and the mechanism-check
series); a within-part retrieval result is a hypothesis about the fitter, never a result about it.
