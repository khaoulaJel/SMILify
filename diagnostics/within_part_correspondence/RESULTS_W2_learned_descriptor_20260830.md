# W2 — the *learned* descriptor also FAILS the bar. It does not beat a rigid transform.

Bar fixed in `PREREGISTRATION_W2_learned_descriptor.md` **before** the run. Inference only — no
training. Frozen C11 CSE head, 12 held-out specimens (corpus tail 3988–3999, inside C11's own
`val_frac=0.05` validation split), 20 pairs, 30 parts, W1's protocol imported wholesale.

---

## Why this ran before funding any new architecture

W1 closed the handcrafted half of the descriptor question and explicitly left the learned half
open. A learned dense descriptor trained on exactly this corpus already existed — C11, a 16-d
per-point embedding supervised by true vertex identity with hard-negative mining. Measuring it is a
forward pass. Training a new network before measuring the one on disk would have repeated a pattern
the record has logged five times.

## Endpoint: FAIL on all four bar segments

Median normalised 3D within-part retrieval error, lower better, ceiling = 0:

| seg | RANDOM | **XYZ_RIGID** | PART_FRAME | CSE_C11 (full-vertex) | **CSE_C11 (density-matched)** |
|---|---:|---:|---:|---:|---:|
| `co` | 0.4054 | **0.2768** | — | 0.4156 | 0.2908 |
| `tr` | 0.3468 | **0.0778** | — | 0.2614 | 0.0998 |
| `fe` | 0.3209 | **0.0958** | — | 0.2832 | 0.1218 |
| `ti` | 0.3183 | **0.0968** | — | 0.4131 | 0.1858 |
| `ta` | 0.3108 | **0.1568** | — | 0.4326 | 0.2755 |

Against the bar, on the density-matched arm — the one the pre-registration designates as counting:

| seg | XYZ_RIGID | CSE_C11 (density) | rel. change | better | sign p | verdict |
|---|---:|---:|---:|---:|---:|---|
| `co` | 0.2768 | 0.2908 | −5.0% | 56/120 | 0.582 | no (**tie**) |
| `tr` | 0.0778 | 0.0998 | **−28.2%** | 25/120 | 8.4e-11 | no |
| `fe` | 0.0958 | 0.1218 | **−27.1%** | 32/120 | 3.2e-07 | no |
| `ti` | 0.0968 | 0.1858 | **−92.0%** | 6/120 | 5.8e-27 | no |

**0 of 4 pass.** On `tr`/`fe`/`ti` the learned descriptor is significantly *worse* than a rigid
transform. On `co` it is a statistical tie (p = 0.58) — the one segment where it is not beaten, and
notably the segment that has resisted every mechanism in this investigation.

## The finding worth carrying

At correct density, `CSE_C11` lands essentially **where `PART_FRAME` landed in W1** (`tr` 0.0998 vs
0.0933; `fe` 0.1218 vs 0.0956). A 16-dimensional embedding trained on 4000 ants with exact
true-vertex supervision and hard-negative mining performs about as well at this task as a
hand-designed normalised part coordinate — and both lose to plain part-local rigid proximity.

Taken with W1, three descriptor families now sit on the same side of the same line:

| family | result vs `XYZ_RIGID` |
|---|---|
| handcrafted local geometry (`LOCAL_GEO`) | near chance |
| anatomy-conditioned normalised coordinates (`PART_FRAME`) | worse on every segment |
| learned contrastive embedding (`CSE_C11`) | worse on 3/4, tie on `co` |

`XYZ_RIGID` differs from all three in one respect: it uses the part's **actual placement** — where
the geometry sits — rather than trying to identify a point from its appearance. The consistent
reading is that what identifies a point within a correct segment is largely *where it is*, not
*what it looks like*. That is a statement about configuration, not about descriptors.

## Mechanism checks — all four, and two of them changed the reading

**Check 1 — checkpoint actually loaded: PASS.** `strict=True`, 0 missing / 0 unexpected, plus
`embed_head.0.weight` compared element-wise against the value read from the raw file. The C11 file
stores weights under `model_state_dict`, the exact key F1 looked up as `model` — matching nothing
under `strict=False` while a randomly-initialised network produced a believable result.

**Check 2 — C11 retrieval reproduces: PASS.** Under C11's own training convention, held-out median
rest-space 3D error **0.02785 vs the published 0.02687 — 3.6% deviation** (bar: 20%). This is what
licenses reading the rest of the run: the network is loaded *and* fed the way it was trained.

**Check 3 — density control: it invalidated my own primary arm, as registered.** The full-vertex
arm (10,235 points, ≈5× C11's training density) collapses to at-or-below chance on `co`/`ti`/`ta`.
That is **not** a descriptor result — check 2 passes at 2048 points, so the network is fine and the
full-vertex input is simply out of distribution for PointNet++'s radius-based neighbourhoods. The
pre-registration fixed in advance that the density-matched arm is the one that counts and that the
discrepancy is reported as the headline rather than a footnote. Both arms FAIL, so the verdict does
not flip; but the *margins* differ by a factor of ~3 and only the density-matched ones are real.

**Check 4 — chance control: initially BROKEN, corrected, then PASS.** The first implementation
applied **one shared permutation to both specimens**, which leaves vertex `perm[i]` on A still
paired with `perm[i]` on B — it does not destroy correspondence at all — and it ran on the
out-of-distribution full-vertex embeddings, so it tested neither thing it was meant to. Corrected
to an **independent permutation per specimen on the density-matched embeddings**, it now lands
exactly at the `RANDOM` floor:

| seg | SHUFFLED | RANDOM | ratio |
|---|---:|---:|---:|
| `co` | 0.4054 | 0.4054 | 1.000 |
| `tr` | 0.3423 | 0.3468 | 0.987 |
| `fe` | 0.3267 | 0.3209 | 1.018 |
| `ti` | 0.3075 | 0.3183 | 0.966 |
| `ta` | 0.3127 | 0.3108 | 1.006 |

Ratios within 3.4% of 1.0 on every segment. The correction did not touch the endpoint — it was
applied to a control arm that gates nothing — and the endpoint verdict is identical before and
after.

---

## What this licenses, stated against the pre-registration

**FAIL** → per the fixed reading, a new descriptor is justified only if the miss is small enough
that W1's lesson (retain morphology; do not canonicalise) plausibly accounts for it, *and that
argument is made explicitly against the measured gap*. The gaps are −28%, −27%, −92%. They are not
small, and W1's lesson does not obviously explain a 92% miss on `ti`.

**The honest caveat, stated rather than buried:** C11 was trained for a *different* objective —
full-vocabulary retrieval against a fixed per-template-vertex key table — not for discriminating
points within a part between two posed specimens. W2 therefore does **not** prove no learned
descriptor can clear the bar. What it removes is the argument that motivated Phase 1. "Handcrafted
failed, so learn it" is no longer available as a reason: the nearest available learned descriptor,
trained on this exact corpus with exact supervision, is worse than a rigid transform. A proposal to
train a new one now has to explain specifically why a different objective would beat
**0.0778 / 0.0958 / 0.0968** when C11's does not come close.

**Not claimed:** anything about real scans, and anything about the fitter. Synthetic, exact
correspondence by construction, oracle part label on both sides, no noise or partiality. Retrieval
gains have failed to reach the fitter four times.

## Artifacts

- `w2_learned_descriptor.py` — the experiment; imports W1's protocol so nothing is re-implemented.
- `submit_W2_20260830.sbatch` — GPU job (`c23g`, account `rwth2151`).
- `out_W2/w2_results.json` — checkpoint report, C11 reproduction, both bars, all four checks.
- `out_W2/w2_rows.csv` — 4,200 rows (pair × part × arm).
- `sbatch_logs/W2_3336615.log` (original), `W2_3336691.log` (corrected controls) — on disk; `*.log` is gitignored repo-wide (`.gitignore:153`), so these are not in the commit. All numbers they contain are in `w2_results.json`.
