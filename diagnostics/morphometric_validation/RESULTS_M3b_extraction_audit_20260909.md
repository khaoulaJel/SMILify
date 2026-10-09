# M3b — real-corpus extraction transfer audit: **PASS**, with one named failure mode

Run 2026-09-09 against `PREREGISTRATION_M3b_extraction_audit.md`. Bars were fixed before the fit
was submitted and are not revised here. **The phenotype panel is unchanged** — this audit had no
licence to change it and did not.

## Provenance

The 757-corpus fits do **not** exist on this cluster, and — checked directly — are **not on the
mounted Google Drive either**: the remote holds 5453 files (3229 `.obj`, 785 `.stl`, 816 `.json`,
456 `.log`, papers, scripts) and **zero `.npz`**; no `Stage_3*` file anywhere on it. The JUWELS
run's outputs were never synced. This audit therefore fitted its own 40 specimens under the
unmodified `D1_PROD.yaml` recipe (`submit_audit40.sbatch` → `run_m1_fit_all.sh`, paths overridden,
no recipe edits).

**Cost fact for M4**: 40 specimens fitted in **11m27s** on one c25g GPU ⇒ the full 757 is
**≈3.6 GPU-hours**, not the 80 minutes first estimated. M4 must budget its own fitting run.

Sample: 40 specimens, `random.Random(20260909)`, 35 genera, list frozen in
`audit40_specimens.txt` before submission.

## Verdict

| check | result | bar | |
|---|--:|--:|:--|
| C1 extraction completeness | 40/40 | 40 | PASS |
| C2 ratio plausibility | 40/40 | 38 | PASS |
| C3 antero-posterior ordering | 37/40 | 36 | PASS |
| C4 head straddles midline | 40/40 | 38 | PASS |
| C5 distributional transfer | gap 0.166 / 0.135 | no gap ≥0.20 with ≥10% minority | PASS |

**OVERALL: PASS.** The extraction pipeline transfers from 20 conspecific *Atta* to a 35-genus real
draw. M4 may proceed on the frozen panel.

Distributions: HW/WL median 0.646 [0.488, 1.252]; HL/WL 0.583 [0.399, 1.002]; head lateral ratio
1.010 [0.895, 1.089] — the head straddles its midline essentially perfectly corpus-wide, which is
the cleanest single number in this audit.

## The failure mode, and why it matters more than the pass

All three C3 failures are the **same** failure, and it is not the one the check was written to
catch. In each, the head's *anterior* landmark (`clypeal_margin_ant_mid`) sits **posterior** to its
*posterior* landmark (`cephalic_margin_post_mid`) — the head's antero-posterior axis is inverted:

| specimen | violated step(s) | HW/WL | HL/WL | lat ratio |
|---|---|--:|--:|--:|
| `Atta_sexdens_CASENT0744597` | clypeal→cephalic | 0.410 | 0.381 | 1.009 |
| `Cataulacus_sp._CASENT0745593` | clypeal→cephalic | 1.038 | 0.815 | 0.993 |
| `Tetramorium_sp._OKENT0105210` | clypeal→cephalic, cephalic→petiole | 0.970 | 0.841 | 1.031 |

Landmark indices are **fixed template vertices**, so an ordering violation cannot be a
mis-indexing: it means the *fit* placed the head backwards. `out/m3b_C6_flagged.png` confirms it
visually — in passing specimens the clypeal marker sits well anterior of the cephalic one; in the
flagged ones the two have nearly collapsed together and swapped, and in `Tetramorium` the petiole
has folded forward into the head region.

**The consequence for the panel is asymmetric, and this is the audit's real finding:**

- **HW is unaffected** (C2 and C4 pass on all three). HW is a Type III part extent, computed after
  rigid alignment of the head to the template frame — it does not depend on landmark AP ordering.
- **HL is directly compromised**, because HL *is* the distance between exactly the two landmarks
  that inverted.

So the two primary traits do **not** share a failure mode on the real corpus: at this draw,
**HW is robust at 40/40 and HL fails on ~7.5%**. Both cleared M3 on *Atta*; only the real corpus
separates them. This is a transfer result the 20-specimen validation could not have produced.

## C6 — failure attribution: partially answered, and stated as such

The per-specimen antscan metadata (found on the Drive during this audit, previously unused —
`voxel_size`, `processed_hole_count`, `processed_mesh_smoothness`, `processed_face_size_cov`)
covers only **16 of the 40** audit specimens and only **1 of the 3** flagged. This is **not** a
download gap — it is a permanent coverage limit: the Drive holds 785 `antscan_data` folders and all
784 JSONs were retrieved, but only **279 of the 757** `worker_ALT` specimens have a folder at all
(478 have none). The scan-pathology question is therefore **underpowered and NOT answered**, and
cannot be fully answered from this source at any sample size.

What can be said: the one flagged specimen with metadata (`Tetramorium_sp._OKENT0105210`) is
scan-quality **indistinguishable** from the passing set — holes 2176 vs 2929 median, smoothness
28.26 vs 28.31, face-size CoV 1.20 vs 1.20. Combined with the renders (heads present and
well-formed, merely mis-oriented), the working classification is **fit failure, not scan pathology
and not extraction bug** — on n=1 for the metadata axis. M4's failure map (§5) must therefore be
built without a scan-quality covariate for ~63% of the corpus, or use an intrinsic mesh-quality
measure computed from the meshes themselves.

## What this does not establish

No accuracy claim: `worker_ALT` has no ground truth, no expert landmarks, no physical measurements.
Every check here is internal-consistency or plausibility. Nothing here licenses a panel change, and
none was made.
