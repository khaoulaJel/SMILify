# bench_17 — joint-annotation benchmark set

Source: `gdrive:UM6P_2026/DATA/mesh_registration/custom_processing/antscan_proofread_castes/worker_ALT`
(757 `*_processed.obj`). Pulled 2026-08-31 with rclone. **Supersedes `../bench_10`**, which was
drawn from `antscan_processed_issue95`.

## Why this source rather than issue95

`worker_ALT`'s 757 filenames match the keys of
`diagnostics/morphometrics/out/registration_error.json` **exactly, 757/757**. It *is* the
corpus every fitted result in `REPORT_MORPHOMETRICS.md` was computed on. Only 4 of the 10
`bench_10` specimens existed in it; the other 6 had no fit to score an annotation against.

## What this set is for, and what that implies about selection

The project's ground truth today is **synthetic, n = 12** (`diagnostics/morphometrics/calib_features.py`:
*"Per-feature R at n=12 has a wide confidence interval (roughly ±0.3 near R=0.6), so these values
are used to form COARSE reliability tiers, never to rank one feature above another."*). Manual
joint annotation is the first **real** ground truth.

The headline metric is `R = corr(fitted, true)` **across specimens**. So the set must **span the
range** of each measurement, not be representative of the corpus — sampling the abundance head
(Pheidole is 39/757) would give a tight cluster and an uninterpretable R. Three axes are spanned
deliberately:

1. **Body plan / morphometric extremes** — the genera that hold the max or min of the indices in
   `out/genus_table.json`.
2. **Leg and waist geometry** — the blocks where reliability collapses against synthetic GT
   (`seg_l_2_ti` R = **−0.02**, `seg_l_3_ta` 0.06, `seg_l_1_ta` 0.10, vs `head_len` 0.96), and
   which §2.3 of the report currently drops wholesale. bench_10 left these axes unpinned.
3. **Registration quality** — all four bench_10 specimens that had fits sat in the *worst
   quartile* (77th–94th pct). R computed only on the tail measures the pipeline where it is
   worst. bench_17 spans **0.0th to 93.8th percentile**.

## Specimens

| # | File | Subfamily | Role | reg-err pct |
|---|------|-----------|------|-------------|
| **Block A — body-plan extremes (10, continuous with bench_10)** ||||
| 1 | Acromyrmex_echinatior_CASENT0745011 | Myrmicinae | dense spination; hard surface snapping (same specimen as bench_10) | 77.1% |
| 2 | Cephalotes_atratus_CASENT0744612 | Myrmicinae | flattened/armoured; **cephalic_index max 1.32, head_flatness min 0.56, scape_index min 0.72** (same specimen) | 91.5% |
| 3 | Discothyrea_patrizii_CASENT0744991 | Proceratiinae | smallest/cryptic, very short legs (same specimen) | 86.0% |
| 4 | Eciton_burchellii_CASENT0744558 | Dorylinae | army ant; **scape_index max 1.59, foreleg_hindleg max 1.06** (same specimen) | 93.8% |
| 5 | Odontomachus_bauri_CASENT0878072 | Ponerinae | trap-jaw; **cephalic_index min 0.68, mandible_slenderness max 5.36, mesosoma_slenderness max 2.83**. bench_10's `CASENT0878071` is absent; this is the sibling accession of the same species | 11.9% |
| 6 | Pheidole_megacephala_CASENT0744847 | Myrmicinae | dimorphic major, most abundant genus. bench_10's `CASENT0744094` absent; best-fitting *megacephala* here | 31.4% |
| 7 | Dolichoderus_cf.bidens_CASENT0744033 | Dolichoderinae | stout sculptured mesosoma. bench_10's *bispinosus* absent; *bidens* is its congener in the same species group | 42.8% |
| 8 | Cataglyphis_iberica_CASENT0745158 | Formicinae | long-legged cursorial; the genus's **only** specimen in the corpus | 30.0% |
| 9 | Tetraponera_hespera_CASENT0840869-D4 | **Pseudomyrmecinae** | elongate arboreal, extremely slender. Replaces *Oecophylla* (absent, n=0) and adds a subfamily bench_10 lacked | 12.2% |
| 10 | Streblognathus_peetersi_CASENT0744989 | Ponerinae | large-bodied robust. Replaces *Paraponera* (absent, n=0) | 9.0% |
| **Block B — leg & waist extremes (4, new)** ||||
| 11 | Aphaenogaster_gracillima_OKENT0105348 | Myrmicinae | **head_flatness max 0.99, petiole_index max 0.34, waist_constriction max 0.50**; gracile elongate waist | 24.7% |
| 12 | Formica_cf.fusca_CASENT0878139 | Formicinae | **petiole_index min 0.13, waist_constriction min 0.15** — the opposite pole to #11 | 22.7% |
| 13 | Crematogaster_evallans_CASENT0744252 | Myrmicinae | **tibia_femur max 0.82, hindfemur_index min 0.80**; heart-shaped gaster | 38.3% |
| 14 | Cyphomyrmex_major_CASENT0745147 | Myrmicinae | **hindfemur_index max 1.42**; genus fits poorly throughout (genus median ~70th pct) | 68.4% |
| **Block C — clean-fit anchors (3, new)** ||||
| 15 | Gigantiops_destructor_CASENT0745031 | Formicinae | **best-fitting specimen in the entire 757 corpus**; huge-eyed, long-legged | 0.0% |
| 16 | Leptogenys_peuqueti_CASENT0741327 | Ponerinae | **gaster_slenderness max 3.21, head_mesosoma min 0.65**, and a clean fit — extreme *and* easy | 1.1% |
| 17 | Aenictus_ceylonicus_CASENT0878084 | Dorylinae | **mandible_index min 0.49, gaster_mesosoma min 1.12**; clean fit | 0.4% |

9 subfamilies (Myrmicinae, Ponerinae, Formicinae, Dorylinae, Dolichoderinae, Proceratiinae,
Pseudomyrmecinae — bench_10's Paraponerinae is not represented in this corpus at all).
10 of the 17 genera are in the **38 genera shared with `ALL_ANTS_CLEAN`**, so their annotations
also feed the cross-corpus replication of §5.

## Sanity checks run (2026-08-31)

- All 17 parse as OBJ. 19.5k–49.9k verts, 40k–106k faces, 1.9–4.9 MB. `worker_ALT` meshes are
  far more uniform than issue95's (22k–102k verts) — this processing appears decimated to a
  common budget.
- No stray/floating geometry: full bbox vs 1st–99th-percentile bbox ratio is **1.08×–1.67×** on
  every axis, matching bench_10's 1.0–1.7 range. A detached fragment would show a large ratio.
- **Coordinate scale is arbitrary per specimen** (bbox span 292 to 14,926 units). Normalise
  before any cross-specimen metric. Confirms REPORT_MORPHOMETRICS §2: absolute size is not
  recoverable, only proportions.
- Registration-error percentiles span 0.0% to 93.8%; 6 specimens below the 25th, 5 above the 60th.

## Annotation guidance

- Import one `.obj` into Blender and run `../02_blender_joint_annotator.py`; export
  `<specimen_id>_joints.json` back into this directory. Use the **filename stem** as specimen ID
  so the join back to `registration_error.json` and the fit outputs is exact.
- **Never guess a tarsus (`*_ta`) or pretarsus (`*_pt`) position.** Use the
  `occluded_estimated` / `not_present` tags. Those joints are precisely what the annotation is
  meant to adjudicate (current R 0.05–0.28), so a guessed point would manufacture the signal
  being tested.
- Suggested order: Block C first (clean fits, easiest to learn the tool on), then Block A,
  then Block B.

## Provenance

File list: `_files.txt`. Reproduce with:

```
rclone copy --files-from annotation/bench_17/_files.txt \
  "gdrive:UM6P_2026/DATA/mesh_registration/custom_processing/antscan_proofread_castes/worker_ALT/" \
  annotation/bench_17/
```
