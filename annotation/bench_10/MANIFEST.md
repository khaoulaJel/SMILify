# bench_10 — joint-annotation benchmark set

Source: `gdrive:UM6P_2026/DATA/mesh_registration/custom_processing/antscan_processed_issue95`
(388 `*_processed.obj` files). Pulled 2026-08-28 with rclone.

## Selection criteria

One specimen per genus, chosen to span:
- **8 subfamilies** (Paraponerinae, Ponerinae, Proceratiinae, Dorylinae, Myrmicinae,
  Formicinae, Dolichoderinae) rather than sampling the abundance-skewed head of the
  distribution (Pheidole alone is 35/388).
- **Body-plan extremes** that stress the fitter differently: trap-jaw (Odontomachus),
  flattened/armoured (Cephalotes), heavily spined (Acromyrmex), extremely long-legged
  cursorial (Cataglyphis), elongate arboreal (Oecophylla), tiny cryptic short-legged
  (Discothyrea), large-bodied robust (Paraponera), army-ant worker (Eciton),
  dimorphic major (Pheidole megacephala), stout arboreal (Dolichoderus).
- **Mesh size range**: 22k–102k verts / 2.1–10.8 MB (dataset range is 1.3–64 MB).

## Specimens

| # | File | Subfamily | Why included | verts | faces |
|---|------|-----------|--------------|-------|-------|
| 1 | Paraponera_clavata_CASENT0878073 | Paraponerinae | large robust body, thick legs | 50002 | 65189 |
| 2 | Odontomachus_bauri_CASENT0878071 | Ponerinae | trap-jaw head, elongate slender legs | 49938 | 75808 |
| 3 | Discothyrea_patrizii_CASENT0744991 | Proceratiinae | smallest/cryptic, very short legs | 22470 | 45144 |
| 4 | Eciton_burchellii_CASENT0744558 | Dorylinae | army-ant worker, large mandibles | 48352 | 98312 |
| 5 | Cephalotes_atratus_CASENT0744612 | Myrmicinae | dorsoventrally flattened, expanded margins | 48086 | 95923 |
| 6 | Acromyrmex_echinatior_CASENT0745011 | Myrmicinae | dense spination (hard surface snapping) | 49996 | 73190 |
| 7 | Pheidole_megacephala_CASENT0744094 | Myrmicinae | major caste, most abundant genus in set | 48871 | 103836 |
| 8 | Cataglyphis_savignyi_CASENT0745667 | Formicinae | longest legs, thinnest tarsi | 102287 | 245554 |
| 9 | Oecophylla_longinoda_CASENT0744766 | Formicinae | elongate arboreal weaver | 78416 | 179589 |
| 10 | Dolichoderus_bispinosus_CASENT0744451 | Dolichoderinae | stout, sculptured mesosoma | 50000 | 88630 |

## Sanity checks run

- All 10 parse as OBJ; vertex/face counts above.
- No stray/floating geometry: full bbox vs. 1–99th-percentile bbox differs by
  ×1.0–×1.7 on every axis (a detached fragment would show a large ratio).
- Coordinate scale differs per specimen (bbox spans ~150 to ~26000 units) — the
  meshes are **not** in a common unit; normalise before any cross-specimen metric.

## Use

Import one `.obj` into Blender and run `../02_blender_joint_annotator.py`; export
`<specimen_id>_joints.json` back into this directory.
