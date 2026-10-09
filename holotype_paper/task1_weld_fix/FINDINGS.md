# Task 1 findings log (append-only, dated)

## 2026-10-09 00:10 -- smoke specimen (Acromyrmex_cf.coronatus_CASENT0744044, raw STL 101 MB)

- All three versions exit 0 (V0 113 s, V1 209 s, V2 246 s). Job 4884434 (default/c23ms, approved
  scoped exception) continues with the other 39 specimens.
- **Mechanism visible on a real raw scan.** V2's diagnostic measures the smallest gap between
  topologically distant parts of the surface: `min_self_approach_gap = 1.03`. V0 welds at a fixed
  0.2% of the bbox = **2.87** (2.8x the gap -> vertices across the gap get merged = surfaces fused).
  V1/V2 derive a local-density threshold from the median edge length (3.28 -> **0.98**) and use the
  smaller one; V2's post-weld check reports the gap `preserved`; weld face loss 0.7% (438,609 ->
  435,573), far below the 20% abort.
- **Robustness concern:** V1's threshold is only ~5% below the gap here (0.98 vs 1.03). Added check
  for the full run: per specimen, gap vs V0 threshold vs V1 threshold (from V2/V1 logs); how often
  V0 > gap (fusion expected) and how often V1 > gap (fix insufficient), with the margin distribution.
- V1/V2 also bridged 1 severed island (44 vertices, gap 0.00, 1 hop) and removed 29,269 debris islands
  (< 4 faces) before welding; V0 keeps only the largest component (what it discards is measured by
  coverage in t1_measure.py).

## 2026-10-09 -- pre-push verification of what is being tested and what will be pushed

- Run files are byte-identical to git: V0 == `upstream/master`, V1 == `origin/fix/apply-modifiers-weld`
  (`custom_processing/prepare_antscan_data_for_mesh_fitting.py`).
- Master's script is unchanged since the fix's branch base b5bf9565, and fix commit 0c9687fc touches
  only this file -> it applies cleanly to today's master.
- Code matches the claims: V0 fixed weld = 0.2% bbox (l.54); V1 weld = min(0.2% bbox, 0.3 x median
  edge) (l.104-114); 20% face-loss RuntimeError (l.144-148); debris filter < 4 faces before weld and
  after cleaning, bridging of thin severed connections, all larger islands kept (l.85/130/213/481/508)
  where V0 keeps only the largest piece after cleaning (l.28, 45). Both versions still keep only the
  largest piece of the RAW scan at the start (identical behaviour; mention in the PR). Fallback weld
  distance 2 fires only for threshold <= 0 (zero-size mesh): unreachable in practice.
- **CI would fail on V1 as pushed:** `ruff check` passes but `ruff format --check` (ruff 0.15.20, CI's
  pin) fails; master's version passes both. `ruff format` changes 34 lines and the AST before/after is
  identical -> formatting only, behaviour unchanged. Formatted file:
  `versions/V1_fixbranch_ruff-formatted.py` (passes check + format). The PR carries this file.

## 2026-10-09 ~00:45 -- first 4 complete specimens (early read; full n = 40 pending)

- **Weld gap (logs):** V0's weld exceeds the smallest self-approach gap on 4/4 (median 3.3x; Odontomachus
  5.47 vs 0.40 = 13.6x). V1's exceeds it on **2/4** (Odontomachus 1.01 vs 0.40, Psalidomyrmex 1.17 vs
  0.89) and V2's post-weld check reports those two `fused` (closest pair merged into one vertex). Weld
  face loss 0.1-1.1%: the 20% abort cannot catch a narrow-gap fusion. Severity unknown from a single
  closest pair -> judge by renders + non-manifold edges.
- **Measurements (t1_measure, first pass):** alignment residual 0.16-0.31% diag (QC ok); fidelity p99
  0.12-0.25%, equal across versions (no version adds geometry absent from the scan). Uncovered outer
  surface V0 -> V1: Psalidomyrmex 9.13 -> **2.07%** (anatomy recovered), Temnothorax 0.68 -> 0.26%,
  Acromyrmex 5.68 -> 5.66%, Odontomachus 9.23 -> 9.07% (unchanged).
- **P0 provenance:** today's master reproduces the 2024 worker meshes almost exactly (uncovered 5.68 vs
  5.68, comparable pieces) -> the paper corpus was made with master's current script.
- **Measurement bug found and fixed:** trimesh face_adjacency ignores edges shared by > 2 faces, so the
  non-manifold outputs looked shattered (6,974 "pieces" for V0 Acromyrmex). By vertex/edge connectivity
  (Blender loose parts): V0 40 pieces (99.92% in the largest), V1 126 (99.16%). New, more direct metric:
  **non-manifold edges V0 11,773 vs V1 4,075** (fused-surface signature), boundary edges 9,870 vs 13,421.
- **Render bug fixed:** bpy vertex collections do not support step slicing; now foreach_get.

## 2026-10-09 ~01:00 -- side effect: the script writes into the raw data's JSON

- All three versions append `processed_*` stats to `<raw>.json` next to the input STL, AFTER exporting the
  mesh (V0 l.699-711, V1 l.1053-1065, V2 l.1265-1292). The JSON is not read for processing -> the
  geometry results are NOT contaminated. But our concurrent V0/V1/V2 runs overwrite each other's fields
  in the LOCAL raw copies (/hpcwork/nao48500/antscan_data/*/*.json); canonical Drive copies untouched
  (Drive: 1,220 bytes, 2024-09-28). TODO after the run: restore the 40 local JSONs from Drive and re-check
  the raw STL md5s against `out/drive_md5.txt`. Pre-existing behaviour (master does it too); note it in
  the PR description, not a blocker.
- Corrected measurements (vertex/edge connectivity), first 4 specimens, V0 -> V1:
  non-manifold edges Acromyrmex 11,773 -> 4,075; Psalidomyrmex 2,452 -> 3,592; Odontomachus 7,947 -> 7,474;
  Temnothorax 1,024 -> 1,254. Pieces V0 10-111 (largest 99.1-99.98%) vs V1 126-978; V1 debris share up to
  13% (Psalidomyrmex). Mixed picture at n = 4 -> full run + renders decide.

## 2026-10-09 ~01:15 -- visual review, first 4 (renders/sheets/*.png: raw | V0 | V1 | V2 | 2024)

- **Psalidomyrmex:** V1/V2 contain a leg segment (separate piece, blue) that V0 and the 2024 mesh lack;
  the raw scan has it -> genuine anatomy recovered (uncovered 9.13 -> 2.07%). It is kept as a
  DISCONNECTED piece (bridging did not rejoin it) -- the PR must not claim reconnection. A few tiny
  specks are also kept (raw has floating debris that V0's largest-piece rule removes).
- **Odontomachus:** no visible difference between versions; the V2 `fused` closest-pair verdict
  (gap 0.40 vs weld 1.01) is not visible at full-specimen scale -> treat the gap verdict as a sensitive
  warning, not proof of damage; confirm with close-up crops before concluding. The 9% uncovered surface
  is identical in all versions: the raw scan contains fragments ALREADY disconnected in the published
  mesh (floating antenna/leg segments, a dark detached object) and every version keeps only the largest
  piece of the RAW scan as its first step.
- **Scope of the fix (for the PR and for Task 5):** it recovers anatomy severed by the pipeline's own
  ray-cast cleaning; it cannot recover anatomy already broken in the published scan.
- **Motivating case added:** Acanthognathus cf. ocellatus CASENT0744647 (AntScan scan id 5x/24-41;
  not in our 785-specimen download), where master removed the whole head. Downloaded from biomedisa
  (processed id 5692, 24-41.stl, 2,964,006 triangles, md5 6d705da9...; extra_raw/), run as job 4884586.

## 2026-10-09 ~01:30 -- non-inferiority check (user criterion: V1 must be as good as V0 or better on every specimen)

Split by main body (largest piece) vs rest, 4 specimens:
- **Main-body non-manifold edges (fused-surface signature): V1 <= V0 on 4/4** (11,772 -> 4,008;
  2,415 -> 2,289; 7,947 -> 7,230; 1,012 -> 598). Better.
- **Main-body open edges: V1 > V0 on 3/4** (9,705 -> 12,742; 10,715 -> 14,628; **5,326 -> 21,289**;
  9,203 -> 8,857). Worse: a NEW defect class. Extra pieces are 0.27-0.73% of area (4.15% on
  Psalidomyrmex incl. the recovered leg).
- **Mechanism (V2 snapshots, Odontomachus main body):** pre-weld 70,550 open / 57,082 non-manifold ->
  after weld + holes_fill 24,948 / 7,250 -> after decimation 21,453 / 7,244. Decimation does not open
  holes; the open edges are ray-cast seams the weld leaves unclosed. V0's coarse weld (5.47) closes
  them but also exceeds the narrowest real gap (0.40) 13.6x; V1's fine weld (1.01) avoids most fusion
  but leaves ~4x more seams open. **No single global weld distance satisfies both on this specimen.**
- **Verdict so far: V1 is NOT non-inferior** (better fusion, more open seams). Candidate principled fix:
  topology-aware weld -- coarse merge restricted to open-boundary (seam) vertices, fine merge
  elsewhere. New code; would need the full protocol re-run. Decision for the user.

## 2026-10-09 ~01:50 -- faithfulness to the SCAN (user: judge against the scan, not against master)

`t1_fidelity_maps.py` (job 4884694) + `t1_plot_maps.py`; maps in /hpcwork/nao48500/holotype_task1/maps/.
- **Noise floor** (raw vs itself): median 0.046-0.093%, p99 0.125-0.236% of the diagonal.
- **Deviation output -> scan: at the floor for every version** (V1 median/p99 0.048-0.097 / 0.118-0.242%);
  nothing above 0.5% of the diagonal except 0.01-0.02% of Temnothorax vertices (both versions). Max
  deviation V0 -> V1: 0.51 -> 0.29, 0.30 -> 0.36, 0.38 -> 0.24, 0.87 -> 0.81 (V1 lower on 3/4).
  **No version alters the scan measurably.**
- **Scan surface lost by V0 only: 0.02 / 7.08 / 0.17 / 0.41%; by V1 only: 0.00 / 0.02 / 0.00 / 0.00%.**
  V1 never loses scan surface V0 keeps. Psalidomyrmex map: V0 drops a whole front leg (tibia + tarsus)
  plus mesosoma/gaster patches; zero blue. Surface lost by both = already detached in the published mesh.
- Consequence (user's point, confirmed): where V1 scores worse than master on mesh-hygiene metrics
  (pieces, open edges), part of it is MORE real structure being kept. The open seams are topology
  (near-zero-width cracks), not missing surface; they matter only to closed-surface steps, not to the
  D1 fitter (area-sampled chamfer on the target).
- V1's output has visibly uneven vertex density in places (mesh regularity, not fidelity); quantify via
  the logged face-size CoV across all 40 before the PR.

## 2026-10-09 ~02:00 -- THE MOTIVATING CASE REPRODUCES: Acanthognathus cf. ocellatus CASENT0744647

Job 4884586; raw = AntScan processed mesh 24-41.stl (scan id 5x/24-41), md5 6d705da9...
- **V0 (master) removes the entire head** (head capsule, trap-jaw mandibles, antennal bases): uncovered
  outer scan surface **28.39%**. Visual: renders/sheets/Acanthognathus_cf.ocellatus_CASENT0744647.png.
- **V1 keeps it and reconnects it to the body** (rendered in the main-body colour, not as a separate
  piece): uncovered **0.01%**.
- V1 is better on every other metric too: fidelity p99 0.104 vs 0.162%; non-manifold edges 241 vs 1,085;
  open edges 2,655 vs 4,665. Only more small pieces (203 vs 52; 1.30% of faces in pieces < 1%).
- Mechanism (consistent with the code): ray-cast cleaning severs the thin neck; V0's largest-component
  step then discards the head; V1's bridge_nearby_islands restores the neck faces from the original mesh.

## 2026-10-09 ~02:10 -- P5: does the 20% face-loss abort fire? (t1_abort_test.py, V1's own apply_modifiers)

- A: two overlapping sheets 0.1 apart, weld 0.3 -> 50.0% face loss (coinciding faces become duplicates),
  2 pieces -> 1, **abort raised**.
- B: one sheet with 0.1 spacing, weld 0.3 -> 92.0% loss (collapse), **abort raised**.
- C: control, sheets 2.0 apart -> weld merges nothing (2 pieces stay 2), **no abort** (correct). Total face
  count still drops 3,364 -> 2 because the planar dissolve after the weld merges the flat test triangles;
  irrelevant for non-planar scans, and the abort correctly measures the weld step only.
- **P5 PASSES for catastrophic welds** (whole-surface fusion, collapse). It does NOT catch local
  narrow-gap fusion (real specimens: 0.1-1.1% loss even where V2 reports `fused`). PR wording: "abort
  guards against catastrophic welds", not "against fusion".

## 2026-10-09 ~03:00 -- weld gap at n = 39 (t1_weld_gap.py, logs only; Pseudoneoponera V1/V2 still running)

- V0 weld distance > narrowest self-approach gap on **37/39** (median 2.08x the gap).
- V1 weld distance > gap on **26/39** (median 1.22x, max 3.54x); V2's post-weld check: fused 26, preserved 13.
- V1 weld face loss median 0.80%, max 2.70%; **0 aborts** on real scans.
- **Claim P3 holds only partially: the adaptive weld reduces fusion at the narrowest gap (13 protected vs
  2) but does not prevent it.** Most raw scans contain a spot (e.g. a leg touching the body) closer than
  even the finer weld distance. Severity (benign single-vertex merge vs visible fusion) is judged from
  main-body non-manifold edges, fidelity maps and renders for all 40 (job 4884778). PR wording:
  "reduces, does not eliminate".

## 2026-10-09 late: measurement artefacts in the surface-coverage metric (t1_measure.py)

- Raw AntScan STLs contain thousands of loose pieces, incl. flat mounting plates up to 17-27% of the scan
  area. Every version drops them in its first step (find_largest_component, unchanged). t1_measure.py
  scored against the whole scan, so these counted as "lost" for all versions and distorted the PCA start
  frame: ICP failed for all versions on Cephalotes minutus (p99 5.5%) and for V1 only on Discothyrea
  (flipped frame, residual 0.53% vs V2 0.16%), producing a spurious "V1 loses 16.3%".
- t1_measure_v2.py: reference = largest raw piece (pipeline's definition), cross-seeded ICP. Discothyrea:
  V1 uncovered 0.24% vs V0 0.38% (V1 better); deviation p99 V1 0.364% vs V0 0.170% (to review).
  Full re-score of 39: job 4886050 -> out/measure_v2_*.json.
- Pseudoneoponera (1.2 GB, 11.9M vertices): V1/V2 exceeded the 2 h cap (V0 44 min); excluded at the
  user's request; runtime on very large scans is an open question for the bridging step.
- PR table restricted to alignment-free numbers (weld gap, face loss, non-manifold, aborts, Acanthognathus).
