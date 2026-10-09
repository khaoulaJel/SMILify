# Task 1: is the weld fix real? Test protocol (written 2026-10-08, before any run)

## Context established before writing (read-only checks)

- Three versions of `custom_processing/prepare_antscan_data_for_mesh_fitting.py` exist:
  **V0** master (`upstream/master`, 761 lines; byte-identical to `custom_processing/fabian_processing.py`),
  **V1** the pushed fix (`origin/fix/apply-modifiers-weld`, commit 0c9687fc, 30 Jul, 1119 lines),
  **V2** a later local version (`feature/investigation` WIP snapshot 546ea0fb, 1345 lines; adds
  fidelity/degeneracy reporting and bridge-search parameters).
- **No fit used in the paper depends on this script's changes.** All 757 worker_ALT meshes are dated
  Oct-Nov 2024 (Fabian's original pipeline), and the 11 JAB meshes are byte-identical copies of them
  (md5 checked). The fix is a robustness PR; it does not change any reported number.
- Earlier evidence (bench50, 17 Aug, `comparison_output/bench50_clean_20260814_125245_Glc7oS/report/
  DECISION_SUMMARY.md` §6): a patched version produced **48/50 outputs with > 5 connected components
  (median 27)** vs V0's **7/50 (median 2)**. Which patched version that was is not recorded. So the fix
  may trade one failure (V0 discards real anatomy by keeping only the largest component) for another
  (fragmentation). This protocol tests both sides.
- CI = `ruff check` + `ruff format --check` (ruff 0.15.20) + `pytest -m "not slow"`. Nothing in CI
  exercises this Blender script.

## Claims under test (from the fix's commit message and docstrings)

| id | claim | how it would show |
|---|---|---|
| P1 | V0's fixed weld distance can merge vertices across separate surfaces (catastrophic) | large face loss in the Weld step; output geometry not present in the input |
| P2 | V0 discards real anatomy by keeping only the largest component | input surface (legs, antennae) not covered by the output |
| P3 | Adaptive weld distance (bbox AND median edge) avoids P1 | V1/V2 weld face loss small on every specimen |
| P4 | Debris filter + keep-large-islands + bridging recovers anatomy without debris | V1/V2 cover more input surface than V0; non-largest components are large pieces, not debris |
| P5 | 20% face-loss abort fires on genuinely bad welds and never on good ones | abort count on real data; a constructed failing case triggers it |

## Data

The 50 bench50 raw STLs already on disk (`diagnostics/fabian_stl_inputs_20260817/`, the inputs of the
17 Aug V0 run). Same inputs, same Blender 4.2.23 binary, all three versions, one run each (the script
is deterministic given `random_seed=0`; determinism checked by re-running V0 on 3 specimens and
comparing outputs byte-for-byte).

## Measurements (per specimen, per version; script `t1_measure.py`)

Frame: the script's own output frame; input STL transformed identically if the script recentres.

- **Fidelity output -> input**: distance from 50k output surface samples to the input surface:
  p99 and max, as % of the input bbox diagonal. Geometry not present in the input (P1 symptom).
- **Coverage input -> output**: fraction of input outer-surface area farther than tau = 0.5% of bbox
  diagonal from the output (P2 symptom: lost anatomy). The input contains internal geometry the
  pipeline is meant to remove, so coverage is measured on input samples that are visible from outside
  (ray-cast from a bounding sphere, same idea as the script), recorded as `outer_samples`.
- **Components**: count; face share of the largest; faces in components < 1% of total (debris).
- **Weld face loss %, abort events, runtime** from the logs (V1/V2 print them; V0 does not).

## Pass criteria for the PR (fixed now)

The PR carries version V* (V1 or V2, whichever passes; if both, V2 only if it is not worse than V1 on
any criterion) if, over the 50 specimens:
1. **P2 resolved:** median uncovered outer-surface fraction lower than V0, sign test p < 0.05, and no
   specimen worse than V0 by more than 1 percentage point.
2. **P1 absent:** p99 output->input distance not worse than V0 on any specimen by more than 0.1% of
   bbox diagonal.
3. **Fragmentation acceptable:** the faces outside the largest component are >= 90% in components of
   >= 1% of total faces (real pieces, not debris), median over specimens.
4. **No false aborts:** zero aborts on the 50 real specimens, or each abort shown (by inspection
   render) to be a genuinely corrupted weld.
5. **P5 works:** a constructed case (two parallel sheets closer than the weld distance) triggers the
   abort.
6. **CI:** ruff check + format clean on the changed file; fast pytest suite unchanged.

If no version passes, the PR is not opened; the finding goes to Fabian instead.

## Amendment A1 (2026-10-08, before any run): the bench50 "inputs" are processed meshes, not raw scans

Checked on 6 specimens: `diagnostics/fabian_stl_inputs_20260817/*_processed.stl` have ~80-94k faces and
the same extents/orientation as the processed worker meshes (`worker_ALT/*_processed.obj`), while the
raw AntScan STLs (`/hpcwork/nao48500/antscan_data/<name>/<name>.stl`) have 3.2-5.6 M faces in a
different orientation. **The 17 Aug comparison therefore re-processed already-processed meshes**; its
fragmentation numbers (median 27 vs 2 components) say nothing reliable about the fix on raw data and
are not used as evidence for or against this PR.

Data changes accordingly: the test runs on **raw AntScan STLs**. Added check **P0 (provenance)**: V0
on a raw scan should reproduce the 2024 worker mesh for the same specimen (face count and two-way
surface distance), which tests whether master's script is the pipeline that produced the paper's corpus.
