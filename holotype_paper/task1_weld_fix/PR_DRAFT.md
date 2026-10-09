# Issue (draft)

**Title:** AntScan preprocessing drops anatomy severed by ray-cast cleaning and can weld across nearby surfaces

`custom_processing/prepare_antscan_data_for_mesh_fitting.py` on master:

1. After ray-cast cleaning it keeps only the largest connected component. Anatomy that the cleaning
   disconnects through a thin bottleneck is discarded. On *Acanthognathus* cf. *ocellatus*
   CASENT0744647 (AntScan scan 5x/24-41) the entire head (head capsule, trap-jaw mandibles, antennal
   bases) is removed: 28.4% of the visible scan surface is missing from the output.
2. The Weld distance is a fixed 0.2% of the largest bounding-box dimension. On raw AntScan scans this is
   commonly larger than the narrowest gap between separate parts of the surface (e.g. 5.47 vs 0.40 on
   *Odontomachus hastatus* CASENT0745037), so the weld can merge vertices across that gap.
3. There is no check on the weld's effect on the mesh.

# Pull request (draft)

**Title:** fix(antscan): adaptive weld distance, weld face-loss abort, island handling

Closes #<issue>.

## Changes (`custom_processing/prepare_antscan_data_for_mesh_fitting.py` only)

- **Weld distance** = min(0.2% of the largest bounding-box dimension, 0.3 x median edge length).
- **Weld face-loss abort:** `RuntimeError` if the Weld step removes more than 20% of faces
  (`max_weld_face_loss_pct`).
- **Island handling:** ray-cast cleaning keeps `keep_rings` face rings around each hit,
  reconnects kept islands along short paths of the original mesh (`bridge_nearby_islands`, at most
  `max_bridge_hops` edges), removes components below `min_island_faces`; after welding all components
  above that size are kept instead of only the largest.
- Module docstring describing the pipeline. No other file changes; output format and CLI unchanged.

## Validation

Tested on randomly sampled raw AntScan scans (MD5-identical to the shared-drive copies) and on the
*Acanthognathus* specimen, master vs this branch, same Blender 4.2.23.

- *Acanthognathus*: master removes the whole head (28.39% of the scan surface missing); this branch
  keeps it and reconnects it (0.01% missing).
- The weld distance exceeds the narrowest gap between separate surfaces on fewer scans than with master.
- Weld face loss stays small (median 0.80%, max 2.70%); the abort never fired on real scans.
- Fewer non-manifold edges (the signature of merged surfaces) on most scans (median 1,545 vs 4,258).
- Constructed cases: the abort fires for whole-surface fusion (50% face loss) and collapse (92%), not
  for a control.

## Known limitations

- The adaptive weld reduces merging across narrow gaps but does not eliminate it.
- The finer weld can leave more open seams on some scans. Topology only; no scan surface is removed.
- More small pieces are kept; where there are many, they cover scan surface that master drops.
- Anatomy already disconnected in the published scan is still dropped by the initial largest-component
  step (unchanged behaviour).
- The abort guards against catastrophic welds; local merging across a narrow gap is not caught.
- As before, the script writes statistics into the JSON file next to the input STL.

Evidence: `holotype_paper/task1_weld_fix/` on `feature/investigation` (PROTOCOL.md, FINDINGS.md,
verdict_table.csv, renders and fidelity maps).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
