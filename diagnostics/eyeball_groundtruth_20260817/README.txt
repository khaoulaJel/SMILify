Visual ground-truth inspection set (2026-08-17), built from diagnostics/phase1_corpus_20260817/
(the Phase 1 instrumented alpha_wrap run across all 50 bench specimens).

QUESTION THIS SET EXISTS TO ANSWER: are the 6 p99.9 whole-surface-deviation fidelity-gate
failures actually bad reconstructions, or acceptable meshes being rejected by an overly strict
threshold? Secondary: do the low-p5 cases show real thin-feature erosion, and do the hard/fallback
cases look as bad as their numbers suggest?

Do NOT read fidelity_gate_passed/p5_ratio as ground truth - that's exactly what's being tested.
Label each folder GOOD / BORDERLINE / BAD from the actual meshes.

3 files per folder (2 for the crashed pair, no final output exists for those):
  raw_input.obj          - the raw pre-pipeline scan
  pre_reconstruction.obj - post-clean, pre-alpha-wrap (what CGAL/ManifoldPlus actually received)
  final_output.obj       - the pipeline's final exported mesh (MISSING for 10/11, see below)

======================================================================
GROUP 1: fidelity_gate_passed=False (all 6 - the group central to this round's question)
All are approach=alpha_wrap, tier=A, gap_check verdict='preserved' (NOT fused) in 5/6,
'indeterminate' in 1/6 (Strumigenys_alberti) - i.e. none failed via gap fusion, all failed via
worst_fidelity_deviation exceeding 3x target_cell_size.

01_FIDFAIL_Crematogaster_nawai            p5_ratio=50.38   worst_dev=34.86  (gap_ratio=0.00106)
02_FIDFAIL_Mayriella_sp                   p5_ratio=8.96    worst_dev=14.98  (gap_ratio=0.00100)
03_FIDFAIL_Pseudomyrmex_cf_solis          p5_ratio=5.82    worst_dev=24.67  (gap_ratio=0.00138)
04_FIDFAIL_Strumigenys_appretiata_group   p5_ratio=7.30    worst_dev=7.591  (gap_ratio=0.00063)
05_FIDFAIL_Eurhopalothrix_procera         p5_ratio=0.24    worst_dev=3.432  (gap_ratio=0.00191)  <- ALSO low-p5, unlike the other 5
06_FIDFAIL_Strumigenys_alberti            p5_ratio=1.58    worst_dev=80.04  (gap_ratio=0.00191)  <- gap_check was 'indeterminate', not 'preserved'

Note: 01-04 and 06 show healthy-to-generous p5 (thin features NOT collapsed) despite the gate
failure - if these look GOOD visually, that's evidence the 3x threshold is too tight for them.
05 (Eurhopalothrix_procera) is the one case in this group where BOTH signals flag a problem -
treat it as the most likely genuine defect in this group, not a threshold artifact.

======================================================================
GROUP 2: hard cases / fallbacks (Alpha Wrap under stress or defeated)

07_FALLBACK_TierC_Labidus_praedator_237       manifold_plus fallback, needed relative_alpha>2786, worst_dev=6.62, p5=0.00017 (catastrophic)
08_FALLBACK_misTiered_Nesomyrmex_angulatus    manifold_plus fallback, tier B est. was wrong (needed >2500), worst_dev=5.02, p5=0.0017 (catastrophic)
09_FALLBACK_misTiered_Strumigenys_stenorhina  manifold_plus fallback, tier B est. was wrong, worst_dev=3.06, p5=0.0050 (catastrophic)
10_CRASHED_noOutput_Cyphoidris                pipeline OOM-crashed (p5-metric bug, see memory) - NO final_output.obj, raw+pre_reconstruction only
11_CRASHED_noOutput_Lasius                    pipeline OOM-crashed (same bug) - NO final_output.obj, raw+pre_reconstruction only
12_HARD_SUCCESS_Ectatomma_brunneum            alpha_wrap SUCCEEDED despite needing relative_alpha~1575, p5=0.106 (low, ungated)
13_HARD_SUCCESS_Labidus_praedator_236         alpha_wrap SUCCEEDED (relative_alpha~1513), p5=0.194 (low, ungated) - same genus as #07, useful contrast pair

======================================================================
GROUP 3: clean Tier-A successes (baseline "this is what good looks like")

14_CLEAN_TierA_Anochetus_risii_609        p5=1.80, worst_dev=170.4 (gate=True - note worst_dev is large in absolute terms but still under this specimen's own looser target_cell_size)
15_CLEAN_TierA_Atopomyrmex_mocquerysi     p5=1.01, worst_dev=86.25 (gate=True) - previously eyeballed clean in an earlier round this session
16_CLEAN_TierA_Odontomachus_troglodytes   p5=1.42, worst_dev=27.69 (gate=True)

======================================================================
GROUP 4: low-p5 alpha_wrap cases (possible thin-feature erosion, currently UNGATED since
p5 gating only applies to the ManifoldPlus fallback path)

17_LOWP5_Carebara_trechideros   p5=0.107, worst_dev=4.368 (gate=True)
18_LOWP5_Dorylus_fulvus         p5=0.164, worst_dev=4.479 (gate=True)
19_LOWP5_Eciton_hamatum         p5=0.171, worst_dev=5.195 (gate=True)
(12_HARD_SUCCESS_Ectatomma_brunneum and 13_HARD_SUCCESS_Labidus_praedator_236 above also belong
to this group - p5=0.106 and 0.194 respectively - kept in Group 2 to avoid duplicating folders.)

======================================================================
Label each folder GOOD / BORDERLINE / BAD based on the actual mesh (legs/antennae intact,
no visible holes/fusion/distortion) - not the numbers above.
