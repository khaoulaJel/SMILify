6 specimens, 3 files each: raw_input_preprocessing.(obj|stl) (pre-pipeline scan, fed to
both pipelines), original.obj (Weld pipeline output), alphawrap_postfix.obj (alpha_wrap output,
post offset-fix). Open all three in Blender per specimen to compare.


1_WORST_still_broken_ManifoldPlus_Ectatomma_brunneum
  Still on ManifoldPlus fallback after the fix. Worst p5 thin-feature ratio in the whole
  50-specimen batch (7.5e-6 - thinnest features ~0.0007% of original's thickness). If the
  collapse is real and visible anywhere, it's here.

2_REGRESSION_worst_chamfer_Acromyrmex_coronatus
  One of the 4 specimens that WORKED before the offset fix and now falls back to ManifoldPlus.
  Also the single worst Chamfer-distance outlier in the post-fix batch. Best candidate to see
  what a regression actually looks like.

3_REGRESSION_Acanthostichus_aff_brevicornis
  Another of the 4 regressions. Also the very first specimen hand-checked earlier in this
  investigation (original pipeline's non-watertight/fragmented baseline was confirmed on this
  one specifically).

4_FLAGSHIP_FIXED_Cephalotes_spinosus
  Was a CGAL bisection timeout -> ManifoldPlus fallback before the fix. Now clean alpha_wrap,
  fidelity_gate_passed=True, 226s. The main "fix worked" example.

5_BEST_CASE_Pseudomyrmex_cf_solis
  Best (highest) p5 thin-feature ratio of any alpha_wrap specimen post-fix (25.9x - alphawrap's
  thin features come out THICKER than original's, likely because original's own baseline is
  itself degraded here). Positive control / best-case reference.

6_GROUNDTRUTH_large_complex_Dorylus_sp
  Dorylus_sp._CASENT0744703 - not part of the bench50 batch, this pipeline's own historical
  ground-truth/characterization specimen (~968K pre-reconstruction vertices, most complex
  geometry tested). raw_input_preprocessing.stl is the raw scan (no prior "original" pipeline output
  exists for this one outside bench50); alphawrap_postfix.obj is the new-offset result,
  fidelity_gate_passed=True, gap preserved across all 8 bisection steps.

Numbers backing all of this: comparison_output/bench50_offsetfix_20260816/report/ and
comparison_output/bench50_clean_20260814_125245_Glc7oS/report/DECISION_SUMMARY.md
