Metric-validation sample: 9 specimens evenly spaced across the full range of
thickness_p5_ratio_aw_over_orig (5th-percentile SDF thickness ratio, alphawrap/Fabian), sorted
worst to best, from comparison_output/fabian_vs_alphawrapfix_20260817/report/per_specimen.csv.

Per user instruction (2026-08-17): comparisons use fabian_processing.py's output as the baseline,
NOT the patched Weld "original" pipeline used in earlier reports in this diagnostics tree.

Purpose: check whether visual badness actually tracks the metric's severity ranking before
trusting the aggregate collapse statistics further. If a folder ranked "worse" by p5_ratio
doesn't look worse in Blender than one ranked "better", that's evidence the metric (single-ray
SDF sampling) is noisy rather than the reconstruction being fine.

3 files per folder: raw_input_preprocessing.obj (pre-pipeline scan), fabian_baseline.obj
(fabian_processing.py output), alphawrap_postfix.obj (alpha_wrap, post offset-fix + bounds-fix).

Ranked worst -> best by p5_ratio (vs Fabian):
1_worst   p5=0.0000   alpha_wrap              gate=True   Anochetus_risii_CASENT0877608
2         p5=0.0005   manifold_plus_fallback  gate=True   Labidus_praedator_CASENT0744236
3         p5=0.1357   alpha_wrap              gate=True   Eciton_hamatum
4         p5=0.5063   alpha_wrap              gate=True   Dilobocondyla_fouqueti
5_median  p5=1.0000   manifold_plus_fallback  gate=False  Lasius_nr._fuliginosus
6         p5=1.5055   alpha_wrap              gate=True   Odontomachus_troglodytes
7         p5=2.4453   alpha_wrap              gate=True   Strumigenys_sp._appretiata_group
8         p5=5.9334   alpha_wrap              gate=False  Heteroponera_panamensis
9_best    p5=20.2181  alpha_wrap              gate=False  Oxyopomyrmex_saulcyi

Note: gate=False shows up on BOTH ends of this ranking (worst-ish AND best-ish), and several
alpha_wrap (not just fallback) specimens land at the extreme-low end (p5=0.0000-0.14) - itself a
data point for whether p5 is trustworthy per-specimen, or only in aggregate/median form.
