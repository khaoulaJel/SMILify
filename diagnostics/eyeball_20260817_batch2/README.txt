4 files per specimen: raw_input_preprocessing.obj (pre-pipeline scan), original_patched.obj
(prepare_antscan_data_for_mesh_fitting.py, the bug-fixed Weld baseline), fabian_unpatched.obj
(fabian_processing.py, no filter_small_components/bridge_nearby_islands/weld-safety-check),
alphawrap_postfix.obj (alpha_wrap, post offset-fix).

1_Acanthostichus_aff_brevicornis_REGRESSION
  One of the 4 specimens that worked with alpha_wrap before the offset fix and now falls back to
  ManifoldPlus after it (fidelity_gate_passed=True even so).

2_Atopomyrmex_mocquerysi_CLEAN_alpha_wrap
  Clean alpha_wrap success both before and after the offset fix, fidelity_gate_passed=True.
  Reference/positive-control specimen.

3_Lasius_nr_fuliginosus_STILL_BROKEN
  Still on ManifoldPlus fallback after the offset fix, fidelity_gate_passed=False. One of the
  unresolved specimens.
