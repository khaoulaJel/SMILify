# Draft note to Fabian: Task 2 status + one model defect (Khaoula sends; not sent)

Hi Fabian,

Task 2 (Atta stage figure) is done, in `diagnostics/fig_registration_stages/`: a 300-dpi PNG, the
per-stage CSV, and a caption draft. Pre-registered before fitting (specimen, command, metrics,
checks); every number was cross-checked against D1_PROD's own metrics.csv and an independent
recompute.

Four things you should know:

1. **"80 cleaned Atta scans":** there are 20 Atta scans on disk (the WOLO workers). I read the 80 as
   60 artist-cleaned + 20 Atta. Is that right?

2. **The shape-space Atta fits are not D1_PROD.** They were made with master `fitter_3d.optimise`
   (init_rot_lock, no hierarchical stages, no offset/limit/scale terms). The figure shows D1_PROD, as
   your package asked, so its numbers are not those fits' numbers. This also answers part of Task 3:
   the Atta-80 recipe is not unchanged.

3. **Skinning-weight defect in every OmniAnt model file.** 134 left-mandible vertices have skinning
   weights summing to less than 1 (minimum 0.401) in `OmniAnt_25PCs_joint_limited.pkl` (production),
   `_anterior_limited`, `smpl_ATTA.pkl` and the `SMPL_fit*` files. `SMIL_OmniAnt.pkl` has 120 such rows
   on the thorax/coxae (minimum 0.706). Linear blend skinning scales each of those vertices by its weight
   sum, even at zero pose. The result is a thin spike running from the left mandible base back through
   the head, up to 0.32 out of place at rest and still 0.24 after a full fit. It is invisible at figure
   scale, but it is in every posed output, and it affects left-mandible geometry and anything measured
   there. Evidence: `diagnostics/fig_registration_stages/probes/skinning_weight_defect_PROBE*` and
   `run_cpu1/probes/mandible_weight_defect_zoom_PROBE.png`. I have not changed the model (no new
   runs); fixing it means renormalising the weight rows, which changes every fit.

4. **Two F@0.01 definitions in the repo.** One is τ = 0.01 absolute in the unit box (D1_PROD's
   metrics.csv, used in the figure: 0.947). The other is τ = 1% of the bbox diagonal (trainer.py:
   0.996 on the same fit). Before the Task 4 table I will check which one T1–T9 used.

The figure is a positive control: a clean scan, near rest pose, F 0.947. The caption says so, so it
cannot be read as the AntScan result.

Khaoula
