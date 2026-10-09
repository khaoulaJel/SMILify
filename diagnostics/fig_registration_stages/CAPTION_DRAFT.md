# Caption draft: registration stages on a cleaned Atta worker

**Registration stages of the default recipe (D1_PROD) on a cleaned scan (positive control).** One
*Atta vollenweideri* worker (WOLO scAnt scan 13, 12.5 mg; chosen before fitting as the cleanest
standard-resolution scan nearest the median body mass of the 20). **a**, rest template (init).
**b**, after skeleton placement (pose, shape and per-joint scale fitted, no free-form deformation;
body, then legs, then joint refinement). **c**, **d**, after the coarse and fine free-form stages. The
fitted template (blue) is drawn over the scan (grey), so grey that stays visible is scan surface the
fit has not reached. Template surface lying outside the scan is not shown by this overlay; the F-score
counts it. Same orthographic camera and lighting in all panels. Below each panel: bidirectional chamfer
(squared distances, scan normalised to max |coordinate| = 1); F-score at τ = 0.01 of the half-extent
(F₀.₀₁); number of template vertices penetrating a non-adjacent body part. The init count (18) is
mandible–antenna contact already present in the template. Gaster–leg penetration is zero at every
stage. At **d** this specimen ranks 6th of 20 by F₀.₀₁ (median 0.936, range 0.897–0.973). Seed 0, one
run; fitter run-to-run variation is about 0.01 in F. Surface agreement does not certify anatomical
correspondence: the grey rims along the legs show that the fitted legs stay thinner than the scanned
ones, and no expert joints exist for these scans.

Numbers: `stages.csv` (specimen 13) and `stages_all20.csv` (all 20), with config / specimen-set / seed columns; provenance
in `run_cpu1/PROVENANCE.md`.

## Points the main text or Methods must not contradict

- The 20 Atta fits in the paper's shape space used master `fitter_3d.optimise` (`init_rot_lock`, no
  hierarchy, no w_offset / w_limit / w_scale), not D1_PROD. The stage names coincide but the recipes
  differ, so these numbers are not those fits' numbers (relevant to Task 3).
- This is a clean scan in a near-rest posture; the legs already align after body placement. It shows
  the pipeline on the easy case, not on the uncleaned AntScan meshes the paper's claim is about.
- The fit was run on CPU, one specimen per job (`DEVIATIONS.md` D1, D2); recipe, model, code and seed
  unchanged.
