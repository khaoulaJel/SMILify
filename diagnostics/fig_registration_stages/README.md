# Registration stage figure (Holotype work package, Task 2)

D1_PROD on one cleaned *Atta vollenweideri* worker (WOLO scan 13), four stages, same camera and lighting.

| file | what |
|---|---|
| `fig_registration_stages.png` / `.pdf` | the figure (style B: fit in blue over the scan in grey), 180 × 44 mm, 300 dpi; the PDF has editable text (Liberation Sans = Arial metrics) |
| `stages.csv` | per-stage chamfer, F@0.01, penetration for specimen 13 (`panel=True` rows are the four panels), plus the 20-specimen medians |
| `stages_all20.csv` | the same metrics for all 20 Atta scans, with ranks |
| `CAPTION_DRAFT.md` | caption, plus what the text must not contradict |
| `PREREGISTRATION.md` / `.FREEZE` | plan frozen before fitting (sha256 f4f265e2) |
| `DEVIATIONS.md` | D1 CPU / account `default`, D2 one specimen per job; job IDs; scoring rerun |
| `run_cpu1/` | the run behind the figure: `PROVENANCE.md`, raw scorer CSVs, Blender layers and text-free transparent panels (`blender/`), pytorch3d draft panels (`panels/`), checks (`probes/`) |
| `probes/` | orientation check, early skeleton check, skinning-weight defect probe |
| `score_stages.py`, `merge_singles.py`, `annotate_csv.py` | scoring |
| `blender_render.py` (+ `submit_blender.sbatch`), `compose_figure.py` | final render (Blender 5.0.1 Cycles) and figure assembly |
| `render_stages.py` | pytorch3d overlays and checks; the earlier draft figure |
| `submit*.sbatch` | fit and scoring jobs |
