# Presentation figures — SMILify as a validated measurement instrument

Built from **already-completed experiments**. No new experiments were run for any figure here.
Each is written as `figNN_*.py` and emits both `.png` (300 dpi) and `.pdf`.

## Built

| # | file | task | message |
|---|---|---|---|
| 1 | `fig01_wolo_headwidth_mass.png` | 1 (Fabian) | Auto-recovered HW scales with body mass. **slope 0.385 [0.373, 0.397], R²=0.996, n=20**; hand-measured 0.395, published 0.3949 |
| 2 | `fig02_headwidth_anatomy.png` | 2 (Fabian) | What HW *is*, and that it lands on the true widest point across 4 real head shapes |
| 5 | `fig05_trait_accuracy.png` | 5 | Not every measurable quantity is equally trustworthy (recovery error, coloured by panel role) |
| 6 | `fig06_accuracy_vs_repeatability.png` | 6 | **Repeatability ≠ accuracy** — recovery error vs seed CV, quadrant-labelled |
| 7 | `fig07_m4a_qc.png` | 7 | 753/757 HW vs 711/757 HL, and the HL failure is genus-structured (p=0.025) |
| 9 | `fig09_m4b_genus_structure.png` | 9 | Validated head *shape* differs among genera, η²=0.331, p<5×10⁻⁵ |
| 10 | `fig10_m4c_variance.png` | 10 | Genus 23.7% > species 17.7%, and it survives singleton removal (22.5 vs 19.8) |
| 11 | `fig11_m4e_bridge.png` | 11 | Dense→scalar R²=0.719; held-out genus η² 0.409±0.044 vs random 0.236±0.049 |

## Two corrections made while building these — both matter scientifically

**1. Head width is NOT the added `b_h_l`/`b_h_r` joints.** The production model
(`OmniAnt_25PCs_joint_limited.pkl`) has only `b_h`; the reporter bones from the *Atta*
head-width replication are not rig joints in it. Everything in M2/M3/M4 and in Figure 1 uses
`TRAITS["HW"] = ("ext", ("b_h", 0))` — the **maximum transverse extent of the head part** after
rigid alignment to the template head frame, recomputed per specimen. Figure 2 therefore shows the
real measurement locus (the two head vertices realising the maximum) rather than a picture of
added joints. This is the stronger figure anyway: it demonstrates the points track the widest
point as head shape changes, which a fixed vertex pair could not.

**2. Figure 1's independence is partial, and the caption must say so.**
`HW_mm = (HW/WL)_model × BL_mm`, because physical scale is restored per specimen with
`BL_mm / BL_model`. The model therefore supplies the **shape ratio**; absolute size comes from the
hand-measured body length. Figure 1 validates that SMILify recovers head-width-relative-to-body
correctly — it is **not** an independent absolute-size measurement. The reference series is plotted
alongside precisely so this reads as a recovery check rather than a re-demonstration of a known law.

### Also built

| # | file | task | message |
|---|---|---|---|
| 3 | `fig03_pipeline.png` | 3 | Before/after — registration turns incomparable scans into a common instrument |
| 4 | `fig04_why_we_trust.png` | 4 | recoverable → repeatable → transferable → biologically meaningful, in one panel |
| 8 | `fig08_scale_identifiability.png` | 8 | 7.2× implied-scale spread; **voxel correction does not fix it** (7.2× → 7.2×, r 0.43 → 0.71) |
| 14 | `fig14_failure_map.png` | 14 | The whole investigation as one problem × layer matrix with status |
| 15 | `fig15_operating_regime.png` | 15 | What SMILify can currently measure — the scientific product |

A third correction, made while building fig15: **HL is not shown as an unqualified primary.** Its
frozen panel role is unchanged, but the figure carries the M4-A/M4-C caveat inline (genus-structured
failure; genus signal does not survive singleton removal). Presenting it beside HW without that
would misrepresent what M4 found.

## Not yet built

| # | task | blocker |
|---|---|---|
| 12–13 | joint-alignment benchmark + strategy ranking / per-joint heatmap | **needs Fabian's hand-annotated joint GT.** `gt_expert` (n=12) exists in-repo; whether that is the set he means is unconfirmed, and n=12 is thin for ranking strategies |

Everything else in the final figure plan is complete: **12 figures**, covering tasks 1–11 and 14–15.
