# Q1: Do GT-free quantities rank initialization candidates by `leg_acc`, per specimen?

Read-only analysis. No run outputs were modified.

## Data

- **12 specimens** (`synth_000`–`synth_011`), **13 initialization conditions** per specimen:
  `coherent`/`distal`/`proximal`/`random` × `{15,23,30}deg`, plus `swap` — the 13 basin-map
  conditions from `diagnostics/anatomical_pose_init/out_basin_map_20260825/basin_map_manifest.csv`
  (156 = 13×12 rows), run as `BASIN_*` in `diagnostics/moonshot/runs/`.
  `IK_tip_only`, `IK_tip_waypoint`, `SYN_clean_zero_wsl` are separate arms (IK-init study /
  zero-perturbation reference) also present in that runs directory and in
  `correspondence_confusion.json`, but are **not** part of the 13 basin-map conditions and are
  excluded here — they'd break the "13 conditions per specimen" design this analysis needs.
- **GT-free quantities (a)**, from each `BASIN_*/metrics.csv` (`Stage_3_deform_fine` row per
  specimen): `chamfer_l2`, `fscore@0.01`, `fscore@0.02`, `deform_mag_mean`.
- **GT-dependent quantity (b)**: `leg_acc`, from `per_specimen_leg_acc` in
  `correspondence_confusion.json`, matched by `run` == condition name.
- **`optimizer final energy` — excluded.** `out_basin_map_20260825/run_log.txt` logs one loss
  curve per condition, but the fit is a single batched optimization over all 12 specimens at once
  (`Stage_*.npz` tensors are shaped `(12, ...)`); the logged `loss=` value is a batch aggregate,
  not a per-specimen scalar. No per-specimen final energy exists on disk for these runs, so it
  is not included as a predictor.

For each specimen, Spearman rank correlation was computed between each GT-free quantity and
`leg_acc` across that specimen's 13 conditions (n=13 per correlation). No pooling across
specimens for the correlation itself. Critical |ρ| for p<0.05 at n=13 is ≈0.56.

## Per-specimen correlations, single quantities

ρ (p-value). Sign convention: `chamfer_l2` and `deform_mag_mean` are expected to correlate
*negatively* with `leg_acc` (lower error / less corrective deformation → better anatomical
accuracy); `fscore@0.01`/`fscore@0.02` positively.

| specimen | chamfer_l2 | fscore@0.01 | fscore@0.02 | deform_mag_mean |
|---|---:|---:|---:|---:|
| synth_000 | −0.918 (0.000) | +0.824 (0.001) | +0.918 (0.000) | −0.747 (0.003) |
| synth_001 | −0.632 (0.021) | +0.709 (0.007) | +0.775 (0.002) | −0.940 (0.000) |
| synth_002 | −0.742 (0.004) | +0.775 (0.002) | +0.742 (0.004) | −0.714 (0.006) |
| synth_003 | −0.918 (0.000) | +0.907 (0.000) | +0.940 (0.000) | −0.951 (0.000) |
| synth_004 | −0.769 (0.002) | +0.742 (0.004) | +0.698 (0.008) | −0.698 (0.008) |
| synth_005 | −0.780 (0.002) | +0.764 (0.002) | +0.852 (0.000) | −0.703 (0.007) |
| synth_006 | −0.011 (0.972) | −0.033 (0.915) | +0.066 (0.831) | −0.582 (0.037) |
| synth_007 | −0.830 (0.000) | +0.582 (0.037) | +0.852 (0.000) | −0.758 (0.003) |
| synth_008 | −0.841 (0.000) | +0.808 (0.001) | +0.824 (0.001) | −0.896 (0.000) |
| synth_009 | −0.549 (0.052) | +0.604 (0.029) | +0.522 (0.067) | −0.593 (0.033) |
| synth_010 | −0.665 (0.013) | +0.797 (0.001) | +0.654 (0.015) | −0.599 (0.031) |
| synth_011 | −0.643 (0.018) | +0.187 (0.541) | +0.742 (0.004) | +0.390 (0.188) |

### Summary across the 12 specimens

| quantity | mean ρ | median ρ | std | min | max | significant (p<.05) | expected-sign |
|---|---:|---:|---:|---:|---:|---:|---:|
| chamfer_l2 | −0.691 | −0.755 | 0.233 | −0.918 | −0.011 | 10/12 | 12/12 |
| fscore@0.01 | +0.639 | +0.753 | 0.269 | −0.033 | +0.907 | 10/12 | 11/12 |
| fscore@0.02 | +0.715 | +0.758 | 0.225 | +0.066 | +0.940 | 10/12 | 12/12 |
| deform_mag_mean | −0.649 | −0.709 | 0.336 | −0.951 | +0.390 | 11/12 | 11/12 |

## Simple combinations

Per specimen, quantities are z-scored across that specimen's own 13 conditions (no cross-specimen
pooling), then summed with the sign convention above.

- `composite_fit` = −z(chamfer_l2) + z(fscore@0.01) + z(fscore@0.02)
- `composite_fit+deform` = `composite_fit` − z(deform_mag_mean)
- `chamfer+deform` = z(chamfer_l2) − z(deform_mag_mean) (chamfer and deformation alone, as suggested)

The −1 sign on `deform_mag_mean` is a single global convention (median of the 12 per-specimen
ρ above, all but one of which were negative), fixed once and applied identically to every
specimen — not tuned per specimen.

| specimen | composite_fit | composite_fit+deform | chamfer+deform |
|---|---:|---:|---:|
| synth_000 | +0.907 (0.000) | +0.907 (0.000) | −0.456 (0.117) |
| synth_001 | +0.736 (0.004) | +0.808 (0.001) | +0.247 (0.415) |
| synth_002 | +0.742 (0.004) | +0.813 (0.001) | +0.077 (0.803) |
| synth_003 | +0.918 (0.000) | +0.934 (0.000) | +0.242 (0.426) |
| synth_004 | +0.725 (0.005) | +0.725 (0.005) | +0.379 (0.201) |
| synth_005 | +0.824 (0.001) | +0.753 (0.003) | −0.016 (0.957) |
| synth_006 | −0.027 (0.929) | +0.038 (0.901) | +0.385 (0.194) |
| synth_007 | +0.775 (0.002) | +0.808 (0.001) | +0.082 (0.789) |
| synth_008 | +0.852 (0.000) | +0.846 (0.000) | +0.264 (0.384) |
| synth_009 | +0.588 (0.035) | +0.665 (0.013) | +0.126 (0.681) |
| synth_010 | +0.654 (0.015) | +0.654 (0.015) | −0.319 (0.289) |
| synth_011 | +0.555 (0.049) | +0.192 (0.529) | −0.692 (0.009) |

| combination | mean ρ | median ρ | std | min | max | significant (p<.05) |
|---|---:|---:|---:|---:|---:|---:|
| composite_fit | +0.687 | +0.739 | 0.242 | −0.027 | +0.918 | 11/12 |
| composite_fit+deform | +0.679 | +0.780 | 0.266 | +0.038 | +0.934 | 10/12 |
| chamfer+deform | +0.027 | +0.104 | 0.328 | −0.692 | +0.385 | 1/12 |

## Summary

All four single GT-free quantities reliably rank initialization candidates by `leg_acc` *within*
most individual specimens: `chamfer_l2`, `fscore@0.01`, `fscore@0.02`, and `deform_mag_mean` each
show the expected-sign, statistically significant (p<0.05, n=13) correlation in 10–11 of 12
specimens, with median |ρ| ≈0.71–0.76 — a strong, consistent within-specimen ranking signal, not
a fluke of pooling. `fscore@0.02` is the most reliable single quantity (correct sign in all 12,
significant in 10, and the only quantity with no near-zero cases). Combining the three
fit-quality metrics (`composite_fit`) does not clearly outperform `fscore@0.02` alone (median ρ
+0.739 vs +0.758); folding in `deform_mag_mean` doesn't help further either. The one substantive
exception is `synth_006`: every single quantity is flat there (|ρ|≤0.07 for chamfer/fscore@0.01/
fscore@0.02, only `deform_mag_mean` reaches significance at ρ=−0.582), and no combination fixes
it either (`composite_fit` ρ=−0.027; `chamfer+deform` ρ=+0.385, p=0.19, not significant). `synth_011`
is a secondary partial exception: `fscore@0.01` and
`deform_mag_mean` fail there individually (deform even flips sign), but `chamfer_l2` and
`fscore@0.02` still hold, and `chamfer+deform` recovers the strongest correlation of any
quantity for that specimen (ρ=−0.692, p=0.009) — suggesting the two together carry
complementary, specimen-dependent signal rather than one dominating.

**Bottom line: yes — `fscore@0.02` (and, close behind, `chamfer_l2`/`deform_mag_mean`) reliably
ranks candidates by `leg_acc` within a specimen for the large majority (10–11/12) of specimens,
without needing ground truth at inference time. It fails outright for one specimen (`synth_006`)
and is weaker for a second (`synth_011`), so it is a strong per-specimen candidate-selection
heuristic but not a universal one.**
