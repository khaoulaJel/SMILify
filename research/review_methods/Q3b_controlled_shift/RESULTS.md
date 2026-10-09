# Q3b results (pilot tier): controlled shifts and controlled target corruption

Rules fixed in PREREGISTRATION.md before generation; changes in ../DEVIATIONS.md D2-D5. This tier
takes real articulation from the 11 JAB specimens and is the **pilot** (D5). The confirmatory tier
repeats S2 on articulation from 136 independent worker scans of genera never in JAB
(`../P0_real_pose_corpus/`). Outputs: `out/S2_shift_battery.json`, `out/S1_readout.json`.

## S2: which shift reproduces the real failure? (C3 network, inference only)

| condition | median bend | leg accuracy [95% CI] | wrong side | wrong number | non-leg | cycle median | geodesic / sqrt(area) |
|---|---|---|---|---|---|---|---|
| C0 (training distribution) | 10.5 | 0.907 [0.886, 0.926] | 0.009 | 0.049 | 0.034 | 0.0187 | 0.026 |
| ART15 | 15.9 | 0.771 | 0.046 | 0.104 | 0.079 | 0.0218 | 0.057 |
| ART20 | 21.4 | 0.581 | 0.072 | 0.226 | 0.121 | 0.0271 | 0.188 |
| ART25 | 24.2 | 0.484 | 0.108 | 0.274 | 0.134 | 0.0293 | 0.345 |
| **ART-real** | 29.2 | **0.353** [0.313, 0.396] | 0.127 | 0.338 | 0.182 | 0.0371 | 0.533 |
| ART35 | 32.5 | 0.320 | 0.124 | 0.362 | 0.194 | 0.0420 | 0.603 |
| ARTLIM-real (joint limits) | 28.6 | 0.317 | 0.127 | 0.331 | 0.226 | 0.0459 | 0.590 |
| ROT 10 / 20 / 30 deg | 10.6-11.2 | 0.880 / 0.865 / 0.842 | 0.013-0.014 | 0.067-0.092 | | 0.018-0.022 | 0.028-0.035 |
| SCL 0.2 / 0.3 | 12.3 / 14.4 | 0.887 / 0.826 | | | | | |
| SHP 1.5 / 2.0 | 10.8 / 11.8 | 0.863 / 0.784 | | | | | |
| DEG noise / holes / distal removed | 10.5 | 0.895 / 0.879 / 0.906 | | | | | |
| **REALPOSE** | 25.5 | **0.732** [0.694, 0.769] | 0.030 | 0.144 | 0.095 | 0.0295 | 0.065 |
| **REALSCALE** | 22.3 | **0.665** [0.613, 0.715] | 0.028 | 0.202 | 0.105 | 0.0273 | 0.129 |
| **REALSHAPE** | 11.4 | **0.911** [0.884, 0.934] | 0.013 | 0.033 | 0.043 | 0.0115 | 0.025 |
| REALALL | 33.4 | 0.547 | 0.021 | 0.256 | 0.176 | 0.0364 | 0.269 |
| *real scans (Q3a, proxy-corrected)* | *18-38* | *0.695 median* | *0.00-0.05* | *0.10-0.37* | | *0.022-0.260* | |

**H-artic verdict, by the registered rule: PARTIAL.** Four of five criteria pass: accuracy 0.353 ≤ 0.80;
wrong-number 2.7× wrong-side; the articulation drop (0.554) exceeds every other factor (next largest
SHP2.0, 0.123); monotone over five levels. The cycle criterion misses by 0.0003 (0.0371 vs
2 × 0.0187). The threshold was not changed after the fact.

**What the table shows beyond the verdict:**
1. **Articulation is the dominant shift**, several times larger than rotation, per-joint scale,
   extreme shape or scan damage.
2. **Real articulation reproduces the real failure; real shape does not.** REALPOSE (0.732) and
   REALSCALE (0.665) bracket the real 0.695; REALSHAPE (0.911) equals the training distribution.
   This independently refutes the morphology hypothesis that Q3a already contradicted.
3. **The real error type is reproduced only by real articulation.** Isotropic ART creates left/right
   confusion (0.127) that real scans do not show; REAL* conditions match the real near-zero side
   error (0.02-0.03), with errors concentrated on leg number.
4. **Not reproduced:** the extreme cycle distances of Cephalotes (0.26) and Discothyrea (0.16). No
   synthetic condition exceeds 0.046. That residual comes from scan geometry the model cannot
   generate and is reported as unexplained.
5. REALSCALE changes measured bend too (22.3°), because per-joint translations move the pivots;
   its effect is not separable from articulation geometry and is not interpreted as "scale".

## S1: what does the fitter do with wrong correspondence? (P48, 3 seeds, D1_PROD recipe)

Median FK joint error / body-axis length (seed SD in brackets), final stage:

| targets | `full` (all stages) | `hier` (hierarchical stages only) | `prod` (none) |
|---|---|---|---|
| correct (rho 0) | **0.89% (0.02)** | 1.77% (0.07) | 2.43% (0.20) |
| ~22% of leg vertices on the wrong leg | 3.16% (0.08) | 2.25% (0.11) | |
| ~43% on the wrong leg | 4.57% (0.16) | 3.49% (0.27) | |

| registered test | result | verdict |
|---|---|---|
| H-absorb: at rho 0.4, `full` degrades H2 → S3 while `prod` improves | `full` −0.09% L (23/48 up, p 0.89): **stalls**; `prod` −0.70% L (39/48 down) | **NOT SUPPORTED** as registered |
| free channels inflate under wrong targets | deform, joint translation, scales higher than `prod` on 48/48 (p 7e-15) | consistent with absorption |
| `hier` beats `full` at rho 0.4 | 35/48, p 0.002 | supports keeping wrong targets out of surface stages |
| H-surfterm: at rho 0, `full` worse than `hier` | `full` **better**, 0.89 vs 1.77, 48/48 | **NOT SUPPORTED** |

**Reading.**
1. **Correct correspondence is worth a 63% skeleton-error reduction** (2.43 → 0.89% L, 48/48), the
   largest effect measured in this programme, and it should stay on in every stage when it is right.
2. **Its value disappears by ~20% wrong-leg targets** and turns into harm by ~40%. Real CSE targets
   fall on both sides of that line: 0-6% wrong on the good specimens, 46-75% on the bad ones
   (Q3a A2b), which matches the bimodal JAB outcome.
3. On synthetic data, absorption of wrong targets shows up as **stalled improvement**, not as active
   degradation. The real-scan degradation of Aphaenogaster (good targets, worse joints) is not
   explained by any mechanism tested here.

## Consequences for Q2/Q4

- The quantity that decides whether correspondence helps is the **fraction of wrong-leg targets**,
  and the main cause of wrong-leg targets on real scans is **articulation outside the training
  distribution**. Both point at the same intervention family: correspondence trained on realistic
  articulation, plus a specimen-level estimate of target quality to decide when to trust it.
- Confirmatory replication on P0's independent articulation corpus precedes any Q4 build.
