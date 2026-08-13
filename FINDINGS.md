
## Multi-Seed Results Summary (Seeds 0, 1, 2)

| Metric | Baseline (Mean ± Std) | Gentle (Mean ± Std) | Delta (Δ) |
| :--- | :--- | :--- | :--- |
| **Precision@0.01** | 0.8521 ± 0.0022 | 0.8441 ± 0.0043 | -0.0080 (-0.94%) |
| **Penetration Depth** | 0.0294 ± 0.0010 | 0.0174 ± 0.0002 | -0.0120 (-40.7%) |
| **Guardrail Violations (>3% precision drop)** | — | — | 6.0 / 50 specimens avg (12%) |

### Key Takeaways
- **Penetration Reduction**: Gentle regularization cut inter-part penetration depth by ~40.7% with high multi-seed stability (std dropped to 0.0002).
- **Precision Impact**: Precision@0.01 dropped marginally by < 1% overall.
- **Guardrail Analysis**: An average of 6 out of 50 specimens (12%) experienced a >3% drop in precision across seeds (10 in seed 0, 6 in seed 1, 2 in seed 2).

### Outlier & Guardrail Regression Analysis
- **Top Recurring Regressions (2/3 seeds):**
  1. `Anochetus_risii_CASENT0877608_processed.obj` (Trap-jaw geometry)
  2. `Atopomyrmex_mocquerysi_CASENT0744784_processed.obj`
  3. `Pseudomyrmex_cf.solis_CASENT0744065_processed.obj` (Slender body plan)
  4. `Syllophopsis_sechellensis_OKENT0105212_processed.obj`
- **Root Cause:** Strict inter-part collision boundaries prevent slender articulated structures from intersecting, yielding a minor precision trade-off (<1% cohort-wide) for significantly improved physical mesh validity.
