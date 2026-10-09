# P0: an independent real-articulation corpus

Written 2026-10-07, before any fit in this folder ran.

## Why

S2's REALPOSE / REALSCALE conditions and Q2's regimes take real articulation from the 33 JAB fits,
i.e. from only **11 specimens**, which are also the programme's final test set. Two problems:
too few poses to represent real pinned-specimen articulation, and leakage if those poses ever feed
training. Synthetic-to-real transfer for parametric body models works when the synthetic pose
distribution is drawn from a large corpus of real poses rather than from random sampling (SURREAL,
AGORA; BEDLAM, Black et al. CVPR 2023: regressors trained only on synthetic data reach SOTA on real
benchmarks when the synthetic distribution is realistic). P0 builds that corpus for ants.

## Data

`/hpcwork/nao48500/worker_ALT` (757 processed real worker scans) minus **every specimen of the 11 JAB
genera** (100 specimens): 657 specimens, 161 genera. Genus-level exclusion, not just specimen-level,
because pinned posture may be genus-typical.

**Split (genus-disjoint, seed 20261007, fixed before fitting):** genera shuffled, ~80% of specimens
-> `pose_train` (pose library for any future training), ~20% -> `pose_eval` (articulation source for
the confirmatory S2 replication and Q2). Written to `out/split.json` by `make_split.py`.

## Fits

D1_PROD (PROTOCOL §2, JAB A_prod recipe unchanged), seed 0 for all 657. Seeds 1 and 2 for a fixed
random subset of 48 specimens (seed 20261007) to measure fitted-pose stability.

## Measurements

1. Articulation per specimen: the A7 statistic (median |leg bend - rest|) on FK joints of the fit.
2. **Validity of fitted articulation (gate):** on the 11 JAB specimens, A_prod fitted-bend vs
   expert-bend (A7). Reported as Spearman rho and median signed difference. The fitted value is
   used as an articulation proxy only if rho >= 0.5; otherwise P0 reports the distribution as
   "fitted articulation" without claiming it equals true articulation.
3. Stability: per-specimen SD of the statistic over seeds 0-2 on the 48-specimen subset, vs the
   between-specimen SD. Usable if within/between SD ratio <= 0.5.
4. Whether JAB is typical: where the 11 JAB specimens' fitted articulation falls in the P0
   distribution (percentiles).

## Use downstream (fixed now)

- **S2-confirm:** REALPOSE / REALSCALE / REALALL rebuilt from `pose_eval` fits; same metrics and the
  same H-artic reading as S2. S2 (JAB-derived) becomes the disclosed pilot.
- **Q2-confirm:** Q2's regime rule re-applied to S2-confirm's output; Q2 on JAB poses becomes the
  pilot (DEVIATIONS D5).
- **Training (Q4, if built):** poses only from `pose_train`. Never from `pose_eval` or JAB genera.
