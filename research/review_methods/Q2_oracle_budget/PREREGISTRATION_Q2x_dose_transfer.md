# Q2x: does the synthetic dose curve predict which specimens deployed CSE helps? (addendum)

Written 2026-10-08 after seeing ONLY the aggregate B0 vs B1 joint numbers per regime
(`out/B0_B1_early.log`), before any per-specimen relation between target quality and benefit was
computed.

Per specimen in REALPOSE and REALSCALE (33 each): x = wrong-leg fraction of the deployed CSE targets
(true labels: target position's true leg vs template vertex's leg, leg vertices of tr..pt only);
y = seed-mean joint error B1 − B0 (negative = CSE helps).

Predictions from S1/Q3d (controlled corruption, P48):
1. Spearman(x, y) > 0 (more wrong-leg targets -> less benefit), one-sided p < 0.05 in the pooled 66.
2. Specimens with x <= 0.10 improve on a majority (sign p < 0.05 if n allows); specimens with
   x >= 0.30 do not improve on a majority.
3. A logistic fit P(y < 0 | x) crosses 0.5 between x = 0.05 and x = 0.25 (S1/Q3d: break-even between
   0% and ~20%).

Caveat stated in advance: the 33 specimens per regime take their poses from 11 JAB specimens x 3
seeds, so the effective number of independent poses is ~11 per regime; the pooled test is
reported with a cluster bootstrap over source specimen as the honest CI.
