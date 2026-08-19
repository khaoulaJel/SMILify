# Task 5 — hierarchical (subfamily → genus) vs flat genus retrieval

## Verdict: REJECT. The existing subfamily → genus hierarchy does not improve genus retrieval
under honest end-to-end evaluation, and the one construction that looked free (route by the
label of the flat classifier's own nearest neighbour) is not actually a second classifier: it is
**provably identical** to the flat baseline, by construction, on every fold. The one construction
that is a genuinely different second classifier (nearest-subfamily-centroid routing) makes genus
accuracy significantly **worse** (17.2% → 5.7%, 95% CI on the drop [−14.9, −8.4] pp).

## Data and folds

`diagnostics/absolute_scale/morphometrics_source.csv` — the cached full-corpus deliverable table
(838 specimens, already on disk; `analyse_source.py`'s own runs are not on this cluster, see
`diagnostics/reliability/REPORT.md` §5, but this CSV needs no fit outputs to re-derive features
from). Restricted to: worker corpus (the corpus with lot replication), genus with ≥3 specimens,
subfamily known (`GENUS_TO_SUBFAMILY`, `taxonomy_source.py`) → **493 specimens, 54 genera, 9
subfamilies, 31 accession lots**. All four required numbers below are computed on this identical
fold set — leave-one-accession-**lot**-out (`analyse_source.accession_lot`/`loo_1nn`'s protocol),
not leave-one-specimen-out, for the reason already established in this project (same-lot
specimens are near-replicates and inflate everything). 2 of the 9 subfamilies
(Proceratiinae, Pseudomyrmecinae) have only one genus present in this fold set — flagged because
that makes arm 3 partly 100%-by-construction there, not signal, exactly the trap this task warns
against.

Code: `diagnostics/absolute_scale/hier_taxonomy.py`, reusing `analyse_source.py`'s `pcs`,
`accession_lot`, `perm_test`, `features` unchanged. Reproduce with
`python diagnostics/absolute_scale/hier_taxonomy.py` (conda env `pytorch3d`; ~75s, no GPU, no
fit outputs needed). Raw numbers: `diagnostics/absolute_scale/out/hier_taxonomy.json`.

## The four required numbers (lot-blind, n=493, npc=10)

| # | Test | Acc | Null | p | Lift |
|---|---|---|---|---|---|
| 1 | **Flat genus baseline** (features → genus, 1-NN) | 17.2% | 3.1% | <0.0001 | 5.6x |
| 2 | Subfamily prediction (features → subfamily, 1-NN) | 47.7% | 28.3% | <0.0001 | 1.7x |
| 3 | Genus \| **true** subfamily (diagnostic, not a system) | 31.0% | 17.0% | <0.0001 | 1.8x |
| 4 | **End-to-end genus, NN-routed** (features → predicted subfamily → restricted genus 1-NN) | 17.2% | 6.7% | <0.0001 | 2.6x |
| 2b | Subfamily via nearest centroid (a genuinely different top-stage classifier) | 21.7% | 7.0% | <0.0001 | 3.1x |
| 4b | **End-to-end genus, centroid-routed** | 5.7% | 4.2% | 0.045 | 1.3x |

Arm 3 (31.0%) beats arm 1 (17.2%) by a wide margin — and that margin is exactly the artefact the
task instructions call out, not a finding: it is computed by handing the classifier the *true*
subfamily as a candidate filter, information a real system does not have at inference time.
It is reported because the task asked for it as a diagnostic, and used for nothing else.

## Why arm 4 (NN-routed) is exactly, not approximately, equal to arm 1

This is worth stating precisely because "no measurable difference" and "cannot possibly differ"
are different claims, and only the second one is true here. Arm 4's subfamily router (arm 2)
predicts subfamily as **the subfamily label of the nearest neighbour** in the same PCA feature
space and metric the genus stage also uses. Call that neighbour `j*` — it is, by definition, the
single closest training point to the query in that space. Restricting the genus search to
"candidates sharing the query's predicted subfamily" can never exclude `j*`, because `j*`'s own
subfamily *is* the predicted subfamily. So the restricted search space always still contains the
globally closest point, and 1-NN over a superset that is guaranteed to contain the global nearest
neighbour always returns that same nearest neighbour. Arm 4 = arm 1 on every single query, not
just on average — confirmed empirically (Δacc = +0.0 pp, bootstrap CI width 0.0). **Any
"hierarchy" built by routing with the same feature space and the same distance metric as the leaf
classifier is a no-op by construction**, regardless of the data. This is the actual answer to "is
there something for the hierarchy to add here" for that specific construction: no, not
structurally.

Arm 4b breaks that identity on purpose, by using a *different* top-stage classifier (nearest
centroid instead of nearest neighbour) that is not guaranteed to route through the flat nearest
neighbour. It is therefore the only one of the two end-to-end constructions that could actually
test whether the taxonomy hierarchy helps.

## Required comparison: end-to-end vs flat, same folds, paired bootstrap over lots (n=2000, 95% CI)

| Arm | Δacc vs flat | 95% CI | Verdict |
|---|---|---|---|
| 4 (NN-routed) | +0.0 pp | [+0.0, +0.0] | no difference (provably identical, see above) |
| 4b (centroid-routed) | **−11.5 pp** | **[−14.9, −8.4]** | **regression, CI excludes zero** |

Per this project's decision rule (same statistical bar as `reliability/REPORT.md`: a
paired-bootstrap 95% CI on the delta must exclude zero, in the improving direction, to count),
neither construction clears the bar for "improvement." One is structurally incapable of ever
clearing it; the other clears it decisively in the wrong direction.

## Why centroid-routing hurts rather than merely failing to help

Centroid routing is markedly worse at its own job than NN routing (21.7% vs 47.7% subfamily
accuracy) — a subfamily is not a single cluster in this feature space, it's whatever a genus-level
1-NN signal implies about proximity, and collapsing 54 genera into 9 centroids throws most of that
structure away before the genus stage ever runs. Every subfamily misroute at the top of arm 4b
becomes an automatic genus miss at the bottom (the true genus's specimens are then not even in
the candidate pool), which is why 4b's genus accuracy (5.7%) falls even below what its own weak
21.7% subfamily accuracy would suggest a naive best case for a within-subfamily-only genus signal
that is itself no stronger than arm 1's overall signal.

## Bottom line

The subfamily → genus hierarchy, evaluated the way the task specified — end-to-end, on the exact
lot-blind folds the flat baseline uses, with the "genus given the *true* subfamily" number kept
separate as a diagnostic rather than sold as the result — does not improve genus retrieval on this
corpus with either construction tried. `hier_taxonomy.py` is written to be re-run unchanged if a
larger or differently-composed corpus becomes available (it needs only the cached
`morphometrics_source.csv`, not fit outputs); rerunning it is the natural next step, not new code.
