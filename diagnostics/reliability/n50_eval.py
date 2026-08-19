"""Task 2, step 3 -- SECONDARY, exploratory-only real-corpus check: lot-blind genus lift under
each weighting scheme, on the 50 real, genus-labelled worker specimens whose fit outputs already
exist on this cluster (`diagnostics/d1_n50_evidence/runs/seed0/d1/Stage_3_deform_fine.npz`, seed 0
matching this project's reference-seed convention).

THIS DOES NOT SELECT THE WINNING SCHEME. Per the approved plan, the synthetic ground-truth result
(`synth_eval.py`) is the sole basis for any verdict on the weighting schemes; this script only
checks whether the picture also looks reasonable on real data, reported separately, disagreements
included as-is.

SCOPE NOTE. The full 838-specimen corpus (`morphometrics_source.csv`) has no fit outputs on this
cluster (see the approved plan's "compute reality check") and is not attempted here. Cross-corpus
retrieval also needs the ALL_ANTS_CLEAN corpus, which this 50-specimen worker-only evidence set
does not include -- only lot-blind genus lift is checked here, not retrieval.

FOUR SCHEMES, reusing `diagnostics/reliability/weights.py`'s SAME rank-based, frozen-before-results
composite -> weight transform as the primary evaluation:
  hard50              existing `measure.quality_filter` (top 50% by global composite).
  equal               no filtering.
  continuous_global   each specimen's ROW scaled by sqrt(weight) before the 1-NN distance, so an
                       unreliable specimen contributes less to whom it is compared against (and to
                       who is compared to it) without being discarded outright.
  feature_scaling     the task's explicitly optional 4th condition ("reliability-weighted feature
                       scaling"): each COLUMN scaled by its region's mean reliability weight across
                       the corpus, so a whole unreliable block (e.g. leg_distal) contributes less
                       to every specimen's distance, not just to the unreliable specimens'.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
ABS_SCALE = os.path.join(REPO, "diagnostics", "absolute_scale")
MOON = os.path.join(REPO, "diagnostics", "moonshot")
sys.path.insert(0, REPO)
sys.path.insert(0, ABS_SCALE)
sys.path.insert(0, HERE)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import measure as ms  # noqa: E402
from calib_features import block_of  # noqa: E402
import analyse_source as asrc  # noqa: E402
import weights as W  # noqa: E402

RUN = "D1_N50_SEED0"  # symlinked to diagnostics/d1_n50_evidence/runs/seed0/d1 (gitignored, local)
MIN_N_GENUS = 3
NPC = 10


def main():
    M = ms.load_model()
    bones = ms.bone_table(M)
    tpa = ms.template_part_axes(M)

    rows, cols, dev = ms.measure_run(RUN, "worker", M, bones, tpa)
    print(f"{len(rows)} specimens measured from {RUN}")
    labels = [r["label"] for r in rows]

    sc_all, asym = ms.symmetrise(rows, cols)
    sdev_all, _ = ms.symmetrise(rows, dev, key="asym_shapedev")
    sc = asrc.select_features(sc_all, "core")
    sdev = asrc.select_features(sdev_all, "core")
    F_raw, Z, size, X = asrc.features(rows, sc, sdev)
    all_cols = sc + sdev
    assert F_raw.shape[1] == len(all_cols)

    gen = np.array([r["genus"] if r["genus"] else "?" for r in rows])
    lot = np.array([asrc.accession_lot(r["specimen"]) for r in rows])
    cnt = {g: int((gen == g).sum()) for g in set(gen)}
    keep = np.array([cnt.get(g, 0) >= MIN_N_GENUS and g != "?" for g in gen])
    print(f"after min_n>={MIN_N_GENUS} genus filter: {keep.sum()} specimens, {len(set(gen[keep]))} genera")
    if keep.sum() < 15:
        print("too few specimens survive the genus filter at n=50 -- this exploratory check is underpowered; "
              "reporting whatever the numbers are anyway, not suppressing them.")

    global_composite, _ = ms.quality_composite(rows, M)
    global_composite = {lab: float(z) for lab, z in zip(labels, global_composite)}
    part_composite = ms.part_quality_composite(rows, M)

    def run_scheme(name):
        if name in ("hard50", "equal"):
            wmap = W.measurement_weights(labels, all_cols, block_of, name, global_composite, part_composite)
            w_specimen = next(iter(wmap.values()))  # same for every column under these 2 schemes
            F = F_raw.copy()
            k = keep & (w_specimen > 0)
        elif name == "continuous_global":
            w_specimen = W.rank_weight(np.array([global_composite[lab] for lab in labels]))
            F = F_raw * np.sqrt(w_specimen)[:, None]
            k = keep
        elif name == "feature_scaling":
            # per-column reliability = mean rank-weight of that column's region composite across
            # the corpus (falling back to the specimen's global composite where the region lacks
            # support) -- a whole block scaled down together, not a per-specimen weight.
            col_scale = np.array(
                [
                    W.rank_weight(
                        np.array([part_composite.get(block_of(c), {}).get(lab, global_composite[lab]) for lab in labels])
                    ).mean()
                    for c in all_cols
                ]
            )
            F = F_raw * col_scale[None, :]
            k = keep
        else:
            raise ValueError(name)
        if k.sum() < 15:
            return dict(n=int(k.sum()), n_genus=len(set(gen[k])), acc=float("nan"), null=float("nan"), lift=float("nan"))
        Fk = asrc.pcs(F[k], min(NPC, F[k].shape[1]))
        a, m, s, p = asrc.perm_test(Fk, gen[k], groups=lot[k])
        return dict(n=int(k.sum()), n_genus=len(set(gen[k])), acc=a, null=m, p=p, lift=a / max(m, 1e-9))

    print(f"\n{'scheme':<20}{'n':>5}{'genera':>8}{'lot-blind acc':>15}{'null':>8}{'LIFT':>7}")
    results = {}
    for scheme in ("hard50", "equal", "continuous_global", "feature_scaling"):
        r = run_scheme(scheme)
        results[scheme] = r
        print(
            f"{scheme:<20}{r['n']:>5}{r['n_genus']:>8}{100 * r.get('acc', float('nan')):>14.1f}%"
            f"{100 * r.get('null', float('nan')):>7.1f}%{r.get('lift', float('nan')):>7.1f}"
        )

    import json

    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    json.dump(results, open(os.path.join(HERE, "out", "n50_eval.json"), "w"), indent=1)
    print(f"\nwrote {os.path.join(HERE, 'out', 'n50_eval.json')}")
    print(
        "\nReminder: this is an exploratory sanity check on n=50 real specimens ONLY (no cross-corpus "
        "retrieval -- ALL_ANTS_CLEAN is not in this evidence set). It does not select a winning scheme; "
        "the synthetic ground-truth result in synth_eval.py does."
    )


if __name__ == "__main__":
    main()
