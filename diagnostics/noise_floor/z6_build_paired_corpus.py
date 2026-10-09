"""Z6 step 1 -- draw a PAIRED worker corpus: 25 species x 2 conspecific specimens.

WHY A PAIRED CORPUS
After Z1-Z5 the worker residual (gen@20/spread ~ 0.60) is not explained by any parameterisation or
capacity hypothesis, and Z4 showed the free-form field is predominantly per-specimen noise. The
open question is no longer "what else can we fix" but "what can this metric ATTAIN on real worker
scans" -- i.e. is 0.60 a failure, or is it the floor?

Two specimens of the SAME SPECIES are near-identical animals. If the pipeline cannot predict one
from the other, the residual is per-specimen noise and no fitter change will move it. If it can,
then 0.60 reflects genuine between-species variation and the model is simply under-capacity for
the breadth of Formicidae -- a completely different project.

`bench50_clean` cannot answer this: it is diversity-built, 48 species in 50 specimens, only 2
conspecific pairs. This draws 25 species x 2 = 50 specimens, so n matches bench50_clean exactly and
gen/spread's leave-one-out normaliser N/(N-1) is identical (REFUTE1's R2 flagged that this differs
between corpora of different size).

SELECTION, fixed before any fit: one species per GENUS where possible, so the denominator
(population spread) stays broad and cannot inflate the ratio -- the confound that made the original
worker-vs-clean comparison unreadable. Seeded and deterministic.
"""
import argparse
import collections
import json
import os
import random
import shutil

SRC = "/hpcwork/nao48500/worker_alt_data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_species", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="diagnostics/noise_floor/paired50")
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(SRC) if f.endswith(".obj"))
    by_species = collections.defaultdict(list)
    for f in files:
        by_species["_".join(f.split("_")[:2])].append(f)
    # Only DETERMINED species count as conspecifics. Two specimens labelled `Camponotus_sp.` are
    # two unidentified members of a genus, not the same species, and `cf.`/`aff.` are explicit
    # statements of uncertain determination. Including them would put non-conspecifics into the
    # "same species" arm and bias the floor downward -- i.e. make the pipeline look worse at
    # exactly the comparison this experiment exists to make.
    def determined(sp):
        epithet = sp.split("_", 1)[1] if "_" in sp else ""
        return epithet and not epithet.startswith(("sp.", "cf.", "aff.", "nr."))

    cand = {k: sorted(v) for k, v in by_species.items() if len(v) >= 2 and determined(k)}
    print(f"species with >=2 specimens: {sum(1 for v in by_species.values() if len(v) >= 2)}; "
          f"of those, DETERMINED to species: {len(cand)}")

    rng = random.Random(args.seed)
    by_genus = collections.defaultdict(list)
    for sp in sorted(cand):
        by_genus[sp.split("_")[0]].append(sp)

    chosen, used_genera = [], set()
    for genus in sorted(by_genus):                      # one species per genus first
        chosen.append(rng.choice(sorted(by_genus[genus])))
        used_genera.add(genus)
    rng.shuffle(chosen)
    chosen = sorted(chosen)[:args.n_species] if len(chosen) >= args.n_species else chosen
    rng2 = random.Random(args.seed + 1)
    chosen = sorted(rng2.sample(sorted(set(chosen)), args.n_species))

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), args.out)
    os.makedirs(out, exist_ok=True)
    pairs = {}
    for sp in chosen:
        two = rng2.sample(cand[sp], 2)
        pairs[sp] = sorted(two)
        for f in pairs[sp]:
            shutil.copy(os.path.join(SRC, f), os.path.join(out, f))

    genera = {s.split("_")[0] for s in chosen}
    print(f"drew {len(chosen)} species x 2 = {2*len(chosen)} specimens across {len(genera)} genera")
    assert len(genera) == len(chosen), "one species per genus was not achieved -- check selection"
    for sp in chosen:
        print(f"  {sp:<34} {pairs[sp][0]}  |  {pairs[sp][1]}")
    json.dump(pairs, open(os.path.join(out, "pairs.json"), "w"), indent=2)
    print(f"\nwrote {out}  ({len(os.listdir(out)) - 1} .obj + pairs.json)")


if __name__ == "__main__":
    main()
