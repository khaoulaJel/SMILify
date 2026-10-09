"""Add provenance columns (work-package rule 2: every number names config, specimen set, seed) to the
deliverable CSVs. Reads the raw scorer output in run_cpu1/, writes the annotated copies to this folder.
Values are copied unchanged; only columns are added in front."""

import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROV = {
    "config": "D1_PROD (JAB cfg/A_prod.yaml md5 1efa7a14; model OmniAnt_25PCs_joint_limited.pkl md5 08b69daf)",
    "specimen_set": "atta20_WOLO (20 A. vollenweideri workers, /hpcwork/nao48500/atta20)",
    "seed": "0",
    "run": "run_cpu1 (CPU, batch 1 per specimen; DEVIATIONS.md D1+D2)",
}
for name in ("stages.csv", "stages_all20.csv"):
    with open(os.path.join(HERE, "run_cpu1", name)) as f:
        rows = list(csv.DictReader(f))
    keys = list(PROV) + list(rows[0])
    with open(os.path.join(HERE, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({**PROV, **r})
    print(f"{name}: {len(rows)} rows, {len(keys)} columns")
