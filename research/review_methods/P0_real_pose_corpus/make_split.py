"""P0 genus-disjoint split of worker_ALT minus all JAB genera. Writes out/split.json and chunk dirs of
symlinks (<= 48 meshes each) for the fit array, plus the 48-specimen stability subset."""
import json, os, random
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SRC = "/hpcwork/nao48500/worker_ALT"
OUT = "/hpcwork/nao48500/review_methods/P0"
jab = set(json.load(open(os.path.join(REPO, "diagnostics/joint_alignment_benchmark/data/gt_joints_fitframe.json"))))
jg = {s.split("_")[0] for s in jab}
specs = sorted(f[:-len("_processed.obj")] for f in os.listdir(SRC) if f.endswith("_processed.obj"))
specs = [s for s in specs if s.split("_")[0] not in jg]
genera = sorted({s.split("_")[0] for s in specs})
rng = random.Random(20261007)
rng.shuffle(genera)
train, ev = [], []
for g in genera:                      # greedy fill so pose_eval reaches ~20% of SPECIMENS
    members = [s for s in specs if s.split("_")[0] == g]
    (ev if len(ev) < 0.2 * len(specs) else train).extend(members)
stab = sorted(rng.sample(specs, 48))
os.makedirs(OUT, exist_ok=True)
chunks = [specs[i:i + 48] for i in range(0, len(specs), 48)]
for k, ch in enumerate(chunks):
    d = os.path.join(OUT, "chunks", f"c{k:02d}")
    os.makedirs(d, exist_ok=True)
    for s in ch:
        dst = os.path.join(d, f"{s}_processed.obj")
        if not os.path.exists(dst):
            os.symlink(os.path.join(SRC, f"{s}_processed.obj"), dst)
d = os.path.join(OUT, "chunks", "stab48")
os.makedirs(d, exist_ok=True)
for s in stab:
    dst = os.path.join(d, f"{s}_processed.obj")
    if not os.path.exists(dst):
        os.symlink(os.path.join(SRC, f"{s}_processed.obj"), dst)
json.dump(dict(excluded_genera=sorted(jg), n=len(specs), pose_train=train, pose_eval=ev, stab48=stab,
               n_chunks=len(chunks), genera_train=len({s.split('_')[0] for s in train}),
               genera_eval=len({s.split('_')[0] for s in ev})),
          open(os.path.join(HERE, "out", "split.json"), "w"), indent=1)
print(f"{len(specs)} specimens: pose_train {len(train)} ({len({s.split('_')[0] for s in train})} genera), "
      f"pose_eval {len(ev)} ({len({s.split('_')[0] for s in ev})} genera); {len(chunks)} chunks; stab48")
print("genus overlap train/eval:", len({s.split('_')[0] for s in train} & {s.split('_')[0] for s in ev}))
