"""S2-confirm (DEVIATIONS D5): REAL* conditions rebuilt from P0 `pose_eval` fits only (136 worker scans,
45 genera, no JAB genus), seed-0 D1_PROD fits. Same construction as the pilot (make_shift_sets
.real_param_set), new seeds (780000+). C0 is linked from the pilot so the readout is self-contained."""
import glob, json, os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import make_shift_sets as mss
OUT = "/hpcwork/nao48500/review_methods/shift_sets_P0"
split = json.load(open(os.path.join(HERE, "..", "P0_real_pose_corpus", "out", "split.json")))
fits = sorted(glob.glob("/hpcwork/nao48500/review_methods/P0/runs/c??_s0/Stage_3_deform_fine.npz"))
dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT, exist_ok=True)
for j, name in enumerate(mss.REAL_FACTORS):
    d = mss.real_param_set(name, 780000 + j, dev, fit_paths=fits, wanted=set(split["pose_eval"]))
    b = mss.bend_stat(d["joints"])
    np.savez(os.path.join(OUT, f"{name}.npz"), **d, cond=name, rot_deg=0, measured_bend=b,
             factors=np.array(mss.REAL_FACTORS[name]))
    print(f"[S2-confirm] {name}: n={len(d['verts'])} from pose_eval, measured bend {b:.1f}", flush=True)
c0 = os.path.join(OUT, "C0.npz")
if not os.path.exists(c0):
    os.symlink("/hpcwork/nao48500/review_methods/shift_sets/C0.npz", c0)
