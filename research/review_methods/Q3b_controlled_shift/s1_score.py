"""Q3b S1 readout against the pre-registered H-absorb / H-surfterm rules (PREREGISTRATION.md).

Per arm x seed x stage (H2, S3): per-specimen median FK joint error / body-axis length
(common/synthetic_skeleton_score.py; fits reload-verified), and free channels. Seed-mean per
specimen, then paired sign / Wilcoxon / t over the 48 specimens.
"""
import json, os, sys
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import synthetic_skeleton_score as sss
R = "/hpcwork/nao48500/review_methods/Q3b_S1/runs"
ARMS = ["prod"] + [f"{a}_rho{r}" for a in ("full", "hier") for r in ("0.0", "0.2", "0.4")]
KEYS = ["med", "mean", "deform_rms", "betas_trans", "log_scales"]

def load(arm, stage):
    per = []
    for s in (0, 1, 2):
        p = f"{R}/{arm}_s{s}_hier/H2_joint.npz" if stage == "H2" else f"{R}/{arm}_s{s}/Stage_3_deform_fine.npz"
        per.append(sss.score_fit(p))
    sids = sorted(per[0])
    return sids, {k: np.array([np.mean([p[sid][k] for p in per]) for sid in sids]) for k in KEYS}, \
           np.array([[p[sid]["med"] for sid in sids] for p in per])

def paired(a, b):
    d = b - a
    return dict(mean_d=float(d.mean()), n_pos=int((d > 0).sum()), n=len(d),
                sign_p=float(stats.binomtest(int((d > 0).sum()), len(d)).pvalue),
                wil_p=float(stats.wilcoxon(d).pvalue), t_p=float(stats.ttest_rel(b, a).pvalue))

res = {}
for arm in ARMS:
    for st in ("H2", "S3"):
        sids, m, seeds = load(arm, st)
        res[(arm, st)] = m
        sd = float(np.std(seeds, axis=0).mean())
        print(f"{arm:<13} {st}: joint err median {100*np.median(m['med']):.2f}% L  (seed SD {100*sd:.2f})  "
              f"deform {np.mean(m['deform_rms']):.4f}  btrans {np.mean(m['betas_trans']):.4f}  lscale {np.mean(m['log_scales']):.3f}", flush=True)

out = {}
def show(name, a, b, key="med"):
    r = paired(a, b); out[name] = r
    print(f"  {name:<44} d={100*r['mean_d']:+.2f}% L  {r['n_pos']}/{r['n']} up  sign {r['sign_p']:.2g}  wil {r['wil_p']:.2g}  t {r['t_p']:.2g}")

print("\nH-absorb (rho 0.4):")
show("full_rho0.4 H2 -> S3", res[("full_rho0.4", "H2")]["med"], res[("full_rho0.4", "S3")]["med"])
show("prod H2 -> S3", res[("prod", "H2")]["med"], res[("prod", "S3")]["med"])
for k in ("deform_rms", "betas_trans", "log_scales"):
    show(f"S3 {k}: prod -> full_rho0.4", res[("prod", "S3")][k], res[("full_rho0.4", "S3")][k], k)
show("S3: hier_rho0.4 -> full_rho0.4", res[("hier_rho0.4", "S3")]["med"], res[("full_rho0.4", "S3")]["med"])
print("\nH-surfterm control (rho 0): positive d = full worse than hier")
show("S3: hier_rho0.0 -> full_rho0.0", res[("hier_rho0.0", "S3")]["med"], res[("full_rho0.0", "S3")]["med"])
print("\ndose: full vs hier at each rho, and each vs prod (S3)")
for r in ("0.0", "0.2", "0.4"):
    show(f"S3: prod -> full_rho{r}", res[("prod", "S3")]["med"], res[(f"full_rho{r}", "S3")]["med"])
    show(f"S3: prod -> hier_rho{r}", res[("prod", "S3")]["med"], res[(f"hier_rho{r}", "S3")]["med"])
    show(f"H2: prod -> full_rho{r}", res[("prod", "H2")]["med"], res[(f"full_rho{r}", "H2")]["med"])
os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
json.dump({"contrasts": out, "arms": {f"{a}|{s}": {k: v.tolist() for k, v in m.items()} for (a, s), m in res.items()}},
          open(os.path.join(HERE, "out", "S1_readout.json"), "w"), indent=1)
