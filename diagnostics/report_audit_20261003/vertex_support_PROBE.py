"""Probe: vertices per leg segment of the production model, by dominant skinning weight.
Run with the pytorch3d env python from the repo root. Verified 2026-10-03: l_1_*_r ->
co 200, tr 275, fe 153, ti 82, ta 38, pt 8 (backs report finding F5)."""
import pickle, numpy as np, collections
d = pickle.load(open('3D_model_prep/OmniAnt_25PCs_joint_limited.pkl', 'rb'), encoding='latin1')
names = [str(x) for x in d['J_names']]
dom = np.asarray(d['weights']).argmax(1)
cnt = collections.Counter(names[i] for i in dom)
print({n: c for n, c in cnt.items() if n.startswith('l_1_') and n.endswith('_r')})
