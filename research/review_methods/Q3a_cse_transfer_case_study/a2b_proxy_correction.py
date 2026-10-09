"""Q3a / A2b (RULES_AUDIT R9): correct real-scan proxy leg accuracies for the proxy's own error.

Model: observed agreement a_obs = a*q + (1-a)*eps, where q = proxy leg accuracy (synthetic gate) and
eps = chance that a proxy error lands exactly on the network's wrong leg ((1-q)/5, five other legs).
Solved for a. Reported for q = gate mean (0.892) and gate p10 (0.836), i.e. a band, because the
proxy is probably somewhat worse on real scans (fewer placed joints, expert-vs-pivot offsets).
"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
gate = json.load(open(os.path.join(HERE, "out", "A2_gate_synthetic.json")))
qs = {"q_mean": gate["leg"]["mean"]}
import numpy as np
qs["q_p10"] = float(np.percentile(gate["leg"]["per_specimen"], 10))
a2 = json.load(open(os.path.join(HERE, "out", "A2_real_correspondence.json")))
a4 = json.load(open(os.path.join(HERE, "out", "A4_cycle_signal.json")))["real"]
def corr(a, q):
    e = (1 - q) / 5
    return float(np.clip((a - e) / (q - e), 0, 1))
out = {}
print(f"{'specimen':<14} {'CSE targets obs':>16} {'corrected':>18} {'per-point obs':>14} {'corrected':>18}")
for sid in sorted(a2):
    o1, o2 = a2[sid]["cse"]["leg_acc"], a4[sid]["leg_acc_all"]
    c1 = [corr(o1, q) for q in qs.values()]; c2 = [corr(o2, q) for q in qs.values()]
    out[sid] = dict(cse_targets_obs=o1, cse_targets_corr=c1, per_point_obs=o2, per_point_corr=c2)
    print(f"{sid[:14]:<14} {o1:16.2f} {c1[0]:8.2f}-{c1[1]:.2f} {o2:14.2f} {c2[0]:8.2f}-{c2[1]:.2f}")
json.dump(dict(q=qs, rows=out), open(os.path.join(HERE, "out", "A2b_proxy_correction.json"), "w"), indent=1)
