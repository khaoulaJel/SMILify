"""X1 -- did the scale cap fix the anterior anatomy, or only the scale statistics?

Bar fixed in PREREGISTRATION_X1_scalecap_mechanism.md BEFORE the runs.

Measures BOTH quantities side by side on the same fits, which is the whole design:

  PROXY     exp(log_beta_scales).max()/min() per anterior group -- the statistic the shipped
            scale_cap A/B passed its third decision condition on. Reused UNCHANGED from
            diagnostics/morphometrics/ab_scale_cap/integrity_and_anterior_check.py:41.
            It is the POSITIVE CONTROL: if it does not reproduce, we have not reproduced the
            shipped intervention and the mechanism reading is VOID.

  MECHANISM per-group `deform_verts` magnitude RELATIVE TO THORAX -- REPORT.md section 6.10's
            head-carried ratio (head 0.72x, antenna 1.72x). Never measured for the scale cap.
            Head is the endpoint.

GROUPING PROVENANCE. probe_22_anterior_blindspot_PROBE.py, which produced the 0.72x figure, was
never committed (gitignored probe_*.py) and only its _out.txt survives. The grouping is therefore
taken from fitter_3d/part_groups.py PART_GROUPS_COARSE, which is derived programmatically from
kintree_table and whose joint counts match probe_22's own CLAIM 1 table exactly (thorax 1,
gaster 5, waist 4, head 1, mandible 2, antenna 6, leg 36). That equality is asserted at runtime.
"""
import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "fitter_3d"))
sys.path.insert(0, REPO)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import config  # noqa: E402
from part_groups import PART_GROUPS_COARSE  # noqa: E402

# probe_22's own CLAIM 1 counts, asserted so a grouping drift cannot pass silently.
PROBE22_COUNTS = {"thorax": 1, "gaster": 5, "waist": 4, "head": 1,
                  "mandible": 2, "antenna": 6, "legs": 36}
ANTERIOR = ("head", "mandible", "antenna")


def anterior_ratio(log_beta_scales):
    """exp(log_beta_scales) range (max/min across the 3 scale axes) per anterior group.

    Copied verbatim from integrity_and_anterior_check.py:41 so the PROXY is the same statistic the
    shipped A/B passed on, not a re-derivation that could differ.
    """
    s = np.exp(log_beta_scales)
    out = {}
    for part in ANTERIOR:
        vals = s[PART_GROUPS_COARSE[part]].reshape(-1)
        out[part] = float(vals.max() / max(vals.min(), 1e-9))
    return out


def vertex_groups(dd):
    """Map each vertex to an anatomical group by DOMINANT SKINNING WEIGHT -- the same rule
    build_vertex_labels uses, so group membership is consistent with the rest of the record."""
    w = np.asarray(dd["weights"])
    dom = w.argmax(axis=1)
    j2g = {}
    for g, joints in PART_GROUPS_COARSE.items():
        for j in joints:
            j2g[j] = g
    groups = {g: np.where(np.isin(dom, PART_GROUPS_COARSE[g]))[0] for g in PART_GROUPS_COARSE}
    return groups


def three_tests(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 3:
        return {"n": n, "sign_p": float("nan"), "wilcoxon_p": float("nan")}
    n_better = int((d > 0).sum())
    n_eff = int((d != 0).sum())
    out = {"n": n, "n_higher": n_better, "mean_delta": float(d.mean())}
    out["sign_p"] = float(stats.binomtest(n_better, n_eff, 0.5).pvalue) if n_eff else float("nan")
    try:
        out["wilcoxon_p"] = float(stats.wilcoxon(d).pvalue)
    except ValueError:
        out["wilcoxon_p"] = float("nan")
    return out


def load_arm(run_dir, groups, stage="Stage_3_deform_fine"):
    p = os.path.join(REPO, run_dir, f"{stage}.npz")
    if not os.path.exists(p):
        raise SystemExit(f"VOID: {p} not found -- the arm did not persist deform_verts")
    d = np.load(p)
    if "deform_verts" not in d:
        raise SystemExit(f"VOID: {p} has no 'deform_verts' (keys: {list(d.keys())})")
    dv = np.asarray(d["deform_verts"])          # (N, V, 3)
    lbs = np.asarray(d["log_beta_scales"])      # (N, J, 3)
    n = dv.shape[0]

    ratios, proxies, thorax_abs = [], [], []
    for i in range(n):
        mag = np.linalg.norm(dv[i], axis=1)     # (V,)
        th = float(mag[groups["thorax"]].mean())
        thorax_abs.append(th)
        r = {}
        for g, idx in groups.items():
            if g == "thorax" or len(idx) == 0:
                continue
            r[g] = float(mag[idx].mean() / th) if th > 1e-12 else float("nan")
        ratios.append(r)
        proxies.append(anterior_ratio(lbs[i]))
    return {"n": n, "ratios": ratios, "proxies": proxies, "thorax_abs": thorax_abs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm_a", default="diagnostics/moonshot/runs/X1_A_nocap")
    ap.add_argument("--arm_b", default="diagnostics/moonshot/runs/X1_B_scalecap")
    ap.add_argument("--out_dir", default="diagnostics/anterior_mechanism/out_X1")
    args = ap.parse_args()
    out_dir = os.path.join(REPO, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(REPO, config.SMAL_FILE), "rb") as fh:
        u = pickle._Unpickler(fh)
        u.encoding = "latin1"
        dd = u.load()

    # ---------------------------------------------------------------- grouping provenance
    counts = {g: len(j) for g, j in PART_GROUPS_COARSE.items()}
    ok_counts = all(counts[g] == c for g, c in PROBE22_COUNTS.items())
    print(f"[GROUPING] PART_GROUPS_COARSE joint counts {counts}")
    print(f"           match probe_22's CLAIM 1 table: {ok_counts}   "
          f"{'PASS' if ok_counts else 'VOID -- grouping drifted'}")
    if not ok_counts:
        raise SystemExit("VOID: grouping does not match probe_22's")

    groups = vertex_groups(dd)
    print(f"[GROUPING] vertices per group: "
          + ", ".join(f"{g} {len(i)}" for g, i in sorted(groups.items())))

    A = load_arm(args.arm_a, groups)
    B = load_arm(args.arm_b, groups)
    if A["n"] != B["n"]:
        raise SystemExit(f"VOID: arm sizes differ, A={A['n']} B={B['n']}")
    n = A["n"]
    print(f"\nspecimens: {n} per arm")

    # ---------------------------------------------------------------- CHECK 3/4
    tA, tB = float(np.mean(A["thorax_abs"])), float(np.mean(B["thorax_abs"]))
    print(f"\n[CHECK 4] thorax reference deform (absolute): A {tA:.5f}  B {tB:.5f}   "
          f"(section 6.10 range was 0.019-0.043)")
    degenerate = (tA < 1e-8) or (tB < 1e-8)
    print(f"[CHECK 3] deform_verts non-trivial: {'PASS' if not degenerate else 'VOID -- pinned at 0'}")
    if degenerate:
        raise SystemExit("VOID: deform_verts is identically zero; the ratio is undefined")

    # ---------------------------------------------------------------- PROXY (positive control)
    print("\n" + "=" * 88)
    print("PROXY -- anterior joint-scale range max/min  (the SHIPPED condition; positive control)")
    print("B should be SMALLER; the shipped A/B reported mean -4.9..-14.2%, max tail -58..-70%")
    print("=" * 88)
    print(f"{'group':<10}{'A mean':>10}{'B mean':>10}{'delta':>10}{'A max':>10}{'B max':>10}"
          f"{'max delta':>11}{'sign p':>11}")
    proxy_ok = True
    proxy_out = {}
    for g in ANTERIOR:
        a = np.array([p[g] for p in A["proxies"]])
        b = np.array([p[g] for p in B["proxies"]])
        t = three_tests(a, b)                    # a > b  => B compressed
        rel = 100 * (b.mean() - a.mean()) / a.mean()
        relmax = 100 * (b.max() - a.max()) / a.max()
        compressed = b.mean() < a.mean()
        proxy_ok &= compressed
        proxy_out[g] = {"A_mean": float(a.mean()), "B_mean": float(b.mean()), "rel_mean": rel,
                        "A_max": float(a.max()), "B_max": float(b.max()), "rel_max": relmax,
                        "sign_p": t["sign_p"], "compressed": bool(compressed)}
        print(f"{g:<10}{a.mean():>10.2f}{b.mean():>10.2f}{rel:>9.1f}%"
              f"{a.max():>10.2f}{b.max():>10.2f}{relmax:>10.1f}%{t['sign_p']:>11.2e}")
    print(f"\n[CHECK 2 -- VOIDING] proxy reproduces (B compresses on all three): "
          f"{'PASS' if proxy_ok else 'FAIL -- mechanism reading is VOID'}")

    # ---------------------------------------------------------------- MECHANISM (the endpoint)
    print("\n" + "=" * 88)
    print("MECHANISM -- per-group deform_verts relative to thorax  (section 6.10's head-carried ratio)")
    print("head must RISE toward 1.0 (was 0.72x); antenna must FALL toward 1.0 (was 1.72x)")
    print("=" * 88)
    print(f"{'group':<10}{'A':>10}{'B':>10}{'delta':>10}{'|A-1|':>9}{'|B-1|':>9}"
          f"{'closer?':>9}{'sign p':>11}")
    mech = {}
    for g in ["head", "mandible", "antenna", "gaster", "waist", "legs"]:
        if g not in A["ratios"][0]:
            continue
        a = np.array([r[g] for r in A["ratios"]])
        b = np.array([r[g] for r in B["ratios"]])
        da, db = np.abs(a - 1.0), np.abs(b - 1.0)
        t = three_tests(da, db)                  # da > db  => B closer to 1.0
        closer = float(db.mean()) < float(da.mean())
        mech[g] = {"A": float(a.mean()), "B": float(b.mean()), "delta": float(b.mean() - a.mean()),
                   "absA": float(da.mean()), "absB": float(db.mean()),
                   "closer": bool(closer), "sign_p": t["sign_p"],
                   "improvement": float(da.mean() - db.mean())}
        print(f"{g:<10}{a.mean():>10.3f}{b.mean():>10.3f}{b.mean()-a.mean():>+10.3f}"
              f"{da.mean():>9.3f}{db.mean():>9.3f}{str(closer):>9}{t['sign_p']:>11.2e}")

    h = mech["head"]
    improvement = h["improvement"]          # how much closer to 1.0, absolute
    verdict = ("PASS" if (improvement >= 0.05 and h["sign_p"] < 0.05 and h["closer"])
               else "PARTIAL" if (h["closer"] and h["sign_p"] < 0.05)
               else "FAIL")
    if not proxy_ok:
        verdict = "VOID (proxy did not reproduce)"

    print("\n" + "=" * 88)
    print(f"ENDPOINT: head ratio A {h['A']:.3f} -> B {h['B']:.3f}; "
          f"distance to 1.0 improved by {improvement:+.3f} (bar: >=0.05, sign p<0.05)")
    print(f"VERDICT (pre-registered): {verdict}")
    print("=" * 88)

    payload = {"n": n, "verdict": verdict, "proxy": proxy_out, "proxy_reproduces": bool(proxy_ok),
               "mechanism": mech, "thorax_abs": {"A": tA, "B": tB},
               "grouping_counts": counts, "arms": {"A": args.arm_a, "B": args.arm_b}}
    with open(os.path.join(out_dir, "x1_results.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {out_dir}/x1_results.json")


if __name__ == "__main__":
    main()
