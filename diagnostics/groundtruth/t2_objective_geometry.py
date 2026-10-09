"""t2_objective_geometry.py -- does the data term reward inflating thin structures?

Design, predictions and voiding checks are fixed in PREREGISTRATION_T2_objective_geometry.md.
Read that first; this file only implements it.
"""
import sys, os, glob, json, re
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO)
for _d in ("morphometrics", "groundtruth"):
    sys.path.insert(0, os.path.join(REPO, "diagnostics", _d))
from measure import load_model  # noqa: E402

STAGE = os.path.join(REPO, "diagnostics", "full_corpus", "stage")
MIN_TARGET_VERTS = 1000
MIN_SPECIMENS = 600


def read_obj_verts(p):
    """Vertex block only. Bulk-split rather than per-line float() -- 757 files x ~40k lines each
    makes naive parsing the run's bottleneck."""
    with open(p, "rb") as fh:
        blob = fh.read()
    out = []
    for line in blob.split(b"\n"):
        if line[:2] == b"v ":
            out.append(line[2:])
    if not out:
        return np.empty((0, 3))
    return np.fromstring(b" ".join(out), sep=" ").reshape(-1, 3) \
        if hasattr(np, "fromstring") else np.array([l.split()[:3] for l in out], dtype=float)


def main():
    M = load_model()
    dom, jn = M["dominant"], M["jnames"]
    NV = M["v_template"].shape[0]

    # region map: which anatomical part each FITTED vertex belongs to
    def part_of(name):
        if name.startswith("ma"):
            return "mandible"
        if name == "b_h":
            return "head"
        if name == "b_t":
            return "mesosoma"
        if name.startswith("b_a_"):
            return "gaster"
        if name.startswith("an_"):
            return "antenna"
        if name.startswith("l_"):
            return "leg"
        return "other"
    vpart = np.array([part_of(jn[j]) for j in dom])
    REGIONS = ("mandible", "head", "mesosoma", "gaster", "antenna", "leg")

    # trait table from T1
    d = np.load(os.path.join(REPO, "diagnostics", "groundtruth", "traits_Z8.npz"))
    labs = [str(x) for x in d["labels"]]
    T = {k: d[k] for k in d.files if k != "labels"}
    lab_ix = {l: i for i, l in enumerate(labs)}

    targets = {os.path.basename(p): p for p in glob.glob(os.path.join(STAGE, "w*", "*_processed.obj"))}

    rows, dropped = [], {"no_target": 0, "too_small": 0, "unreadable": 0}
    for run in sorted(glob.glob(os.path.join(REPO, "diagnostics", "moonshot", "runs",
                                             "Z8_W*", "Stage_3_deform_fine.npz"))):
        z = np.load(run)
        V = z["verts"].astype(np.float64)
        assert V.shape[1] == NV, f"topology mismatch {V.shape[1]} != {NV} -- RUN VOID"
        for i, lab in enumerate([str(x) for x in z["labels"]]):
            tp = targets.get(lab)
            if tp is None:
                dropped["no_target"] += 1
                continue
            try:
                P = read_obj_verts(tp)
            except Exception:
                dropped["unreadable"] += 1
                continue
            if P.shape[0] < MIN_TARGET_VERTS:
                dropped["too_small"] += 1
                continue

            F = V[i]
            # EXACT replication of filter_quality.registration_error's frame convention, read
            # from that function rather than assumed. It normalises the TARGET by
            # `v - v.mean(0); v / v.abs().max()` and leaves the FIT RAW. An earlier version of
            # this script centroid-size-normalised BOTH, which put them at different per-specimen
            # scales and drove the alignment check to Spearman 0.446 -- the run voided itself.
            Pn = P - P.mean(0)
            Pn = Pn / np.abs(Pn).max()
            Fn = F                      # raw, exactly as the reference does

            tf, tp_ = cKDTree(Fn), cKDTree(Pn)
            d_tf, idx = tf.query(Pn)              # target point -> nearest fitted vertex
            d_ft, _ = tp_.query(Fn)               # fitted vertex -> nearest target point
            # symmetric MEAN SQUARED distance, as pytorch3d's chamfer_distance returns by default
            gres = float((d_tf ** 2).mean() + (d_ft ** 2).mean())
            reg = vpart[idx]
            r = dict(label=lab, n_target=len(Pn), global_resid=gres)
            for g in REGIONS:
                m = reg == g
                r[f"share_{g}"] = float(m.mean())
                # region residual stays a MEAN DISTANCE (not squared): it is compared across
                # specimens, and squaring would let a single outlier point dominate a small region.
                r[f"resid_{g}"] = float(d_tf[m].mean()) if m.any() else np.nan
            rows.append(r)

    n = len(rows)
    print(f"specimens measured: {n}   dropped: {dropped}")
    if n < MIN_SPECIMENS:
        print(f"RUN VOID: fewer than {MIN_SPECIMENS} specimens survived")
        return

    # ---- voiding check: does the recomputed residual track the stored one? -----
    reg_stored = json.load(open(os.path.join(REPO, "diagnostics", "morphometrics",
                                             "out", "registration_error.json")))
    a = np.array([r["global_resid"] for r in rows])
    b = np.array([reg_stored.get(r["label"], np.nan) for r in rows])
    m = ~np.isnan(b)
    rho_check = spearmanr(a[m], b[m]).statistic
    print(f"alignment check: Spearman(recomputed, stored registration error) = {rho_check:.3f}"
          f"   [>=0.80 required]")
    if rho_check < 0.80:
        print("RUN VOID: recomputed residual does not track the stored one; alignment is wrong.")
        return

    ix = np.array([lab_ix[r["label"]] for r in rows])
    MLHL = T["ML"][ix] / T["HL"][ix]
    WL = T["WL"][ix]
    WLdev = np.abs(np.log(WL) - np.log(np.median(WL)))

    print()
    print("=== P1: share of the objective carried by each region ===")
    print(f"{'region':10s} {'mean share':>11s} {'median resid':>13s}")
    for g in REGIONS:
        s = np.array([r[f"share_{g}"] for r in rows])
        rr = np.array([r[f"resid_{g}"] for r in rows])
        print(f"{g:10s} {100*s.mean():10.2f}% {np.nanmedian(rr):13.5f}")
    mshare = np.array([r["share_mandible"] for r in rows]).mean()
    p1 = mshare < 0.05
    print(f"\nP1 (mandible share < 5%): {'PASS' if p1 else 'FAIL'}   ({100*mshare:.2f}%)")
    if mshare > 0.15:
        print("   >15%: hypothesis dead on arrival, T1's explanation is WITHDRAWN.")

    print()
    print("=== P2: does inflating the mandible lower the mandible-region residual? ===")
    rm = np.array([r["resid_mandible"] for r in rows])
    ok = ~np.isnan(rm)
    rho2 = spearmanr(MLHL[ok], rm[ok])
    print(f"  Spearman(ML/HL, mandible residual) = {rho2.statistic:+.3f}   p={rho2.pvalue:.2g}   n={ok.sum()}")
    p2 = (rho2.statistic <= -0.15)
    print(f"  P2 (rho <= -0.15): {'PASS' if p2 else 'FAIL'}")

    print()
    print("=== CONTROL: the same effect must NOT appear on the thickest part ===")
    rs = np.array([r["resid_mesosoma"] for r in rows])
    ok2 = ~np.isnan(rs)
    rhoc = spearmanr(WLdev[ok2], rs[ok2])
    print(f"  Spearman(|log WL dev|, mesosoma residual) = {rhoc.statistic:+.3f}   p={rhoc.pvalue:.2g}")
    ctrl_null = not (rhoc.statistic <= -0.15)
    print(f"  CONTROL null (rho > -0.15): {'PASS' if ctrl_null else 'FAIL -- effect is not thin-structure-specific'}")

    print()
    print("=== VERDICT ===")
    if not p1:
        print("  Hypothesis rejected at P1. T1's mechanism is WITHDRAWN.")
    elif p1 and p2 and ctrl_null:
        print("  P1 PASS, P2 PASS, CONTROL null ->")
        print("  NAMED DEFECT: the area-sampled data term rewards inflating thin structures.")
        print("  Fix direction: region-weighted sampling of the data term.")
    elif p1 and p2 and not ctrl_null:
        print("  Effect present but NOT thin-structure-specific. T1's explanation WITHDRAWN;")
        print("  report the correlation without the mechanism.")
    else:
        print("  P2 FAILS -> the inversion is likely a LANDMARK artefact, not a fitter one.")
        print("  Blocked on the expert-adjudicated mandibular apex (protocol risk R1).")

    json.dump(dict(n=n, dropped=dropped, alignment_check=float(rho_check),
                   mandible_share=float(mshare), rho_P2=float(rho2.statistic),
                   rho_control=float(rhoc.statistic),
                   P1=bool(p1), P2=bool(p2), control_null=bool(ctrl_null),
                   region_share={g: float(np.mean([r[f"share_{g}"] for r in rows])) for g in REGIONS},
                   region_resid={g: float(np.nanmedian([r[f"resid_{g}"] for r in rows])) for g in REGIONS}),
              open(os.path.join(REPO, "diagnostics", "groundtruth", "t2_results.json"), "w"), indent=1)
    # per-specimen rows: the aggregate hid the structure that mattered, so keep the raw table
    np.savez(os.path.join(REPO, "diagnostics", "groundtruth", "t2_rows.npz"),
             label=np.array([r["label"] for r in rows]),
             global_resid=np.array([r["global_resid"] for r in rows]),
             **{f"share_{g}": np.array([r[f"share_{g}"] for r in rows]) for g in REGIONS},
             **{f"resid_{g}": np.array([r[f"resid_{g}"] for r in rows]) for g in REGIONS},
             ML_HL=MLHL, WLdev=WLdev)
    print("\nwrote t2_results.json and t2_rows.npz")


if __name__ == "__main__":
    main()
