"""THE decisive check Fabian's request rests on: do the regressed b_h_l/b_h_r sit at the widest
point of the head -- on the fitted mesh AND on the actual scan?"""
import json, numpy as np
from scipy.spatial import cKDTree
from hw_core import *

d = load_all(); M = d["M"]
bh = blender_hw(EXPORT); bh_sub = blender_hw(EXPORT_SUBMITTED)
out = []
for i, lab in enumerate(d["labels"]):
    V = d["obs"][i]
    Rk, c, idx, bf = head_frame(V, M)
    H = to_head_coords(V[idx], Rk, c, bf)
    pl, pr = to_head_coords(d["PL"][i], Rk, c, bf)[0], to_head_coords(d["PR"][i], Rk, c, bf)[0]
    ap_len = H[:, 1].max() - H[:, 1].min()
    # width profile along AP
    edges = np.linspace(H[:, 1].min(), H[:, 1].max(), 41)
    prof, mids = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (H[:, 1] >= a) & (H[:, 1] < b)
        if m.sum() >= 5:
            prof.append(H[m, 0].max() - H[m, 0].min()); mids.append((a + b) / 2)
    prof, mids = np.array(prof), np.array(mids)
    w_max_mesh = H[:, 0].max() - H[:, 0].min()
    ap_widest = mids[prof.argmax()]
    ap_rep = (pl[1] + pr[1]) / 2
    lat_sep = abs(pl[0] - pr[0])
    # width of the mesh AT the reporters' AP level
    m = np.abs(H[:, 1] - ap_rep) < ap_len / 40
    w_at_rep = H[m, 0].max() - H[m, 0].min() if m.sum() > 5 else np.nan

    # ---- scan: points near the fitted head surface, same frame
    S = load_scan_normalised(lab)
    tree = cKDTree(V[idx])
    dist, _ = tree.query(S, k=1)
    edge_len = np.median(np.linalg.norm(V[d["faces"][:, 0]] - V[d["faces"][:, 1]], axis=1))
    S_head = S[dist < 4 * edge_len]
    HS = to_head_coords(S_head, Rk, c, bf)
    w_max_scan = np.percentile(HS[:, 0], 99.5) - np.percentile(HS[:, 0], 0.5)
    ms = np.abs(HS[:, 1] - ap_rep) < ap_len / 40
    w_scan_at_rep = (np.percentile(HS[ms, 0], 99.5) - np.percentile(HS[ms, 0], 0.5)) if ms.sum() > 20 else np.nan
    # scan AP profile
    sprof = []
    for a, b in zip(edges[:-1], edges[1:]):
        mm = (HS[:, 1] >= a) & (HS[:, 1] < b)
        sprof.append(np.percentile(HS[mm, 0], 99.5) - np.percentile(HS[mm, 0], 0.5) if mm.sum() > 30 else np.nan)
    sprof = np.array(sprof); smids = (edges[:-1] + edges[1:]) / 2
    ap_widest_scan = smids[np.nanargmax(sprof)]
    # reporters' distance to the scan surface
    st = cKDTree(S)
    dl = st.query(d["PL"][i])[0]; dr = st.query(d["PR"][i])[0]
    diag = np.linalg.norm(S.max(0) - S.min(0))

    out.append(dict(label=lab,
        HW_python_mm=float(d["HW_mm"][i]), HW_blender_mm=bh.get(lab), HW_submitted_mm=bh_sub.get(lab),
        HW_ref_mm=float(d["HW_ref"][i]),
        ratio_rep_to_widest_mesh=float(lat_sep / w_max_mesh),
        ratio_rep_to_widest_scan=float(lat_sep / w_max_scan),
        ratio_rep_to_width_at_level_scan=float(lat_sep / w_scan_at_rep),
        ap_offset_mesh_pct_headlen=float(100 * (ap_rep - ap_widest) / ap_len),
        ap_offset_scan_pct_headlen=float(100 * (ap_rep - ap_widest_scan) / ap_len),
        rep_to_scan_surface_pct_diag=float(100 * (dl + dr) / 2 / diag),
        dv_rep_pct_headheight=float(100 * ((pl[2] + pr[2]) / 2 - H[:, 2].min()) / (H[:, 2].max() - H[:, 2].min())),
    ))
json.dump(out, open(f"{SHOW}/data/widest_point_check.json", "w"), indent=1)

def s(k):
    v = np.array([o[k] for o in out], float); return f"median {np.nanmedian(v):7.3f}  [{np.nanmin(v):.3f}, {np.nanmax(v):.3f}]"
py = np.array([o["HW_python_mm"] for o in out]); bl = np.array([o["HW_blender_mm"] for o in out])
print(f"python vs blender(rematch) : mean|rel| {100*np.mean(np.abs(py/bl-1)):.3f}%  max {100*np.max(np.abs(py/bl-1)):.3f}%")
sub = np.array([o["HW_submitted_mm"] for o in out])
print(f"rematch vs submitted export: mean|rel| {100*np.mean(np.abs(bl/sub-1)):.2f}%")
for k in ["ratio_rep_to_widest_mesh", "ratio_rep_to_widest_scan", "ratio_rep_to_width_at_level_scan",
          "ap_offset_mesh_pct_headlen", "ap_offset_scan_pct_headlen", "rep_to_scan_surface_pct_diag",
          "dv_rep_pct_headheight"]:
    print(f"{k:36s} {s(k)}")
