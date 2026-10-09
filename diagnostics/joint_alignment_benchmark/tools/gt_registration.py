"""Fit-INDEPENDENT registration of the expert joint annotations into the fitter's frame.

WHY THIS EXISTS
Every previous joint-accuracy number in this project (G1, G6-G10, V5-V13) put the ground-truth
joints into the fitter frame with a similarity solved AGAINST A FITTED SKELETON -- per evaluation
(G1) or once against the production fit (V9's `sim_fit(GT, production joints)`, and
`annotation/joint_to_obj_transform.json`). The ruler then depends on one of the things being
measured (Fitzpatrick's FRE/TRE distinction: an alignment fitted on the evaluated points hides the
error it absorbs).

The expert .blend files contain the scan mesh in the exact coordinates in which the joints were
placed. A mesh-to-mesh similarity with vertex correspondence gives the annotation -> .obj map at
float precision without consulting any model. Audit findings encoded below (see
../PREREGISTRATION.md section 3, and ../data/gt_registration_audit.json):

  EXACT        blend mesh == obj up to a proper similarity (Blender's Y-up -> Z-up + scale).
  MIRRORED     blend mesh == obj up to an IMPROPER similarity (det -1). The annotator labelled
               sides by the anatomy seen in the mirrored scene (posture-robust chirality cue is
               positive in blend frame on 12/12), so carrying the points to the scan requires the
               reflection AND swapping _r <-> _l.
  CHAIN        Cephalotes was annotated on an older processed mesh (annotation/bench_10, 48086 v):
               blend -> bench_10 obj exact (improper); bench_10 obj -> fit-target obj by trimmed
               similarity ICP (residual reported; no vertex correspondence exists).
  EXCLUDED     Dolichoderus: the .blend mesh IS the Discothyrea scan (resid 8e-9) and the markers
               lie inside it. It is not an annotation of the Dolichoderus scan.

Then `fitter_3d.utils.load_meshes` normalises each target as (v - mean) / max|v - mean|.

OUTPUT  ../data/gt_joints_fitframe.json   per specimen: joint -> xyz (obj and fit frame),
                                          visibility, registration tier and residuals, WL.
        ../data/gt_registration_audit.json
"""
import glob
import json
import os
import re
import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(BENCH, "..", ".."))
BLEND_DUMPS = "/hpcwork/nao48500/jab_blend"
MESHDIR = "/hpcwork/nao48500/worker_alt_data"
EXCLUDE = {"Dolichoderus_cf.bidens_CASENT0744033":
           "annotated in a scene whose mesh is Discothyrea_patrizii_CASENT0744991 (resid 8e-9); "
           "markers lie inside that mesh -- not an annotation of this specimen"}
CHAIN_VIA = {"Cephalotes_atratus_CASENT0744612":
             os.path.join(REPO, "annotation/bench_10/Cephalotes_atratus_CASENT0744612_processed.obj")}


def norm_sid(s):
    s = s.replace("_EDITED_MARKERS", "").replace("_edited", "")
    h = len(s) // 2
    if len(s) % 2 == 0 and s[:h] == s[h:]:
        s = s[:h]
    return s


def read_obj_verts(p):
    V = []
    with open(p) as fh:
        for L in fh:
            if L.startswith("v "):
                V.append([float(x) for x in L.split()[1:4]])
    return np.asarray(V, dtype=np.float64)


def similarity(src, dst, allow_reflection):
    """dst ~= s * M @ src + t (Umeyama 1991); M orthogonal, det(M) = -1 allowed if requested."""
    cs, cd = src.mean(0), dst.mean(0)
    S0, D0 = src - cs, dst - cd
    U, sig, Vt = np.linalg.svd(D0.T @ S0 / len(src))
    E = np.eye(3)
    if not allow_reflection and np.linalg.det(U) * np.linalg.det(Vt) < 0:
        E[2, 2] = -1
    M = U @ E @ Vt
    s = np.trace(np.diag(sig) @ E) / (S0 ** 2).sum(1).mean()
    return s, M, cd - s * M @ cs


def icp_similarity(src, dst, n_iter=100, trim=95, stride=4):
    """Correspondence-free proper-similarity ICP from the 24 axis-aligned starts; returns the best
    transform (src -> dst) and symmetric median / p95 residuals relative to the dst diagonal."""
    tD = cKDTree(dst)
    diag = np.linalg.norm(np.ptp(dst, 0))
    best = None
    sub = src[::stride]
    for R0 in Rotation.create_group("O").as_matrix():
        X = sub @ R0.T
        s0 = np.sqrt(((dst - dst.mean(0)) ** 2).sum(1).mean() / ((X - X.mean(0)) ** 2).sum(1).mean())
        T = (s0, R0, dst.mean(0) - s0 * R0 @ sub.mean(0))
        for _ in range(n_iter):
            Y = sub @ T[1].T * T[0] + T[2]
            d, i = tD.query(Y)
            k = d <= np.percentile(d, trim)
            s_, M_, t_ = similarity(sub[k], dst[i[k]], allow_reflection=False)
            T = (s_, M_, t_)
        Y = src @ T[1].T * T[0] + T[2]
        d1, _ = tD.query(Y)
        d2, _ = cKDTree(Y).query(dst)
        score = (np.median(d1) + np.median(d2)) / 2
        if best is None or score < best[0]:
            best = (score, T, np.percentile(d1, 95), np.percentile(d2, 95))
    return best[1], dict(sym_median_rel=float(best[0] / diag), p95_fwd_rel=float(best[2] / diag),
                         p95_bwd_rel=float(best[3] / diag))


def swap_side(name):
    if name.endswith("_r"):
        return name[:-1] + "l"
    if name.endswith("_l"):
        return name[:-1] + "r"
    return name


def chirality(P):
    """cos(anterior x dorsal, mean(right) - mean(left)); dorsal = body-axis joints minus distal
    leg joints (tarsi/pretarsi hang ventrally in pinned specimens). +1 = template convention."""
    g = lambda ks: np.mean([P[k] for k in ks if k in P], 0)
    body = ["b_t", "b_a_1", "b_a_2", "b_a_3", "b_a_4", "b_a_5"]
    ant = g(["ma_r", "ma_l", "an_1_r", "an_1_l", "b_h"]) - g(["b_a_3", "b_a_4", "b_a_5"])
    R = [k for k in P if k.endswith("_r") and k[:2] in ("l_", "ma", "an") and swap_side(k) in P]
    lat = g(R) - g([swap_side(k) for k in R])
    dors = g(body) - g([k for k in P if k.startswith("l_") and k[4:6] in ("ta", "pt")])
    dors = dors - ant * np.dot(dors, ant) / np.dot(ant, ant)
    right = np.cross(ant, dors)
    return float(np.dot(right, lat) / np.linalg.norm(right) / np.linalg.norm(lat))


def weber_length_obj(sid):
    """WL from the human trait landmarks. Stored `original` is Blender Z-up (obj = (x, z, -y));
    corrected here, and the endpoints are checked to lie on the scan surface."""
    p = os.path.join(REPO, "annotation/landmarks", f"{sid}_traits.json")
    if not os.path.exists(p):
        return None, None
    L = json.load(open(p))["landmarks"]
    if "wl_anterior_r" not in L or "wl_posterior_r" not in L:
        return None, None
    fix = lambda v: np.array([v[0], v[2], -v[1]], float)
    a, b = fix(L["wl_anterior_r"]["original"]), fix(L["wl_posterior_r"]["original"])
    return float(np.linalg.norm(a - b)), (a, b)


def main():
    out, audit = {}, []
    tpl_chir = None
    for gj in sorted(glob.glob(os.path.join(REPO, "annotation/gt_expert/*_joints.json"))):
        gt = json.load(open(gj))
        sid = norm_sid(gt["specimen_id"])
        dumps = [p for p in glob.glob(os.path.join(BLEND_DUMPS, "*.npz"))
                 if norm_sid(os.path.basename(p)[:-4]) == sid]
        assert len(dumps) == 1, (sid, dumps)
        z = np.load(dumps[0], allow_pickle=True)
        mesh_keys = [k for k in z.files if k.startswith("mesh__")]
        assert len(mesh_keys) == 1, (sid, mesh_keys)
        B = z[mesh_keys[0]]
        blend_mesh_name = mesh_keys[0][6:]
        P_blend = {j["joint_name"]: np.asarray(j["position"], float)
                   for j in gt["joints"] if j.get("position") is not None}
        vis = {j["joint_name"]: j.get("visibility") for j in gt["joints"]}
        emp = {re.sub(r"^empty__MARKER_\d+_", "", k): z[k] for k in z.files if k.startswith("empty__")}
        diagB = np.linalg.norm(np.ptp(B, 0))
        json_vs_blend = max(np.linalg.norm(P_blend[n] - emp[n]) / diagB for n in P_blend if n in emp)
        rec = dict(sid=sid, blend_mesh_object=blend_mesh_name, n_placed=len(P_blend),
                   json_vs_blend_max_rel=float(json_vs_blend),
                   chirality_blend_frame=chirality(P_blend))
        if sid in EXCLUDE:
            rec.update(tier="EXCLUDED", reason=EXCLUDE[sid])
            audit.append(rec)
            continue

        O = read_obj_verts(os.path.join(MESHDIR, f"{sid}_processed.obj"))
        if sid in CHAIN_VIA:
            A = read_obj_verts(CHAIN_VIA[sid])
            # step 1: older mesh -> blend. No shared vertex order, but ICP proved every older-mesh
            # vertex lands on a blend vertex; solve by ICP allowing reflection via a flipped copy.
            Af = A * np.array([1, 1, -1])
            T1, r1 = icp_similarity(Af, B)
            s1, M1, t1 = T1[0], T1[1] @ np.diag([1, 1, -1]), T1[2]      # A -> B, improper
            # step 2: older mesh -> fit-target obj (different processing of the same scan)
            T2, r2 = icp_similarity(A, O)
            s2, M2, t2 = T2
            # compose blend -> obj: A = M1^T (B - t1)/s1 ; O = s2 M2 A + t2
            to_obj = lambda p: s2 * M2 @ (M1.T @ (p - t1) / s1) + t2
            det = float(np.sign(np.linalg.det(M1)))
            rec.update(tier="CHAIN", step1_blend_vs_older=r1, step2_older_vs_target=r2,
                       reg_resid_rel=r2["sym_median_rel"])
        else:
            assert len(O) == len(B), (sid, len(O), len(B))
            s, M, t = similarity(O, B, allow_reflection=True)           # obj -> blend
            resid = np.linalg.norm(O @ M.T * s + t - B, axis=1).max() / diagB
            assert resid < 1e-5, (sid, resid)
            det = float(np.sign(np.linalg.det(M)))
            to_obj = lambda p, s=s, M=M, t=t: M.T @ (p - t) / s
            rec.update(tier="EXACT" if det > 0 else "MIRRORED", reg_resid_rel=float(resid))

        swap = det < 0
        P_obj = {(swap_side(n) if swap else n): to_obj(p) for n, p in P_blend.items()}
        vis_obj = {(swap_side(n) if swap else n): v for n, v in vis.items()}
        c = O.mean(0)
        sc = np.abs(O - c).max()
        P_fit = {n: (p - c) / sc for n, p in P_obj.items()}
        wl, wl_pts = weber_length_obj(sid)
        rec.update(det=det, labels_swapped=bool(swap), chirality_scan_frame=chirality(P_obj),
                   WL_obj=wl, WL_fit=(wl / sc if wl else None))
        # the corrected WL endpoints must lie on the scan surface
        if wl_pts is not None:
            d, _ = cKDTree(O).query(np.stack(wl_pts))
            rec["WL_endpoint_to_scan_vertex_rel"] = float(d.max() / np.linalg.norm(np.ptp(O, 0)))
        out[sid] = dict(tier=rec["tier"], labels_swapped=bool(swap), obj_centre=c.tolist(),
                        obj_scale=float(sc), WL_obj=wl, WL_fit=rec["WL_fit"],
                        joints={n: dict(obj=P_obj[n].tolist(), fit=P_fit[n].tolist(),
                                        visibility=vis_obj.get(n)) for n in P_obj},
                        not_placed=[swap_side(n) if swap else n for n, v in vis.items()
                                    if n not in P_blend])
        audit.append(rec)

    os.makedirs(os.path.join(BENCH, "data"), exist_ok=True)
    json.dump(out, open(os.path.join(BENCH, "data", "gt_joints_fitframe.json"), "w"), indent=1)
    json.dump(audit, open(os.path.join(BENCH, "data", "gt_registration_audit.json"), "w"), indent=1)
    print(f"{'specimen':38s}{'tier':>9}{'resid':>10}{'swap':>6}{'chir blend':>11}{'chir scan':>10}{'WL/sc':>7}")
    for a in audit:
        print(f"{a['sid']:38s}{a['tier']:>9}{a.get('reg_resid_rel', float('nan')):>10.1e}"
              f"{str(a.get('labels_swapped', '-')):>6}{a['chirality_blend_frame']:>11.2f}"
              f"{a.get('chirality_scan_frame', float('nan')):>10.2f}"
              f"{(a.get('WL_fit') or float('nan')):>7.3f}")
    kept = [a for a in audit if a["tier"] != "EXCLUDED"]
    assert all(a["chirality_scan_frame"] > 0 for a in kept), "label side convention violated"
    print(f"\n{len(kept)} specimens registered; {len(audit) - len(kept)} excluded")


if __name__ == "__main__":
    sys.exit(main())
