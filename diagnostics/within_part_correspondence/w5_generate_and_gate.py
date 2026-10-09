"""W5 step 1 -- build W3b correspondence for synth_power48, and GATE the fitter runs on it.

Bar and decision tree fixed in PREREGISTRATION_W5_fitter_propagation.md BEFORE this ran.

WHAT THIS DOES
--------------
1. Leak check (VOIDING): no synth_power48 specimen may appear in W3b's training corpus.
2. Builds a per-template-vertex correspondence npz from W3b, in the exact format
   `--cse_correspondence_from` consumes (names / verts (N,V,3) / mask (N,V)).
3. GATE (VOIDING for the fitter runs): measures, against ground truth on these 48 specimens,
   whether W3b's COXAL correspondence is actually better than C3's. If it is not, the
   pre-registration's decision-tree row 3 applies and the fitter arms are NOT launched.
4. Builds arm B = C3's npz with COXA VERTICES ONLY replaced by W3b's, and verifies that non-coxa
   entries stay byte-identical.

WHY IT IS NOT CIRCULAR
----------------------
synth_power48 shares the template topology, so target vertex i IS template vertex i -- sampling
target VERTICES would hand the method the answer. Target points are therefore sampled from the
mesh SURFACE (as C3 does), and their true vertex is used only to SCORE, never to retrieve. The one
declared oracle is the part label of each target point (taken from its nearest true vertex), which
the pre-registration records as making this an upper bound rather than a deployable method.

W3b HAS NO KEY TABLE -- it is a part-conditioned query-to-query matcher -- so template-vertex
identity is carried by REFERENCE specimens: held-out members of the b2 corpus (disjoint from W3b's
training), whose vertex indices are known. A target point matched to reference vertex j is a vote
that it corresponds to template vertex j.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "fitter_3d"))
sys.path.insert(0, os.path.join(REPO, "fitter_3d", "pointcloud2smil"))
sys.path.insert(0, REPO)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

import config  # noqa: E402
from pytorch3d.io import load_obj  # noqa: E402
from pytorch3d.ops import knn_points, sample_points_from_meshes  # noqa: E402
from pytorch3d.structures import Meshes  # noqa: E402
from smil_correspondence_net import (  # noqa: E402
    build_segment_taxonomy, build_vertex_labels, load_model_dict,
)
from smil_cse_net import SMILCSENet  # noqa: E402
from w1_within_part_retrieval import LEG_SEGMENTS, MIN_VERTS, three_tests  # noqa: E402
from w3_contextual_embedding import W3Net, sample_subset  # noqa: E402


def load_w3b(path, n_parts, device):
    raw = torch.load(path, map_location="cpu")
    sd = raw["model_state_dict"]
    m = W3Net(n_parts, int(raw["embed_dim"]), int(raw["part_dim"]))
    inc = m.load_state_dict(sd, strict=True)
    assert not list(getattr(inc, "missing_keys", [])), "VOID: missing keys"
    assert not list(getattr(inc, "unexpected_keys", [])), "VOID: unexpected keys"
    probe = "head.0.weight"
    assert torch.allclose(dict(m.named_parameters())[probe].detach(), sd[probe], atol=0, rtol=0), \
        "VOID: probe tensor does not match raw file"
    m.to(device).eval()
    return m, int(raw.get("epoch", -1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w3b", default="diagnostics/within_part_correspondence/out_W3b/w3_best.pt")
    ap.add_argument("--c3", default="diagnostics/anatomical_pose_init/out_C3_cse_head_20260826/best_model.pt")
    ap.add_argument("--c3_npz", default="diagnostics/anatomical_pose_init/out_C3_cse_head_20260826/cse_p48_all.npz")
    ap.add_argument("--mesh_dir", default="diagnostics/moonshot/synth_power48")
    ap.add_argument("--corpus", default="diagnostics/moonshot/synth_b2_train_corrb06.npz")
    ap.add_argument("--n_refs", type=int, default=12, help="reference specimens (held-out b2 tail)")
    ap.add_argument("--n_points", type=int, default=4096, help="must match W3b's training density")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="diagnostics/within_part_correspondence/out_W5")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = os.path.join(REPO, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"device: {device}")

    # ---------------------------------------------------------------- taxonomy
    dd = load_model_dict(config.SMAL_FILE)
    class_names, name_to_id = build_segment_taxonomy(list(dd["J_names"]))
    labels, _ = build_vertex_labels(dd, class_names, name_to_id)
    V_TEMPLATE = np.asarray(dd["v_template"], dtype=np.float64)
    n_verts = len(labels)
    faces = torch.as_tensor(np.asarray(dd["f"]).astype(np.int64), device=device)

    parts = {}
    for cls, cid in name_to_id.items():
        if cls == "body":
            continue
        idx = np.where(labels == cid)[0]
        if len(idx) >= MIN_VERTS:
            parts[cls] = idx
    part_names = sorted(parts)
    part_idx_lists = [parts[c] for c in part_names]
    coxa_verts = np.concatenate([parts[c] for c in part_names if c.endswith("_co")])
    print(f"parts {len(parts)}; coxa template vertices: {len(coxa_verts)}")

    # per-vertex part id in W3b's convention (0 = 'other')
    vert_pid = np.zeros(n_verts, dtype=np.int64)
    for pi, c in enumerate(part_names):
        vert_pid[parts[c]] = pi + 1

    # ---------------------------------------------------------------- targets
    import glob
    obj_files = sorted(glob.glob(os.path.join(REPO, args.mesh_dir, "*.obj")))
    names = [os.path.splitext(os.path.basename(f))[0] for f in obj_files]
    targets = []
    for f in obj_files:
        v, _, _ = load_obj(f)
        targets.append(v.numpy().astype(np.float64))
    T = np.stack(targets)
    print(f"targets: {T.shape} from {args.mesh_dir}")

    # ---------------------------------------------------------------- CHECK 1: leak (voiding)
    b2 = np.load(os.path.join(REPO, args.corpus))["verts"]
    b2_sig = np.linalg.norm(b2, axis=2).sum(axis=1)
    exact = []
    for i, t in enumerate(T):
        d = np.abs(b2_sig - np.linalg.norm(t, axis=1).sum())
        j = int(d.argmin())
        if np.allclose(b2[j], t, atol=1e-5):
            exact.append((names[i], j))
    print(f"\n[CHECK 1 -- VOIDING] training-set leak: {len(exact)} of {len(T)} targets found in the "
          f"b2 corpus   {'PASS' if not exact else 'FAIL -- VOID: ' + str(exact[:3])}")
    if exact:
        raise SystemExit("VOID: synth_power48 specimens appear in W3b's training corpus")

    # ---------------------------------------------------------------- models
    w3b, w3b_ep = load_w3b(os.path.join(REPO, args.w3b), len(parts) + 1, device)
    print(f"W3b loaded strict, epoch {w3b_ep}")
    craw = torch.load(os.path.join(REPO, args.c3), map_location="cpu")
    c3 = SMILCSENet(n_vertices=int(craw["n_vertices"]), embed_dim=int(craw["embed_dim"]))
    c3.load_state_dict(craw["model_state_dict"], strict=True)
    c3.to(device).eval()
    print(f"C3 loaded strict, {craw['n_vertices']} verts, dim {craw['embed_dim']}")

    # ---------------------------------------------------------------- references (held-out b2)
    ref_ids = np.arange(len(b2) - args.n_refs, len(b2))
    R = b2[ref_ids].astype(np.float64)
    print(f"reference specimens: b2 {ref_ids[0]}..{ref_ids[-1]} (held out from W3b training)")

    # one shared vertex subset for the references, so their embeddings and identities line up
    sel, rows_per_part, row_pid = sample_subset(rng, part_idx_lists, args.n_points, n_verts)
    pid_t = torch.as_tensor(row_pid, device=device)
    with torch.no_grad():
        ref_emb = np.stack([
            w3b(torch.as_tensor(R[s][sel], dtype=torch.float32, device=device).unsqueeze(0),
                pid_t)[0].cpu().numpy() for s in range(len(R))])
    print(f"reference embeddings: {ref_emb.shape}")

    # ---------------------------------------------------------------- per-target correspondence
    pseudo = np.zeros((len(T), n_verts, 3), dtype=np.float32)
    mask = np.zeros((len(T), n_verts), dtype=bool)
    gate_w3b, gate_c3_full, gate_c3_part, gate_c3_same = [], [], [], []
    reachable_frac = []

    keys = c3.vertex_embeddings()
    with torch.no_grad():
        for ti, t in enumerate(T):
            vt = torch.as_tensor(t, dtype=torch.float32, device=device).unsqueeze(0)
            mesh = Meshes(verts=vt, faces=faces.unsqueeze(0))
            pts = sample_points_from_meshes(mesh, args.n_points)          # (1,N,3) SURFACE points
            true_vert = knn_points(pts, vt, K=1).idx[0, :, 0].cpu().numpy()   # scoring only
            tgt_pid = torch.as_tensor(vert_pid[true_vert], device=device)     # DECLARED ORACLE

            q_w3b = w3b(pts, tgt_pid)[0].cpu().numpy()
            q_c3 = c3(pts)[0]
            pts_np = pts[0].cpu().numpy()

            # --- C3, full vocabulary (its deployed mode) ---
            pred_c3_full = (q_c3 @ keys.t()).argmax(1).cpu().numpy()

            # --- W3b: match to reference vertices WITHIN the same part, vote across references ---
            pred_w3b = np.full(args.n_points, -1, dtype=np.int64)
            for pi, c in enumerate(part_names):
                m = vert_pid[true_vert] == pi + 1
                if not m.any():
                    continue
                rows = rows_per_part[pi]
                if len(rows) < 4:
                    continue
                cand_vids = sel[rows]
                # cosine similarity summed over reference specimens = a vote
                sims = np.zeros((int(m.sum()), len(rows)))
                for s in range(len(R)):
                    sims += q_w3b[m] @ ref_emb[s][rows].T
                pred_w3b[m] = cand_vids[sims.argmax(1)]

            # --- FAIRNESS CONTROL (added after the first gate run exposed the confound) ---
            # W3b's candidate set is limited to the 4096 sampled REFERENCE vertices, because its
            # embeddings are only valid at that density (W2's density lesson). So for ~60% of
            # target points the TRUE vertex is not even in W3b's candidate set, while C3 retrieves
            # over all 10,235. Comparing them directly measures the candidate-set handicap I
            # imposed, not the descriptor. This arm restricts C3 to the SAME sampled candidates
            # within the same oracle part, which is the only apples-to-apples comparison.
            pred_c3_same = np.full(args.n_points, -1, dtype=np.int64)
            for pi, c in enumerate(part_names):
                m = vert_pid[true_vert] == pi + 1
                if not m.any():
                    continue
                rows = rows_per_part[pi]
                if len(rows) < 4:
                    continue
                cand = sel[rows]
                sub = (q_c3[torch.as_tensor(m, device=device)] @ keys[cand].t()).argmax(1)
                pred_c3_same[m] = cand[sub.cpu().numpy()]

            # --- C3 restricted to the same oracle part, for a like-for-like comparison ---
            pred_c3_part = np.full(args.n_points, -1, dtype=np.int64)
            for pi, c in enumerate(part_names):
                m = vert_pid[true_vert] == pi + 1
                if not m.any():
                    continue
                cand = parts[c]
                sub = (q_c3[torch.as_tensor(m, device=device)] @ keys[cand].t()).argmax(1)
                pred_c3_part[m] = cand[sub.cpu().numpy()]

            # --- GATE: coxal correspondence error against ground truth, rest space ---
            is_co = np.isin(true_vert, coxa_verts)
            if is_co.any():
                tv = true_vert[is_co]
                gate_w3b.append(np.linalg.norm(V_TEMPLATE[pred_w3b[is_co]] - V_TEMPLATE[tv], axis=1))
                gate_c3_full.append(np.linalg.norm(V_TEMPLATE[pred_c3_full[is_co]] - V_TEMPLATE[tv], axis=1))
                gate_c3_part.append(np.linalg.norm(V_TEMPLATE[pred_c3_part[is_co]] - V_TEMPLATE[tv], axis=1))
                gate_c3_same.append(np.linalg.norm(V_TEMPLATE[pred_c3_same[is_co]] - V_TEMPLATE[tv], axis=1))
                # how often is the TRUE vertex even reachable in W3b's candidate set?
                reachable_frac.append(float(np.isin(tv, sel).mean()))

            # --- accumulate per-template-vertex targets (W3b) ---
            ok = pred_w3b >= 0
            acc = np.zeros((n_verts, 3))
            cnt = np.zeros(n_verts)
            np.add.at(acc, pred_w3b[ok], pts_np[ok])
            np.add.at(cnt, pred_w3b[ok], 1)
            hit = cnt > 0
            pseudo[ti, hit] = (acc[hit] / cnt[hit, None]).astype(np.float32)
            mask[ti, hit] = True
            if (ti + 1) % 12 == 0:
                print(f"  {ti+1}/{len(T)} targets", flush=True)

    # ---------------------------------------------------------------- CHECK 3: the gate
    med = lambda L: float(np.median(np.concatenate(L)))
    per_spec = lambda L: np.array([np.median(x) for x in L])
    gw, gcf, gcp, gcs = med(gate_w3b), med(gate_c3_full), med(gate_c3_part), med(gate_c3_same)
    t_full = three_tests(per_spec(gate_w3b), per_spec(gate_c3_full))
    t_part = three_tests(per_spec(gate_w3b), per_spec(gate_c3_part))
    print("\n" + "=" * 92)
    print("[CHECK 3 -- GATE] coxal correspondence error vs GT on these 48 specimens (rest space)")
    print("=" * 92)
    print(f"  W3b (oracle part + reference vote) : {gw:.5f}")
    print(f"  C3  (full vocabulary, deployed)    : {gcf:.5f}   "
          f"W3b better by {100*(gcf-gw)/gcf:+.1f}%  sign p {t_full['sign_p']:.2e} "
          f"({t_full.get('n_better')}/{t_full['n']})")
    print(f"  C3  (restricted to oracle part)    : {gcp:.5f}   "
          f"W3b better by {100*(gcp-gw)/gcp:+.1f}%  sign p {t_part['sign_p']:.2e} "
          f"({t_part.get('n_better')}/{t_part['n']})")
    t_same = three_tests(per_spec(gate_w3b), per_spec(gate_c3_same))
    print(f"  C3  (SAME candidate subset)        : {gcs:.5f}   "
          f"W3b better by {100*(gcs-gw)/gcs:+.1f}%  sign p {t_same['sign_p']:.2e} "
          f"({t_same.get('n_better')}/{t_same['n']})   <-- APPLES-TO-APPLES")
    print(f"\n  true coxal vertex reachable in W3b's candidate set: "
          f"{100*float(np.mean(reachable_frac)):.1f}% of target points")
    gate_pass = (gw < gcf) and (t_full["sign_p"] < 0.05)
    print(f"\n  GATE (W3b beats C3 as deployed, sign p<0.05): "
          f"{'PASS -- fitter arms are justified' if gate_pass else 'FAIL'}")
    if not gate_pass:
        print("  -> pre-registration decision-tree row 3: the retrieval result did NOT survive the")
        print("     deployment regime. The fitter arms are NOT launched.")

    # ---------------------------------------------------------------- arm B npz + CHECK 2
    c3npz = np.load(os.path.join(REPO, args.c3_npz), allow_pickle=True)
    c3_names = [str(x) for x in list(c3npz["names"])]
    order = [c3_names.index(n) for n in names]
    B_verts = c3npz["verts"][order].copy()
    B_mask = c3npz["mask"][order].copy()
    A_verts, A_mask = B_verts.copy(), B_mask.copy()
    B_verts[:, coxa_verts, :] = pseudo[:, coxa_verts, :]
    B_mask[:, coxa_verts] = mask[:, coxa_verts]

    noncoxa = np.setdiff1d(np.arange(n_verts), coxa_verts)
    same_noncoxa = (np.array_equal(A_verts[:, noncoxa], B_verts[:, noncoxa])
                    and np.array_equal(A_mask[:, noncoxa], B_mask[:, noncoxa]))
    differ_coxa = not np.array_equal(A_verts[:, coxa_verts], B_verts[:, coxa_verts])
    print(f"\n[CHECK 2 -- VOIDING] arms differ on coxa: {differ_coxa}; "
          f"non-coxa byte-identical: {same_noncoxa}   "
          f"{'PASS' if (differ_coxa and same_noncoxa) else 'FAIL -- VOID'}")

    print(f"[CHECK 4] coverage  A(C3) {100*A_mask.mean():.1f}%   B(hybrid) {100*B_mask.mean():.1f}%"
          f"   coxa-only: A {100*A_mask[:, coxa_verts].mean():.1f}%  "
          f"B {100*B_mask[:, coxa_verts].mean():.1f}%")

    np.savez_compressed(os.path.join(out_dir, "w5_armB_c3_with_w3b_coxa.npz"),
                        names=np.array(names), verts=B_verts, mask=B_mask)
    np.savez_compressed(os.path.join(out_dir, "w5_w3b_raw.npz"),
                        names=np.array(names), verts=pseudo, mask=mask)
    payload = {"config": vars(args), "leak_exact_matches": len(exact), "w3b_epoch": w3b_ep,
               "gate": {"w3b": gw, "c3_full": gcf, "c3_part": gcp,
                        "rel_vs_c3_full": (gcf - gw) / gcf, "rel_vs_c3_part": (gcp - gw) / gcp,
                        "c3_same_candidates": gcs, "rel_vs_c3_same": (gcs - gw) / gcs,
                        "sign_p_same": t_same["sign_p"],
                        "reachable_frac": float(np.mean(reachable_frac)),
                        "sign_p_full": t_full["sign_p"], "sign_p_part": t_part["sign_p"],
                        "pass": bool(gate_pass)},
               "arms_differ_coxa": bool(differ_coxa), "noncoxa_identical": bool(same_noncoxa),
               "coverage": {"A": float(A_mask.mean()), "B": float(B_mask.mean())}}
    with open(os.path.join(out_dir, "w5_gate.json"), "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nwrote {out_dir}/w5_gate.json and the arm-B npz")


if __name__ == "__main__":
    main()
