"""R11 -- can pretrained semantic features assign LATERALIZED anatomical part identity?

Bar and design fixed in PREREGISTRATION_R11_lateralized_identity.md BEFORE this ran.

Reuses T1.0's machinery unmodified (template load, 8-view render, per-vertex backprojection).
Changes only what the question changed: lateralized labels, a supervised probe evaluated on
HELD-OUT specimens, and DINOv2 loaded via torch.hub (this cluster pins torch 2.3.1; transformers>=5
requires >=2.5 -- same weights, different loader).
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "out_R11")
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "diagnostics", "moonshot"))
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")
os.environ.setdefault("TORCH_HOME", "/hpcwork/nao48500/torch_hub")

import t10_diff3f_sanity as T10  # noqa: E402  reuse, do not reimplement
import render_3d as R3D            # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
# render_3d hardcodes DEV = "cuda:0" at module level; override at runtime rather than editing a
# shared module, so this also runs CPU-only when the GPU queue is unusable.
R3D.DEV = DEV
T10.DEV = DEV
SYNTH = os.path.join(REPO, "diagnostics/moonshot/synth_power48/ground_truth.npz")
N_TEST = 6


def lateral_group_of(name):
    """Level-2 lateralized anatomy. Mirrors T10.group_of's buckets but keeps the SIDE."""
    side = "_L" if name.endswith("_l") else ("_R" if name.endswith("_r") else "")
    if name.startswith("b_a_"):
        return "gaster"
    if name == "b_h":
        return "head"
    if name.startswith("ma"):
        return "mandible" + side
    if name.startswith("an_"):
        return "antenna" + side
    if name.startswith("l_"):
        # l_<chain>_<segment>_<side>; chain 1/2/3 = fore/mid/hind
        chain = name.split("_")[1]
        return {"1": "foreleg", "2": "midleg", "3": "hindleg"}.get(chain, "leg") + side
    return "thorax"


MIRROR = {}
for base in ("mandible", "antenna", "foreleg", "midleg", "hindleg"):
    MIRROR[base + "_L"] = base + "_R"
    MIRROR[base + "_R"] = base + "_L"


def dino_features_hub(imgs, device=DEV):
    """DINOv2 ViT-B/14 patch tokens upsampled to image resolution. torch.hub loader."""
    torch.hub.set_dir(os.path.join(os.environ["TORCH_HOME"], "hub"))
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14",
                           pretrained=True, verbose=False).to(device).eval()
    P, S = 14, 518  # patch size, input side (multiple of 14)
    feats = []
    with torch.no_grad():
        for img in imgs:
            x = torch.as_tensor(img[..., :3], dtype=torch.float32, device=device)
            x = x.permute(2, 0, 1).unsqueeze(0)
            x = F.interpolate(x, size=(S, S), mode="bilinear", align_corners=False)
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            out = model.forward_features((x - mean) / std)
            tok = out["x_norm_patchtokens"][0]                 # (S/P * S/P, 768)
            g = S // P
            fmap = tok.reshape(g, g, -1).permute(2, 0, 1).unsqueeze(0)
            fmap = F.interpolate(fmap, size=img.shape[:2], mode="bilinear", align_corners=False)
            feats.append(fmap[0].permute(1, 2, 0).cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return feats


def per_vertex_features(verts, faces):
    """(features (V,D), seen (V,) bool). backproject zero-fills never-visible vertices and
    returns the visibility mask separately -- coverage must come from that mask, not from NaN."""
    imgs, rasts = T10.render_views(verts, faces)
    fmaps = dino_features_hub(imgs)
    return T10.backproject(verts, faces, rasts, fmaps)


def main():
    os.makedirs(OUT, exist_ok=True)
    # T10.load_template returns (verts, faces, unlateralized_group) with verts already
    # centred and scaled to abs-max 1. Joint names / dominant weights are read separately.
    v_t, f_t, _ = T10.load_template()
    import pickle as _pk
    with open(os.path.join(REPO, "3D_model_prep", "OmniAnt_25PCs_joint_limited.pkl"), "rb") as fh:
        _u = _pk._Unpickler(fh); _u.encoding = "latin1"; dd = _u.load()
    jn = [str(x) for x in dd["J_names"]]
    dom = np.asarray(dd["weights"]).argmax(1)
    v_t_np = np.asarray(dd["v_template"], dtype=np.float64)
    labels_all = np.array([lateral_group_of(jn[d]) for d in dom])
    classes = sorted(set(labels_all))
    print(f"[r11] {len(classes)} classes: {classes}")

    # ---- CHECK 1: bilateral classes non-empty and L/R balanced on a symmetric template ----
    checks = {}
    for base in ("mandible", "antenna", "foreleg", "midleg", "hindleg"):
        nl = int((labels_all == base + "_L").sum()); nr = int((labels_all == base + "_R").sum())
        bal = (nl > 0 and nr > 0 and abs(nl - nr) / max(nl, nr) < 0.10)
        checks[base] = dict(n_L=nl, n_R=nr, balanced=bool(bal))
        print(f"[check1] {base:9s} L={nl:5d} R={nr:5d} balanced={bal}")
    if not all(c["balanced"] for c in checks.values()):
        print("[r11] VOID: lateral labelling is not balanced on a mirror-symmetric template")

    Ft, cov_t = per_vertex_features(v_t, f_t)
    print(f"[r11] template feature coverage {cov_t.mean()*100:.1f}%")

    # ---- probe: trained ONLY on the template ----
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    Xtr = Ft[cov_t]; ytr = labels_all[cov_t]
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0, multi_class="multinomial")
    clf.fit(sc.transform(Xtr), ytr)

    gt = np.load(SYNTH, allow_pickle=True)
    V = gt["verts"].astype(np.float64)
    names = [str(x) for x in gt["names"]]
    test_idx = list(range(min(N_TEST, len(names))))
    print(f"[check2] probe trained on template only; test specimens = "
          f"{[names[i] for i in test_idx]} (disjoint by construction)")

    lat_mask_all = np.isin(labels_all, list(MIRROR))
    rows = []
    for i in test_idx:
        # normalise each specimen exactly as load_template normalises the template, so the
        # renderer sees the same framing (centre on vertex mean, divide by abs max).
        vi = V[i] - V[i].mean(0)
        vi = vi / np.abs(vi).max()
        vi_t = torch.tensor(vi, dtype=torch.float32, device=DEV)
        Fs, cov = per_vertex_features(vi_t, f_t)
        pred = np.array(clf.predict(sc.transform(Fs[cov])), dtype=object)
        true = labels_all[cov]

        lat = np.isin(true, list(MIRROR))
        side_true = np.array([t[-1] for t in true[lat]])
        side_pred = np.array([p[-1] if p in MIRROR else "?" for p in pred[lat]])
        lateral_acc = float((side_true == side_pred).mean())
        part_acc = float((pred == true).mean())
        wrong = pred[lat] != true[lat]
        mirror_conf = float(np.mean([pred[lat][k] == MIRROR[true[lat][k]]
                                     for k in np.where(wrong)[0]])) if wrong.any() else 0.0

        # ---- geometric baseline on the IDENTICAL covered subset ----
        # nearest template vertex by rigid proximity (Procrustes-free: same topology frame)
        from scipy.spatial import cKDTree
        gi = cKDTree(np.asarray(v_t.cpu())).query(np.asarray(vi_t.cpu())[cov])[1]
        gpred = labels_all[gi]
        gside = np.array([p[-1] if p in MIRROR else "?" for p in gpred[lat]])
        geo_lateral = float((side_true == gside).mean())

        # ---- CHECK 4: shuffled-label control ----
        rs = np.random.RandomState(0)
        sh = rs.permutation(len(ytr))
        clf_sh = LogisticRegression(max_iter=400, C=1.0, multi_class="multinomial")
        clf_sh.fit(sc.transform(Xtr), ytr[sh])
        psh = np.array(clf_sh.predict(sc.transform(Fs[cov])), dtype=object)
        ssh = np.array([p[-1] if p in MIRROR else "?" for p in psh[lat]])
        chance_lateral = float((side_true == ssh).mean())

        rows.append(dict(specimen=names[i], coverage=float(cov.mean()),
                         n_lateral=int(lat.sum()), lateral_acc=lateral_acc,
                         part_acc=part_acc, mirror_confusion=mirror_conf,
                         geometric_baseline_lateral=geo_lateral,
                         shuffled_control_lateral=chance_lateral))
        print(f"  {names[i]:11s} cov {cov.mean()*100:5.1f}%  LATERAL {lateral_acc*100:5.1f}%  "
              f"(geo {geo_lateral*100:5.1f}%, shuffled {chance_lateral*100:5.1f}%)  "
              f"part {part_acc*100:5.1f}%  mirror-conf {mirror_conf*100:5.1f}%", flush=True)

    la = np.array([r["lateral_acc"] for r in rows])
    gb = np.array([r["geometric_baseline_lateral"] for r in rows])
    mc = np.array([r["mirror_confusion"] for r in rows])
    ntot = int(np.sum([r["n_lateral"] for r in rows]))
    from scipy import stats as st
    k = int(round(la.mean() * ntot))
    p_chance = float(st.binomtest(k, ntot, 0.5, alternative="greater").pvalue)
    beats_geo = bool(la.mean() > gb.mean())
    verdict = ("PASS" if (la.mean() >= 0.70 and p_chance < 0.05 and beats_geo)
               else "PARTIAL" if (p_chance < 0.05 and beats_geo) else "FAIL")

    res = dict(classes=classes, label_checks=checks,
               template_coverage=float(cov_t.mean()), per_specimen=rows,
               mean_lateral_acc=float(la.mean()), mean_geometric_baseline=float(gb.mean()),
               mean_mirror_confusion=float(mc.mean()),
               binomial_p_vs_chance=p_chance, beats_geometric_baseline=beats_geo,
               verdict=verdict)
    with open(os.path.join(OUT, "r11_results.json"), "w") as fh:
        json.dump(res, fh, indent=2)

    print("\n" + "=" * 92)
    print("R11 -- lateralized anatomical identity from pretrained semantic features")
    print("=" * 92)
    print(f"  mean LATERAL accuracy      {la.mean()*100:.1f}%   (bar: >=70%, chance 50%)")
    print(f"  geometric baseline         {gb.mean()*100:.1f}%   (must be beaten: {beats_geo})")
    print(f"  binomial p vs chance       {p_chance:.3g}")
    print(f"  mean mirror-confusion      {mc.mean()*100:.1f}%   (high => semantics ok, laterality not)")
    print(f"\n  VERDICT: {verdict}")
    print(f"\nwrote {os.path.join(OUT, 'r11_results.json')}")


if __name__ == "__main__":
    main()
