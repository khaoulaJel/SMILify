"""T1.0 -- cheap sanity check: do extrinsic (SD-turbo diffusion + DINOv2) features show
anatomical structure on the SMIL ant template, before investing in Diff3F's full
symmetry-refinement pipeline for T1.1?

WHY THIS EXISTS
Intrinsic/geometric descriptors are proven dead for the within-part correspondence problem
(HKS: 0.66% correct even with a part oracle -- diagnostics/moonshot/hull/within_part_signal.py,
diagnostics/moonshot/hull/out/within_part_signal.json). FINAL_REPORT's proposed next step is
extrinsic, LEARNED features (Diff3F-style: image-diffusion features, which carry global
semantic context a local curvature/heat-kernel value cannot). Before forking
within_part_signal.py to build that properly (T1.1), check the cheapest possible version of
the question first: do raw, off-the-shelf diffusion/DINO features even look anatomically
structured on this template, or is the whole family a dead end here too?

This mirrors diagnostics/moonshot/sdf_prior_PROBE.py's own discipline: a signal-quality gate
before any correspondence-restriction machinery gets built on top of it.

METHOD
  1. Load the SMIL template + per-vertex anatomical group from skinning-weight argmax,
     grouped into {thorax, gaster, head, leg, antenna} -- the SAME 5-group scheme
     sdf_prior_PROBE.py used (so this result is directly comparable to that probe's
     AUC 0.97-0.99 benchmark), adapted to THIS model's joint-name convention (OmniAnt_25PCs
     uses abbreviated names -- 'b_a_1'..'b_a_5', 'l_k_seg_side', 'b_h', 'ma_r/l',
     'an_k_r/l' -- not sdf_prior_PROBE's substring style, which was written against a
     different model file's full joint names and does not match this model's names at all;
     see `group_of()` below, which is the same 5-bucket logic re-expressed for this
     model's names, not a reuse of that probe's string-matching code).
  2. Render the template from 8 views (azimuths 0-315 step 45, one elevation), plain grey
     Phong shading (diagnostics/moonshot/render_3d.py's make_renderer/render -- reused
     directly, not reimplemented).
  3. Per view: SD-turbo VAE-encode -> add noise at a fixed, modest timestep (t=100 of 1000,
     scaled-linear SD1.x schedule, sd-turbo's own scheduler_config.json) -> one UNet forward
     pass with a null text embedding -> grab one up_block's feature map, upsampled to render
     resolution. Separately: DINOv2-base patch tokens, reshaped to a spatial grid, upsampled
     to render resolution.
  4. Backproject to vertices: rasterize each view (pytorch3d, same rasterizer
     render_3d.make_renderer already builds), for each visible pixel assign its feature
     vector to the NEAREST of that pixel's face's 3 vertices (by barycentric weight -- a
     cheap simplification, not full barycentric splatting, appropriate for a sanity check
     not a production pipeline), accumulate a running per-vertex mean across all 8 views.
  5. k-means, k=5 (matching the 5 ground-truth groups), on (a) DINO features alone,
     (b) SD features alone, (c) both concatenated. Purity + ARI against the ground-truth
     groups, plus a chance control: k-means on random per-vertex vectors of the same
     dimensionality, same k, same seed protocol as within_part_signal.py's `random_within`.

PRE-REGISTERED PASS CONDITION, fixed before running:
  At least one of {DINO, SD, combined} must clear the chance control by a WIDE margin on
  BOTH purity and ARI (not one metric alone -- purity alone can be inflated by a
  low-entropy, mostly-one-bucket clustering). "Wide margin" = the same >=3x-over-chance bar
  within_part_signal.py's own reading logic used (`hw > 3 * max(rw, 1e-9)`), applied to ARI
  specifically since ARI is already chance-corrected (a chance clustering has ARI ~0, so any
  clearly-positive ARI is meaningful; purity's chance level with 5 unbalanced true groups is
  NOT 1/5 and must be read against the empirical chance-control run, not a naive 20%).
  FAIL (no candidate clears both) -> hard stop. Do not fork within_part_signal.py, do not
  investigate the Uzolas et al. symmetry-refinement layer, do not scope T1.2. Log the result
  and report it as a negative, exactly like every other dead end in this investigation.

Artifacts kept on disk per CLAUDE.md: this script, its stdout, and the cluster-coloured
render figure.
"""

import argparse
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MOON = os.path.abspath(os.path.join(HERE, "..", "moonshot"))
sys.path.insert(0, REPO)
sys.path.insert(0, MOON)
os.environ.setdefault("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")
os.environ.setdefault("HF_HOME", "/p/scratch/cias-7/jellal1/hf_cache")

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GROUPS = ["thorax", "gaster", "head", "leg", "antenna"]


def group_of(name):
    """5-bucket anatomical group from an OmniAnt_25PCs joint name.

    Same 5 buckets sdf_prior_PROBE.py used, re-expressed for THIS model's abbreviated
    joint-name convention (see module docstring). Mandible folds into 'head' and wings
    fold into 'thorax', matching sdf_prior_PROBE's own else-branch behaviour.
    """
    if name.startswith("b_a_"):
        return "gaster"
    if name.startswith("l_"):
        return "leg"
    if name.startswith("an_"):
        return "antenna"
    if name == "b_h" or name.startswith("ma"):
        return "head"
    return "thorax"


def load_template():
    with open(os.path.join(REPO, "3D_model_prep", "OmniAnt_25PCs_joint_limited.pkl"), "rb") as fh:
        u = pickle._Unpickler(fh)
        u.encoding = "latin1"
        dd = u.load()
    v = torch.tensor(np.asarray(dd["v_template"], dtype=np.float32), device=DEV)
    v = v - v.mean(0)
    v = v / v.abs().max()
    f = torch.tensor(np.asarray(dd["f"]).astype(np.int64), device=DEV)
    jn = [str(x) for x in dd["J_names"]]
    dom = np.asarray(dd["weights"]).argmax(1)
    vgroup = np.array([group_of(jn[j]) for j in dom])
    return v, f, vgroup


def render_views(v, f, n_views=8, image_size=512, elev=22.0, dist=2.6):
    from render_3d import make_renderer, render

    imgs, rasterizers = [], []
    grey = torch.full((v.shape[0], 3), 0.65, device=DEV)
    for k in range(n_views):
        azim = 360.0 * k / n_views
        renderer = make_renderer(image_size=image_size, dist=dist, elev=elev, azim=azim)
        img = render(v, f, grey, renderer)  # (H, W, 3) in [0,1]
        imgs.append(img)
        rasterizers.append(renderer.rasterizer)
    return imgs, rasterizers


def sd_features(imgs, device=DEV, timestep=100):
    from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
    from transformers import CLIPTextModel, CLIPTokenizer

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae").to(device).eval()
    unet = UNet2DConditionModel.from_pretrained("stabilityai/sd-turbo", subfolder="unet").to(device).eval()
    tok = CLIPTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer")
    txt = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder").to(device).eval()
    sched = DDPMScheduler.from_pretrained("stabilityai/sd-turbo", subfolder="scheduler")

    with torch.no_grad():
        null_ids = tok([""], padding="max_length", max_length=tok.model_max_length, return_tensors="pt").input_ids
        null_emb = txt(null_ids.to(device))[0]

    feats = []
    with torch.no_grad():
        for img in imgs:
            x = torch.tensor(img, device=device).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0  # [-1,1], (1,3,H,W)
            lat = vae.encode(x).latent_dist.mean * vae.config.scaling_factor
            noise = torch.randn_like(lat)
            t = torch.tensor([timestep], device=device)
            noisy = sched.add_noise(lat, noise, t)

            captured = {}

            def hook(module, inp, out):
                captured["feat"] = out

            h = unet.up_blocks[1].register_forward_hook(hook)
            unet(noisy, t, encoder_hidden_states=null_emb)
            h.remove()

            fmap = captured["feat"]  # (1, C, h, w), roughly H/8
            fmap = F.interpolate(fmap, size=img.shape[:2], mode="bilinear", align_corners=False)
            feats.append(fmap[0].permute(1, 2, 0).cpu().numpy())  # (H, W, C)
    del vae, unet, txt
    torch.cuda.empty_cache()
    return feats


def dino_features(imgs, device=DEV):
    from transformers import Dinov2Model, AutoImageProcessor

    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = Dinov2Model.from_pretrained("facebook/dinov2-base").to(device).eval()
    patch = model.config.patch_size

    feats = []
    with torch.no_grad():
        for img in imgs:
            pil_size = proc.crop_size["height"] if "height" in proc.crop_size else 518
            inputs = proc(images=(img * 255).astype(np.uint8), return_tensors="pt").to(device)
            out = model(**inputs)
            tok = out.last_hidden_state[0, 1:]  # drop CLS
            gh = gw = int(tok.shape[0] ** 0.5)
            fmap = tok[: gh * gw].reshape(gh, gw, -1).permute(2, 0, 1).unsqueeze(0)
            fmap = F.interpolate(fmap, size=img.shape[:2], mode="bilinear", align_corners=False)
            feats.append(fmap[0].permute(1, 2, 0).cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return feats


def backproject(v, f, rasterizers, feat_maps):
    """Nearest-vertex (by barycentric weight) splat of per-pixel features onto vertices,
    averaged across all views where a vertex is visible. Cheap simplification appropriate
    for a sanity check, not full barycentric interpolation.
    """
    n_v = v.shape[0]
    C = feat_maps[0].shape[-1]
    acc = np.zeros((n_v, C), dtype=np.float64)
    cnt = np.zeros(n_v, dtype=np.int64)

    for rasterizer, fmap in zip(rasterizers, feat_maps):
        from pytorch3d.structures import Meshes

        mesh = Meshes(verts=[v], faces=[f])
        frags = rasterizer(mesh)
        pix_to_face = frags.pix_to_face[0, ..., 0].cpu().numpy()  # (H, W)
        bary = frags.bary_coords[0, ..., 0, :].cpu().numpy()  # (H, W, 3)
        faces_np = f.cpu().numpy()

        valid = pix_to_face >= 0
        ys, xs = np.nonzero(valid)
        for y, x in zip(ys, xs):
            face_idx = pix_to_face[y, x]
            tri = faces_np[face_idx]
            vidx = tri[np.argmax(bary[y, x])]
            acc[vidx] += fmap[y, x]
            cnt[vidx] += 1

    seen = cnt > 0
    out = np.zeros((n_v, C), dtype=np.float32)
    out[seen] = (acc[seen] / cnt[seen, None]).astype(np.float32)
    return out, seen


def purity_ari(labels_pred, labels_true):
    from sklearn.metrics import adjusted_rand_score

    ari = adjusted_rand_score(labels_true, labels_pred)
    # purity: for each predicted cluster, take the majority true label, sum correct / N
    correct = 0
    for c in np.unique(labels_pred):
        m = labels_pred == c
        vals, counts = np.unique(labels_true[m], return_counts=True)
        correct += counts.max()
    purity = correct / len(labels_true)
    return purity, ari


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_views", type=int, default=8)
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--timestep", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from sklearn.cluster import KMeans

    print(f"[T1.0] loading template + ground-truth groups ...", flush=True)
    v, f, vgroup = load_template()
    for g in GROUPS:
        print(f"    {g:8s} n={int((vgroup == g).sum())}")

    print(f"[T1.0] rendering {args.n_views} views ...", flush=True)
    imgs, rasterizers = render_views(v, f, n_views=args.n_views, image_size=args.image_size)

    print("[T1.0] extracting SD-turbo features ...", flush=True)
    sd_maps = sd_features(imgs, timestep=args.timestep)
    print("[T1.0] extracting DINOv2 features ...", flush=True)
    dino_maps = dino_features(imgs)

    print("[T1.0] backprojecting SD features to vertices ...", flush=True)
    sd_v, sd_seen = backproject(v, f, rasterizers, sd_maps)
    print("[T1.0] backprojecting DINO features to vertices ...", flush=True)
    dino_v, dino_seen = backproject(v, f, rasterizers, dino_maps)

    seen = sd_seen & dino_seen
    print(f"[T1.0] vertices seen in >=1 view: SD {sd_seen.sum()}/{len(seen)}, "
          f"DINO {dino_seen.sum()}/{len(seen)}, both {seen.sum()}/{len(seen)}")

    y_true_str = vgroup[seen]
    name2id = {g: i for i, g in enumerate(GROUPS)}
    y_true = np.array([name2id[g] for g in y_true_str])

    rng = np.random.default_rng(args.seed)
    candidates = {
        "DINO": dino_v[seen],
        "SD": sd_v[seen],
        "combined": np.concatenate([dino_v[seen], sd_v[seen]], axis=1),
    }

    print(f"\n{'candidate':<12}{'dim':>6}{'purity':>10}{'ARI':>10}   vs chance control")
    print("-" * 70)
    results = {}
    for name, X in candidates.items():
        Xn = (X - X.mean(0)) / (X.std(0) + 1e-6)
        km = KMeans(n_clusters=len(GROUPS), n_init=10, random_state=args.seed).fit(Xn)
        pur, ari = purity_ari(km.labels_, y_true)

        # chance control: k-means on random vectors of the same shape, same seed protocol
        Xr = rng.normal(size=Xn.shape)
        km_r = KMeans(n_clusters=len(GROUPS), n_init=10, random_state=args.seed).fit(Xr)
        pur_r, ari_r = purity_ari(km_r.labels_, y_true)

        clears = ari > 3 * max(ari_r, 1e-9) and pur > pur_r
        results[name] = dict(purity=pur, ari=ari, purity_chance=pur_r, ari_chance=ari_r, clears=bool(clears))
        print(f"{name:<12}{X.shape[1]:>6}{pur:>10.3f}{ari:>10.3f}   "
              f"chance: purity={pur_r:.3f} ARI={ari_r:.3f}   "
              f"{'CLEARS >=3x' if clears else 'does NOT clear'}")

    any_pass = any(r["clears"] for r in results.values())
    print(f"\nPRE-REGISTERED VERDICT: {'PASS' if any_pass else 'FAIL'} "
          f"({'at least one candidate' if any_pass else 'no candidate'} clears the "
          f">=3x-ARI-over-chance, purity-over-chance bar)")

    import json

    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    with open(os.path.join(HERE, "out", "t10_result.json"), "w") as fh:
        json.dump(dict(results=results, any_pass=bool(any_pass), n_views=args.n_views,
                        timestep=args.timestep, n_vertices_seen=int(seen.sum())), fh, indent=1)
    print(f"\nwrote {os.path.join(HERE, 'out', 't10_result.json')}")


if __name__ == "__main__":
    main()
