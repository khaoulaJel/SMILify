"""Task 2: renders for the registration stage figure (PREREGISTRATION.md s3, gates 4-6).

One camera per view, framed once on the fitter frame (scan centred, max |coord| = 1) and reused
for every stage, one world-fixed point light (diagnostics/moonshot/render_3d.make_renderer
settings). Meshes are rotated (x, y, z) -> (x, z, -y) so the template's dorsal axis (+z) is the
renderer's up axis (+Y); head points to +x.

    --probe_orientation   template rest pose next to the normalised scan (gate 6, before any fit)
    (default)             stage panels, overlays, penetration probe, figure; needs score_stages.py output
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

from pytorch3d.io import load_obj  # noqa: E402
from pytorch3d.renderer import (  # noqa: E402
    FoVOrthographicCameras,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
    TexturesVertex,
    look_at_view_transform,
)
from pytorch3d.structures import Meshes  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
HERE = os.path.dirname(os.path.abspath(__file__))
VIEWS = {"dorsolateral": (22.0, 140.0), "dorsal": (89.0, 0.0), "lateral": (0.0, 0.0)}
FIG_STAGES = [("init", "init"), ("H2_joint", "pose"), ("Stage_2_deform_coarse", "Stage_2"),
              ("Stage_3_deform_fine", "Stage_3")]
C_FIT = (0.85, 0.62, 0.30)
C_SCAN = (0.72, 0.72, 0.72)
C_PEN = (0.85, 0.10, 0.10)
ROT = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])  # (x,y,z) -> (x,z,-y)


def make_renderer(view, image_size):
    elev, azim = VIEWS[view]
    R, T = look_at_view_transform(dist=2.6, elev=elev, azim=azim)
    cameras = FoVOrthographicCameras(device=DEV, R=R, T=T, scale_xyz=((1.15, 1.15, 1.15),))
    raster = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1)
    lights = PointLights(device=DEV, location=[[2.0, 2.0, 2.0]])
    return MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster),
        shader=SoftPhongShader(device=DEV, cameras=cameras, lights=lights),
    )


def render(parts, renderer):
    """parts: list of (verts (V,3), faces (F,3), colours (V,3)). Returns RGBA float image."""
    vs, fs, cs, off = [], [], [], 0
    for v, f, c in parts:
        v = torch.as_tensor(np.asarray(v, np.float32)) @ ROT.T
        vs.append(v)
        fs.append(torch.as_tensor(np.asarray(f, np.int64)) + off)
        cs.append(torch.as_tensor(np.asarray(c, np.float32)))
        off += len(v)
    m = Meshes(verts=[torch.cat(vs).to(DEV)], faces=[torch.cat(fs).to(DEV)],
               textures=TexturesVertex([torch.cat(cs).to(DEV)]))
    frags = renderer.rasterizer(m)
    rgb = renderer.shader(frags, m)[0, ..., :3].detach().cpu().numpy()
    alpha = (frags.pix_to_face[0, ..., 0] >= 0).float().cpu().numpy()
    return np.dstack([np.clip(rgb, 0, 1), alpha])


def solid(v, colour):
    return np.tile(np.asarray(colour, np.float32), (len(v), 1))


def load_scan(path):
    """Same normalisation as fitter_3d.utils.load_meshes."""
    v, f, _ = load_obj(path, load_textures=False)
    v = v - v.mean(0)
    v = v / v.abs().max(0)[0].max()
    return v.numpy(), f.verts_idx.numpy()


def save_rgba(img, path):
    plt.imsave(path, img)


def crop_box(imgs, pad=12):
    a = np.max([im[..., 3] for im in imgs], axis=0)
    ys, xs = np.nonzero(a > 0)
    return max(ys.min() - pad, 0), ys.max() + pad, max(xs.min() - pad, 0), xs.max() + pad


def probe_orientation(args):
    import pickle

    with open(os.path.join(REPO, "3D_model_prep", "OmniAnt_25PCs_joint_limited.pkl"), "rb") as f:
        dd = pickle.load(f, encoding="latin1")
    tv, tf = np.asarray(dd["v_template"], np.float32), np.asarray(dd["f"], np.int64)
    sv, sf = load_scan(os.path.join(args.mesh_dir, f"{args.focus}.obj"))
    os.makedirs(os.path.join(HERE, "probes"), exist_ok=True)
    fig, ax = plt.subplots(2, 3, figsize=(12, 7))
    for j, view in enumerate(VIEWS):
        r = make_renderer(view, 512)
        ax[0, j].imshow(render([(tv, tf, solid(tv, C_FIT))], r))
        ax[1, j].imshow(render([(sv, sf, solid(sv, C_SCAN))], r))
        ax[0, j].set_title(f"template rest, {view}")
        ax[1, j].set_title(f"scan {args.focus} normalised, {view}")
    for a in ax.ravel():
        a.axis("off")
    out = os.path.join(HERE, "probes", f"orientation_template_vs_scan{args.focus}_PROBE.png")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print("wrote", out)


def fmt_row(r):
    return (f"chamfer {float(r['chamfer_l2']):.2e}\nF@0.01 {float(r['fscore@0.01']):.3f}\n"
            f"penetration {int(round(float(r['pen_hard_triples'])))}")


def full(args):
    OUT = args.out_dir
    z = np.load(os.path.join(OUT, "probes", "focus_meshes.npz"), allow_pickle=True)
    faces, tv, tf = z["faces"], z["target_verts"], z["target_faces"]
    with open(os.path.join(OUT, "stages.csv")) as f:
        rows = {r["stage"]: r for r in csv.DictReader(f)}
    pdir, odir = os.path.join(OUT, "panels"), os.path.join(OUT, "probes")
    os.makedirs(pdir, exist_ok=True)

    for view in VIEWS:
        r = make_renderer(view, args.image_size)
        imgs = [render([(z[st], faces, solid(z[st], C_FIT))], r) for st, _ in FIG_STAGES]
        scan = render([(tv, tf, solid(tv, C_SCAN))], r)
        y0, y1, x0, x1 = crop_box(imgs + [scan])  # one crop for all panels: the camera stays shared
        for (st, lab), im in zip(FIG_STAGES, imgs):
            save_rgba(im[y0:y1, x0:x1], os.path.join(pdir, f"{view}_{lab}.png"))
        save_rgba(scan[y0:y1, x0:x1], os.path.join(pdir, f"{view}_scan.png"))

        # gate 4: fit on scan, every stage (scan grey, fit orange); plus fit-only and scan-only
        fig, ax = plt.subplots(3, 4, figsize=(16, 10))
        for j, (st, lab) in enumerate(FIG_STAGES):
            ov = render([(tv, tf, solid(tv, C_SCAN)), (z[st], faces, solid(z[st], C_FIT))], r)
            ax[0, j].imshow(ov[y0:y1, x0:x1])
            ax[1, j].imshow(imgs[j][y0:y1, x0:x1])
            ax[0, j].set_title(f"{lab}: fit (orange) + scan (grey)")
            ax[1, j].set_title(f"{lab}: fit only")
        ax[2, 0].imshow(scan[y0:y1, x0:x1])
        ax[2, 0].set_title("scan only")
        pen = z["pen_mask_stage3"]
        col = solid(z["Stage_3_deform_fine"], C_FIT)
        col[pen] = C_PEN
        ax[2, 1].imshow(render([(z["Stage_3_deform_fine"], faces, col)], r)[y0:y1, x0:x1])
        ax[2, 1].set_title(f"Stage_3 penetrating verts (red): {int(pen.sum())}")
        parts = z["part_of_vertex"]
        for k, part in enumerate(("legs", "gaster")):
            c = solid(z["Stage_3_deform_fine"], (0.85, 0.85, 0.85))
            c[parts == part] = C_PEN
            ax[2, 2 + k].imshow(render([(z["Stage_3_deform_fine"], faces, c)], r)[y0:y1, x0:x1])
            ax[2, 2 + k].set_title(f"Stage_3 part '{part}' (red)")
        for a in ax.ravel():
            a.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(odir, f"overlay_{view}_PROBE.png"), dpi=100)
        plt.close(fig)

    compose(args)
    print("renders done")


def compose(args):
    """Assemble the 300-dpi figure from the cropped transparent panels already in panels/ (no rendering).
    Figure height follows the panel aspect, so there are no empty bands."""
    OUT = args.out_dir
    with open(os.path.join(OUT, "stages.csv")) as f:
        rows = {r["stage"]: r for r in csv.DictReader(f)}
    ims = [plt.imread(os.path.join(OUT, "panels", f"{args.fig_view}_{lab}.png")) for _, lab in FIG_STAGES]
    h, w = ims[0].shape[:2]
    w_in = 180 / 25.4
    gap, title_in, text_in = 0.02, 0.18, 0.42
    panel_w = (w_in - 3 * gap * w_in / 4) / 4
    panel_h = panel_w * h / w
    H = title_in + panel_h + text_in
    fig = plt.figure(figsize=(w_in, H))
    for j, ((st, lab), im) in enumerate(zip(FIG_STAGES, ims)):
        left = j * (panel_w + gap * w_in / 4) / w_in
        ax = fig.add_axes([left, text_in / H, panel_w / w_in, panel_h / H])
        ax.imshow(im)
        ax.axis("off")
        fig.text(left + panel_w / w_in / 2, 1 - 0.02 / H, lab, ha="center", va="top", fontsize=8)
        fig.text(left + panel_w / w_in / 2, (text_in - 0.03) / H, fmt_row(rows[st]), ha="center", va="top",
                 fontsize=6.5, linespacing=1.3)
    fig.savefig(os.path.join(OUT, "fig_registration_stages.png"), dpi=300, facecolor="white")
    plt.close(fig)
    print("figure composed:", os.path.join(OUT, "fig_registration_stages.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh_dir", default="/hpcwork/nao48500/atta20")
    ap.add_argument("--focus", default="13")
    ap.add_argument("--image_size", type=int, default=1024)
    ap.add_argument("--fig_view", default="dorsolateral")
    ap.add_argument("--probe_orientation", action="store_true")
    ap.add_argument("--out_dir", default=HERE, help="folder holding score_stages.py output; renders go here too")
    ap.add_argument("--compose_only", action="store_true", help="only re-assemble the figure from panels/")
    args = ap.parse_args()
    if args.probe_orientation:
        probe_orientation(args)
    elif args.compose_only:
        compose(args)
    else:
        full(args)


if __name__ == "__main__":
    main()
