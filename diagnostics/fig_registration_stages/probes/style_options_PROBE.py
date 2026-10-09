"""Style options for the Task 2 panels (mock-up sheet, not the figure): init and Stage_3 of specimen 13,
same dorsolateral camera, four surface treatments.

  A clay      neutral light-grey matte, softer light (ambient up, specular down)
  B ghost     fit in one colour drawn over the scan's grey silhouette: grey = scan not covered by the fit
  C distance  fit coloured by distance to the scan surface, one-hue sequential ramp (light -> dark blue),
              clipped at 0.02 (2 x the F-score tau), same scale in every panel
  D parts     anatomical parts in the validated categorical order (blue, orange, aqua, yellow, magenta)

    python diagnostics/fig_registration_stages/probes/style_options_PROBE.py
"""

import os
import sys

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import render_stages as RS  # noqa: E402
from pytorch3d.ops import sample_points_from_meshes  # noqa: E402
from pytorch3d.renderer import (  # noqa: E402
    FoVOrthographicCameras, Materials, MeshRasterizer, MeshRenderer, PointLights, RasterizationSettings,
    SoftPhongShader, look_at_view_transform,
)
from pytorch3d.structures import Meshes  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

RUN = os.path.join(os.path.dirname(HERE), "run_cpu1")
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # blue 100..700
PARTS = {"head": "#2a78d6", "mandible": "#eb6834", "antenna": "#eb6834", "thorax": "#1baf7a",
         "waist": "#1baf7a", "wing": "#1baf7a", "gaster": "#eda100", "legs": "#e87ba4"}
CLAY, FIT, SCAN = (0.80, 0.80, 0.78), (0.165, 0.47, 0.84), (0.86, 0.86, 0.85)
DMAX = 0.02


def soft_renderer(size=640):
    elev, azim = RS.VIEWS["dorsolateral"]
    R, T = look_at_view_transform(dist=2.6, elev=elev, azim=azim)
    cam = FoVOrthographicCameras(device=RS.DEV, R=R, T=T, scale_xyz=((1.15, 1.15, 1.15),))
    lights = PointLights(device=RS.DEV, location=[[2.0, 3.0, 2.0]], ambient_color=((0.55, 0.55, 0.55),),
                         diffuse_color=((0.5, 0.5, 0.5),), specular_color=((0.05, 0.05, 0.05),))
    mat = Materials(device=RS.DEV, shininess=8.0)
    return MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cam, raster_settings=RasterizationSettings(image_size=size, faces_per_pixel=1)),
        shader=SoftPhongShader(device=RS.DEV, cameras=cam, lights=lights, materials=mat),
    )


def hex_rgb(h):
    return tuple(int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))


def over(top, bottom):
    a = top[..., 3:4]
    rgb = top[..., :3] * a + bottom[..., :3] * (1 - a)
    return np.dstack([rgb, np.maximum(top[..., 3], bottom[..., 3])])


def main():
    z = np.load(os.path.join(RUN, "probes", "focus_meshes.npz"), allow_pickle=True)
    faces, tv, tf = z["faces"], z["target_verts"], z["target_faces"]
    parts = z["part_of_vertex"]
    tgt = Meshes(verts=[torch.tensor(tv, dtype=torch.float32)], faces=[torch.tensor(tf)])
    torch.manual_seed(0)
    tree = cKDTree(sample_points_from_meshes(tgt, 300_000)[0].numpy())
    cmap = LinearSegmentedColormap.from_list("blue_seq", SEQ)
    r = soft_renderer()
    scan_img = RS.render([(tv, tf, RS.solid(tv, SCAN))], r)
    stages = [("init", "init"), ("Stage_3_deform_fine", "Stage_3")]
    styles = ["A clay", "B ghost (fit over scan silhouette)", f"C distance to scan (0 to {DMAX})", "D anatomical parts"]
    imgs = {}
    for st, lab in stages:
        v = z[st]
        d, _ = tree.query(v)
        imgs[("A", lab)] = RS.render([(v, faces, RS.solid(v, CLAY))], r)
        imgs[("B", lab)] = over(RS.render([(v, faces, RS.solid(v, FIT))], r), scan_img)
        imgs[("C", lab)] = RS.render([(v, faces, cmap(np.clip(d / DMAX, 0, 1))[:, :3])], r)
        imgs[("D", lab)] = RS.render([(v, faces, np.array([hex_rgb(PARTS.get(p, "#898781")) for p in parts]))], r)
        print(lab, "distance fit->scan: median", round(float(np.median(d)), 4), "p95", round(float(np.percentile(d, 95)), 4),
              "frac > 0.01", round(float((d > 0.01).mean()), 3), flush=True)
    y0, y1, x0, x1 = RS.crop_box(list(imgs.values()) + [scan_img])
    fig, ax = plt.subplots(2, 4, figsize=(18, 6.5))
    for j, (key, title) in enumerate(zip("ABCD", styles)):
        for i, (_, lab) in enumerate(stages):
            ax[i, j].imshow(imgs[(key, lab)][y0:y1, x0:x1])
            ax[i, j].set_title(f"{title}: {lab}", fontsize=9)
            ax[i, j].axis("off")
    out = os.path.join(HERE, "style_options_PROBE.png")
    fig.tight_layout()
    fig.savefig(out, dpi=100, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()
