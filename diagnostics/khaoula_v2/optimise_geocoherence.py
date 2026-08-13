"""Fork of fitter_3d/optimise_moonshot.py, swapping MoonshotStage for
GeoCoherentMoonshotStage (diagnostics/khaoula_v2/trainer_geocoherence.py). Two changes from the
original, everything else byte-identical: the import, and the stage-construction line (passes
--geodesic_npz through). fitter_3d/optimise_moonshot.py and fitter_3d/trainer_moonshot.py are
NOT modified -- this is an isolated fork, same discipline as within_part_signal_dino.py.
"""

import argparse
import gc
import glob
import os
import sys
import warnings

import numpy as np
import torch
import yaml

warnings.filterwarnings("ignore", message=".*torch.sparse.SparseTensor.*")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "diagnostics", "moonshot"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # trainer_geocoherence.py

import config  # noqa: E402
from fitter_3d.utils import load_meshes  # noqa: E402
from fitter_3d.trainer import SMAL3DFitter  # noqa: E402
from fitter_3d.trainer_moonshot import MoonshotManager  # noqa: E402
from trainer_geocoherence import GeoCoherentMoonshotStage  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh_dir", type=str, required=True)
    p.add_argument("--yaml_src", type=str, required=True)
    p.add_argument("--results_dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shape_family_id", type=int, default=-1)
    p.add_argument("--max_meshes", type=int, default=-1)
    p.add_argument("--eval", action="store_true")
    p.add_argument("--quiet", action="store_true", default=True)
    p.add_argument("--init_from", type=str, default=None)
    p.add_argument(
        "--geodesic_npz",
        type=str,
        default=os.path.join(REPO, "diagnostics", "khaoula_v2", "out", "template_geodesic.npz"),
        help="cached anchor-to-all geodesic distances (train_refine_autoencoder.py's output), "
        "reused here, not recomputed",
    )
    p.add_argument("--geocoh_sigma_frac", type=float, default=0.05)
    return p


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.yaml_src) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    stage_options = cfg["stages"]
    for k, v in (cfg.get("args") or {}).items():
        if getattr(args, k, None) is None or k not in ("results_dir",):
            setattr(args, k, v)
    if args.results_dir is None:
        args.results_dir = cfg.get("args", {}).get("results_dir", "moonshot_results")

    os.makedirs(args.results_dir, exist_ok=True)

    mesh_files = sorted(glob.glob(os.path.join(args.mesh_dir, "*.obj")))
    if args.max_meshes > 0:
        mesh_files = mesh_files[: args.max_meshes]
    mesh_names = [os.path.basename(f) for f in mesh_files]
    print(f"[geocoh] {len(mesh_files)} meshes from {args.mesh_dir}", flush=True)

    _, target_meshes = load_meshes(mesh_files=mesh_files, device=device)
    n = len(target_meshes)

    smal = SMAL3DFitter(batch_size=n, device=device, shape_family=args.shape_family_id)

    if args.init_from:
        d = np.load(args.init_from, allow_pickle=True)
        with torch.no_grad():
            for k in ["global_rot", "joint_rot", "betas", "log_beta_scales", "trans", "betas_trans", "deform_verts"]:
                if k in d:
                    getattr(smal, k).data = torch.tensor(d[k], dtype=torch.float32, device=device)
        print(f"[geocoh] initialised pose/shape from {args.init_from}", flush=True)

    manager = MoonshotManager(out_dir=args.results_dir, labels=mesh_names)

    common = dict(
        target_meshes=target_meshes,
        smal_3d_fitter=smal,
        out_dir=args.results_dir,
        device=device,
        mesh_names=mesh_names,
    )

    for stage_name, kw in stage_options.items():
        kw = dict(kw)
        stage = GeoCoherentMoonshotStage(
            name=stage_name, **kw, **common,
            geodesic_npz_path=args.geodesic_npz, geocoh_sigma_frac=args.geocoh_sigma_frac,
        )
        manager.add_stage(stage)

    print(f"[geocoh] running {len(manager.stages)} stages", flush=True)
    manager.run()
    manager.plot_losses("losses")

    del smal, manager, target_meshes
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main(build_parser().parse_args())
