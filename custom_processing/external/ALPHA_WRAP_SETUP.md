# Alpha-wrap reconstruction (CGAL + ManifoldPlus) setup

Used by `custom_processing/prepare_antscan_data_for_mesh_fitting_alphawrap.py`, which replaces
this pipeline's reconstruction core (previously Weld+`holes_fill`+`dissolve_limit`, tiled
winding-number marching cubes, or raw `hjwdzh/Manifold` - see that module's docstring and the
research notes that motivated it) with:

- **Primary: CGAL 3D Alpha Wrapping** (`Alpha_wrap_3/examples/Alpha_wrap_3/triangle_soup_wrap.cpp`)
  - guarantees a watertight, 2-manifold, intersection-free mesh that *strictly contains* the
    input surface - the one property none of this pipeline's prior reconstruction methods had,
    and the direct fix for the leg-fusion failures `close_holes_via_manifold_external` needed
    `verify_gap_preservation`/`verify_reconstruction_fidelity` to catch after the fact.
- **Fallback: ManifoldPlus** (`hjwdzh/ManifoldPlus`) - same author as `hjwdzh/Manifold`
  (already vendored, see `MANIFOLD_SETUP.md`), fixes that tool's documented non-manifold-vertex
  defect natively. Lighter build than CGAL if the alpha-wrap build proves impractical on a given
  machine.

## Status: both binaries build; CLI confirmed from source; not yet run through process_stl()

Built successfully on khaoula's machine 2026-08-13 after two fixes (both already folded into
`setup_alpha_wrap.sh`, see its inline comments):
1. `find_package(CGAL)` needs `-DCGAL_DIR=<path to the cloned CGAL repo>` - CGAL ships a
   `CGALConfig.cmake` at its own repo root specifically so it can be used straight from a git
   checkout, with no separate install step.
2. The build must use the **system** compiler (`/usr/bin/gcc`/`/usr/bin/g++`), not whatever
   conda env is active - conda-forge's compiler build uses a sysroot scoped to the conda env, so
   it never sees apt-installed Boost even with `libboost-dev` present.

`triangle_soup_wrap`'s CLI (argument order, relative-vs-absolute alpha/offset, output filename)
was confirmed by reading `Alpha_wrap_3/examples/Alpha_wrap_3/triangle_soup_wrap.cpp` +
`output_helper.h` directly in the cloned repo, not inferred - see `close_holes_via_alpha_wrap`'s
docstring in the Python file for the exact contract. ManifoldPlus's `--input`/`--output`/`--depth`
flags were likewise confirmed by reading `src/main.cc` + `src/Parser.cc` directly.

**What's still open**: neither method has actually been run end-to-end through `process_stl()` on
a real antscan specimen yet. Per this project's diagnostics rules ("probe before editing",
"preserve diagnostic artifacts"): **do not run a full batch until the validation protocol below
has actually been carried out and its output reviewed.**

## Build

```bash
bash custom_processing/external/setup_alpha_wrap.sh       # primary
bash custom_processing/external/setup_manifoldplus.sh     # fallback
```

Requires (Ubuntu 24.04):
```bash
sudo apt install cmake build-essential libboost-dev libgmp-dev libmpfr-dev libeigen3-dev
```

Produces `custom_processing/external/cgal_alpha_wrap/build/triangle_soup_wrap` and
`custom_processing/external/ManifoldPlus/build/manifold`, both `.gitignore`d - rebuild on each
machine that needs them, same convention as the existing `Manifold` binary.

## Confirmed CLI contracts

- `triangle_soup_wrap <input_file> [relative_alpha=20] [relative_offset=600]`. `relative_alpha`/
  `relative_offset` are ratios of the input's bbox diagonal length, NOT absolute lengths -
  internally `alpha = diag_length / relative_alpha`. Output is always written to the process's
  CWD as `<input_stem>_<int(relative_alpha)>_<int(relative_offset)>.off` (always `.off`,
  regardless of input extension). `close_holes_via_alpha_wrap()` converts its absolute
  `alpha`/`offset` args to this relative form right before invoking the subprocess, runs with
  `cwd=tmp_dir`, computes the exact expected output filename, and loads the `.off` result via
  `trimesh` before re-exporting to `.obj` for Blender (Blender has no native `.off` importer).
- `manifold --input <in.obj> --output <out.obj> --depth <int>` (ManifoldPlus). Confirmed via
  `src/Parser.cc`'s `--key value` pair parsing.

## Required environment fix: scipy/trimesh/rtree inside Blender

Same category of fix as documented in `MANIFOLD_SETUP.md` for `scipy` - applies here too, and
extends to two more packages this file's code paths need inside Blender's own bundled Python
(confirmed missing on khaoula's machine 2026-08-13, one at a time, each only surfacing once the
prior one was fixed and the pipeline reached the next code path that needed it):

- `scipy`: `_derive_alpha_wrap_params` reuses `base._min_self_approach_gap`.
- `trimesh`: `close_holes_via_alpha_wrap` loads the `.off` CGAL produces (Blender has no native
  `.off` importer) and `verify_reconstruction_fidelity` does its Hausdorff-style distance check.
- `rtree`: trimesh's own dependency, needed by `mesh.triangles_tree` /
  `trimesh.proximity.closest_point` - without it, `verify_reconstruction_fidelity` (the mandatory
  fidelity gate) fails with `ModuleNotFoundError: No module named 'rtree'` partway through an
  otherwise-successful run (confirmed: this fired only after reconstruction, decimation, and hole
  checks had already all passed cleanly on Metapone_emersoni_CASENT0745558).

```bash
<path-to-blender-4.2>/4.2/python/bin/python3.11 -m pip install scipy trimesh rtree
```

## Validation protocol before a full batch run

Mirrors the research notes' recommended rollout (do not skip steps):

1. Pick 3-4 ground-truth specimens: one compact/simple, one with long thin legs, one with
   antennae/mandibles (self-approach-heavy), one previously known to have visually deformed after
   reconstruction under an earlier pipeline version.
2. Build both binaries (above), confirm their CLIs by hand.
3. Run `compare_reconstruction_methods(stl_path, output_dir)` from
   `prepare_antscan_data_for_mesh_fitting_alphawrap.py` on each ground-truth specimen.
4. Check `processed_fidelity_gate_passed` / `processed_worst_fidelity_deviation` in each output
   JSON, **and** open both exported OBJs in Blender/MeshLab and look at them directly - the
   research notes are explicit that trusting the log alone was exactly the mistake that produced
   this rewrite (`is_watertight=False` on 44/44 "successful" specimens went unnoticed under the
   prior pipeline's logging alone).
5. Only once both numeric and visual checks pass on the small set, run a ~10-specimen batch and
   spot-check a few outputs again before running the full corpus.
