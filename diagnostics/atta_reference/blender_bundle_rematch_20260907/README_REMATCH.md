# Rematch bundle — 2026-09-07

Fresh fit, packaged for a single uninterrupted Blender sitting so there is zero gap between
npz generation and CSV export this time.

## Files
- `ATTA20_ARM_A_rematch.npz` — `Stage_3_deform_fine.npz` from SLURM job 3790072, run on
  worktree `/hpcwork/nao48500/atta20_master` @ `b5bf9565`, same config
  (`cfg_arm_a_master_rerun_20260907.yaml`, identical to `cfg_arm_a_master.yaml` except
  `results_dir`) and the same 20 meshes at `/hpcwork/nao48500/atta20/{01..20}.obj`. Produced
  2026-09-07 10:31.
- `OmniAnt_25PCs_joint_limited.pkl` — sha256 `1cc3c82d...87cddea`, confirmed byte-identical to
  `blender_bundle/OmniAnt_25PCs_joint_limited.pkl`.
- `smil_importer.zip` — same addon build as the original bundle.
- `ant_body_lengths.csv` — same reference-length CSV used for the Morphometry panel.

## What to do in Blender (one sitting, do not save/reopen partway)
1. Install/enable `smil_importer.zip` if not already active.
2. SMPL tab, **Direct Import SMIL Model**, all processing toggles **off**:
   - `pkl_filepath = OmniAnt_25PCs_joint_limited.pkl`
   - `npz_filepath = ATTA20_ARM_A_rematch.npz`
   - `shapekeys_from_PCA = False`, `clean_mesh = False`, `symmetrise = False`,
     `regress_joints = False`
3. Add/verify the `b_h_l` / `b_h_r` reporter bones as children of `b_h`, at the widest point of
   the head capsule (per `REPORT_ATTA_HEADWIDTH.md` §"Head-width reporter bones") — same
   placement procedure as before, checked against the **Basis** shape.
4. Load `ant_body_lengths.csv` in the Morphometry panel (`Joint Pair: b_t to b_a_5 [mm]`,
   confirm `Number of Shapes: 20`).
5. Export **Joint Distances** with the mesh selected. Confirm the console line:
   `[joint distances] J_regressor: 55 trained rows preserved, 2 reporter rows computed (b_h_l, b_h_r)`.
6. Save the resulting CSV as:
   `diagnostics/atta_reference/blender_export_rematch_20260907/OmniAnt_25PCs_joint_limited_joint_distances.csv`
   (create that directory; keep the same filename pattern as the original export).

Do steps 2-5 back-to-back without saving/reopening the .blend in between — that is the whole
point of this bundle (close the timestamp gap that Step 0 flagged).
