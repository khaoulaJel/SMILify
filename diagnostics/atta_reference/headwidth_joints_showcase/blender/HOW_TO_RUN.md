# Running the Blender screenshots

The script needs Blender 4.2 with the SMIL Model Importer. It won't run on the cluster, which has
no Blender, so it has to run on your laptop.

## 0. Copy the files to the laptop

Create the destination folder first, because `scp` won't create it:

```powershell
mkdir C:\atta_blender
scp <cluster>:~/SMILify/diagnostics/atta_reference/headwidth_joints_showcase/blender/blender_screenshots.py C:\atta_blender\
scp <cluster>:~/SMILify/diagnostics/atta_reference/blender_export/bone_placement.blend C:\atta_blender\
scp -r <cluster>:~/SMILify/diagnostics/atta_reference/blender_bundle_rematch_20260907 C:\atta_blender\
```

## Route A: matches the figures (recommended, ~10 min)

The figures use the **rematch** fit. `bone_placement.blend` holds the Sep 6 session, whose specimen
shapes can't be reproduced, so build a fresh scene from the rematch bundle.

1. Start Blender and enable `smil_importer.zip` (Edit → Preferences → Add-ons → Install).
2. In the **SMPL tab**, use **Direct Import SMIL Model** with
   `blender_bundle_rematch_20260907\OmniAnt_25PCs_joint_limited.pkl` and `ATTA20_ARM_A_rematch.npz`.
   Turn **all** processing toggles OFF: PCA, clean_mesh, symmetrise, regress_joints.
3. Bring in the two bones with their exact original placement: **File → Append →**
   `bone_placement.blend` **→ Object →** select the armature (`…_Armature`). Blender adds it as
   `…_Armature.001`, which is fine; the script uses whichever armature contains `b_h_l`/`b_h_r`.
   *(Alternatively, add the two bones by hand as before, parented to `b_h`, with every shape key at 0.)*
4. **Save the .blend** into `C:\atta_blender\`. The images are written next to it.
5. **Scripting tab → Open → `blender_screenshots.py` → Run Script** (▶).
6. Check `C:\atta_blender\blender_screens\00_template_anterior.png` first. If the head is upside
   down or seen side-on, stop and send it to me.

## Route B: quickest (~2 min), but the per-specimen shots won't match the figures

Open `bone_placement.blend` and run the script (Scripting tab → Run Script), or from PowerShell:

```powershell
& "C:\Program Files\Blender Foundation\Blender 4.2\blender.exe" -b C:\atta_blender\bone_placement.blend -P C:\atta_blender\blender_screenshots.py -- C:\atta_blender\blender_screens
```

(PowerShell needs the leading `&`; cmd.exe doesn't.)

The `00_template_*` shots are valid either way, because the bone placement is identical in both
sessions. The `01…20` shots, however, show the Sep 6 shapes rather than the ones in the figures.

## What you get
- `00_template_anterior.png`, `…_dorsal`, `…_lateral`: **head capsule only**, with both joints as spheres
- `00_template_anterior_xray.png`: semi-transparent head showing the head-width bar between the joints
- `00_template_anterior_context.png`: the full model (legs and antennae visible) for orientation
- `00_whole_model_dorsal.png`
- `01_head_anterior.png`, `01_head_anterior_xray.png`, `01_head_dorsal.png` … `20_…`: each specimen
  with its **regressed** joints

"Head only" works through a Mask modifier (`HW_head_only`, vertex group `b_h`) that is switched on
for rendering only. The joint computation always uses the full mesh. To remove the modifier,
delete it in the Modifier tab.

The console prints the maximum regressor weight for each bone, which should be ≈0.30 for `b_h_l` and
≈0.80 for `b_h_r`. That's a quick check that it found the same vertices as the Python analysis.

## Screenshots worth taking by hand (Fabian asked for "screenshots")

Renders don't show the Blender interface, and these four UI shots do most of the explaining:
1. **Outliner** with the armature expanded: `b_t → b_h → b_h_l, b_h_r` next to `ma_l`, `an_1_l` and the others.
2. **Bone Properties → Relations** for `b_h_l`: **Parent = b_h**.
3. **Viewport, full-face, head close-up**: armature in Edit Mode with *In Front* on, so both bones
   show on the head surface.
4. The **console line** after Export Joint Distances:
   `J_regressor: 55 trained rows preserved, 2 reporter rows computed (b_h_l, b_h_r)`.

## If it fails
- *"No armature contains bones b_h_l and b_h_r"*: step 3 was skipped.
- *"No mesh with specimen shape keys"*: the import in step 2 didn't happen, or PCA was on.
- Images come out black or empty: send the console output; the camera framing is the untested part.
