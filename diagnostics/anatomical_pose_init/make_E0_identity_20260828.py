"""E0 -- build the identity "fit": the ground-truth mesh presented to the audit AS the fit.

No fit is run. Placement is exactly correct by construction, so auditing this measures the
CORRESPONDENCE METRIC'S OWN FLOOR -- what `co` leg-level error reads when nothing is wrong.

Why this exists: every coxal number this session (C13, C14p, the three-mechanisms-converge
finding) was reported against an implicit zero point of 0. E0 measures the real floor at
co = 0.0908 -- 8x the next-worst segment, ~300x `ti` -- because adjacent coxae sit close enough
that nearest-vertex matching misassigns ~9% of coxal points even on an exact mesh. The addressable
coxal residual is therefore ~0.32, not ~0.41.

It also settles representability on this corpus: a perfect parameter set scores 0.09, not 0.41,
so parameters placing the coxa well demonstrably exist. P48 targets are generated FROM the model
(`make_synth_corpus.py`), so this was expected -- E0 makes it measured rather than asserted.

The output npz is ~29MB and derivable in seconds, so it is gitignored; this script is the record.

Run: `python diagnostics/anatomical_pose_init/make_E0_identity_20260828.py` (pytorch3d env).
Audited via the `E0_identity` entry in `diagnostics/correspondence_accuracy/run_audit.py`.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
GT = os.path.join(REPO, "diagnostics/moonshot/synth_power48/ground_truth.npz")
# faces are not stored in ground_truth.npz; borrow them from any fit on the same corpus (all
# arms share template topology -- run_audit asserts this).
FACES_FROM = os.path.join(REPO, "diagnostics/moonshot/runs/C14p_gtinit_nocse/Stage_3_deform_fine.npz")
OUT = os.path.join(HERE, "out_E0_identity_20260828", "identity.npz")


def main():
    gt = np.load(GT, allow_pickle=True)
    faces = np.load(FACES_FROM)["faces"]
    assert faces.shape[0] == gt["verts"].shape[0], "specimen count mismatch"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(
        OUT,
        verts=gt["verts"],  # the target itself, verified byte-identical to the .obj files
        faces=faces,
        labels=np.array([f"{n}.obj" for n in gt["names"]]),
    )
    print(f"wrote {OUT}  verts={gt['verts'].shape}")


if __name__ == "__main__":
    main()
