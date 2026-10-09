"""
01_extract_joint_schema.py

Run this on the cluster, same env you've been using for the CSE / B2 work.
Purpose: pull J_names and kintree_table directly out of the model pkl, so every
annotation point we design in Phase 2 maps 1:1 to a real model joint index —
no hand-typed joint names, no fuzzy matching later.

Usage:
    python 01_extract_joint_schema.py

Paste the full printed output back into chat — that's what Phase 2 is built from.
"""

import pickle
import json
import os

MODEL_PATH = os.environ.get("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")

with open(MODEL_PATH, "rb") as f:
    u = pickle._Unpickler(f)
    u.encoding = "latin1"
    dd = u.load()

print("=" * 70)
print(f"Loaded: {MODEL_PATH}")
print("=" * 70)

print("\nAll keys in model dict:")
print(sorted(dd.keys()))

J_names = dd.get("J_names", None)
kintree = dd.get("kintree_table", None)

print("\n--- J_names ---")
if J_names is not None:
    for i, name in enumerate(J_names):
        print(f"  joint {i}: {name}")
else:
    print("  NOT FOUND under key 'J_names' — check the key list above for the actual name")

print("\n--- kintree_table (parent/child joint index pairs) ---")
if kintree is not None:
    print("  shape:", getattr(kintree, "shape", None))
    print("  raw:")
    print(" ", kintree)
    # kintree_table is typically 2 x N_JOINTS: row 0 = parent idx, row 1 = child idx
    # -1 or a sentinel usually marks the root
    if J_names is not None and hasattr(kintree, "shape") and len(kintree.shape) == 2:
        print("\n  Human-readable parent -> child (using J_names):")
        n = kintree.shape[1]
        for i in range(n):
            parent_idx = int(kintree[0, i])
            child_idx = int(kintree[1, i])
            parent_name = J_names[parent_idx] if 0 <= parent_idx < len(J_names) else "ROOT/NONE"
            child_name = J_names[child_idx] if 0 <= child_idx < len(J_names) else "?"
            print(f"    {parent_name}  ->  {child_name}")
else:
    print("  NOT FOUND under key 'kintree_table' — check the key list above")

# Also dump J (the actual rest-pose 3D joint locations baked into the template)
J = dd.get("J", None)
print("\n--- J (rest-pose 3D joint locations, shape) ---")
if J is not None:
    print("  shape:", getattr(J, "shape", None))
else:
    print("  NOT FOUND under key 'J'")

# Save a clean JSON summary for Phase 2 to consume programmatically
out = {
    "model_path": MODEL_PATH,
    "J_names": list(J_names) if J_names is not None else None,
    "n_joints": len(J_names) if J_names is not None else None,
}
out_path = "joint_schema_summary.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nWrote {out_path} — keep this, Phase 2 uses it.")