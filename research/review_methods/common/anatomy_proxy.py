"""Expert-skeleton proxy labels: give every surface point an anatomical label from a skeleton alone.

Real scans have no dense correspondence ground truth, but JAB supplies expert joint positions. A
point is labelled by the nearest skeleton bone (segment from a joint to its child, using the
model's kinematic tree), and the bone carries the name of its PROXIMAL joint -- the joint whose
rotation moves it, which is how the template's dominant-skinning-weight labels are defined.
Joints that were not placed are bridged: a placed joint connects to its nearest placed ancestor.
Leaf joints (no placed child) contribute a zero-length bone at their own position.

The proxy is only trustworthy at resolutions that pass a validation gate on synthetic data with
true labels (Q3a A2). Use `coarse()` to read labels at a chosen resolution.
"""
import os
import pickle

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
MODEL = os.environ.get("SMILIFY_SMAL_FILE", "3D_model_prep/OmniAnt_25PCs_joint_limited.pkl")


def load_dd(path=MODEL):
    with open(os.path.join(REPO, path), "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        return u.load()


def joint_tree(dd):
    names = [str(x) for x in np.asarray(dd["J_names"]).ravel()]
    kt = np.asarray(dd["kintree_table"])
    parent = {}
    for i in range(kt.shape[1]):
        c, p = int(kt[1, i]), int(kt[0, i])
        parent[names[c]] = names[p] if (i > 0 and 0 <= p < len(names)) else None
    return names, parent


def template_vertex_labels(dd):
    """Per template vertex: name of the dominant skinning joint (the definition labels.py uses)."""
    names = [str(x) for x in np.asarray(dd["J_names"]).ravel()]
    dom = np.asarray(dd["weights"]).argmax(1)
    return np.array([names[j] for j in dom], dtype=object)


def build_bones(joints, parent):
    """joints: {name: (3,)} placed joints only. Returns (A (B,3), Bv (B,3), proximal_name list)."""
    def placed_ancestor(n):
        p = parent.get(n)
        while p is not None and p not in joints:
            p = parent.get(p)
        return p

    A, Bv, lab = [], [], []
    has_child = set()
    for n in joints:
        p = placed_ancestor(n)
        if p is not None:
            A.append(joints[p]); Bv.append(joints[n]); lab.append(p)
            has_child.add(p)
    for n in joints:                       # leaves: zero-length bone at the joint itself
        if n not in has_child:
            A.append(joints[n]); Bv.append(joints[n]); lab.append(n)
    return np.asarray(A, float), np.asarray(Bv, float), lab


def label_points(P, joints, parent, exclude=()):
    """Nearest-bone proximal-joint name for each point in P (N,3)."""
    joints = {k: np.asarray(v, float) for k, v in joints.items() if k not in exclude}
    A, B, lab = build_bones(joints, parent)
    AB = B - A
    L2 = np.maximum((AB ** 2).sum(1), 1e-12)
    out = np.empty(len(P), dtype=object)
    dist = np.empty(len(P))
    for s in range(0, len(P), 4096):
        p = P[s:s + 4096, None, :]                                   # (n,1,3)
        t = np.clip(((p - A[None]) * AB[None]).sum(-1) / L2[None], 0.0, 1.0)
        d = ((p - (A[None] + t[..., None] * AB[None])) ** 2).sum(-1)  # (n,B)
        k = d.argmin(1)
        out[s:s + 4096] = np.asarray(lab, dtype=object)[k]
        dist[s:s + 4096] = np.sqrt(d[np.arange(len(k)), k])
    return out, dist


def coarse(name, level):
    """Map a joint name to a label at a resolution.

    level 'leg'      : 'l1_r'.. for leg joints, 'other' otherwise
    level 'leg_seg'  : 'l1_r_fe'.. for leg joints, 'other' otherwise
    level 'appendage': leg id, 'ant_r'/'ant_l', 'ma_r'/'ma_l', or 'trunk' (body + head + wings)
    """
    n = str(name)
    if n.startswith("l_"):
        b = n.split("_")
        leg = f"l{b[1]}_{b[-1]}"
        if level == "leg":
            return leg
        if level == "leg_seg":
            return f"{leg}_{b[2]}"
        return leg
    if level in ("leg", "leg_seg"):
        return "other"
    if n.startswith("an_"):
        return f"ant_{n[-1]}"
    if n.startswith("ma_"):
        return n
    return "trunk"


def coarse_vec(names, level):
    return np.array([coarse(n, level) for n in names], dtype=object)
