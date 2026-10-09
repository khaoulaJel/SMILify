"""Stratified TARGET sampling by anatomical face group (Task 7, Intervention C).

See `diagnostics/registration_interventions/MECHANISM_C.md` for the full pre-registered design,
leakage analysis, and quota rationale. Summary: draws the SAME total `n_sample` as baseline, but
guarantees a fixed fraction lands on distal (tibia/tarsus/pretarsus, pooled across all 6 legs)
faces instead of pure area-weighting. Face group membership comes ONLY from the TEMPLATE's own
skinning weights (`weights.argmax(1)`), valid only on topology-preserving corpora where every
specimen's target mesh shares the template's face indexing (verified by the caller, not this
module -- see `MECHANISM_C.md`'s leakage section). No ground truth, no fitted correspondence, no
learned model anywhere in this file.

When `distal_quota is None` this module is not used at all -- callers keep calling
`pytorch3d.ops.sample_points_from_meshes` exactly as before, so baseline behaviour is provably
unaffected (see `HierarchicalStage.run()`'s call site).
"""

import torch


def distal_face_mask(dd, jnames):
    """(F,) bool tensor, True where a face's DOMINANT skinning joint is ti/ta/pt on any leg.

    Purely a function of the template's own weights/topology -- no scan, no GT.
    """
    weights = torch.as_tensor(dd["weights"])
    faces = torch.as_tensor(dd["f"], dtype=torch.long)
    dom = weights.argmax(dim=1)  # (V,) dominant joint per template vertex

    def is_distal_joint(j):
        n = jnames[j]
        return n.startswith("l_") and n.split("_")[2] in ("ti", "ta", "pt")

    vertex_distal = torch.tensor([is_distal_joint(j) for j in dom.tolist()])
    face_vertex_distal = vertex_distal[faces]  # (F,3)
    # a face counts as distal if a MAJORITY (>=2 of 3) of its vertices are distal-dominant --
    # same majority-vote convention as `face_group_by_segment` (Task 6 D2a/D2b), not invented here
    return face_vertex_distal.sum(dim=1) >= 2


def _face_areas(verts, faces):
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    return 0.5 * torch.linalg.cross(v1 - v0, v2 - v0, dim=-1).norm(dim=-1)


def _sample_from_face_pool(verts, faces, face_ids, n, generator=None):
    """n area-weighted samples restricted to `face_ids` (a 1-D index tensor into `faces`)."""
    if n <= 0 or face_ids.numel() == 0:
        return torch.zeros(0, 3, dtype=verts.dtype, device=verts.device)
    sub_faces = faces[face_ids]
    areas = _face_areas(verts, sub_faces)
    p = areas / areas.sum().clamp_min(1e-12)
    chosen = torch.multinomial(p, n, replacement=True, generator=generator)
    f = sub_faces[chosen]
    v0, v1, v2 = verts[f[:, 0]], verts[f[:, 1]], verts[f[:, 2]]
    # barycentric coords, same scheme pytorch3d's sample_points_from_meshes uses
    u = torch.rand(n, device=verts.device, generator=generator)
    v = torch.rand(n, device=verts.device, generator=generator)
    swap = (u + v) > 1
    u = torch.where(swap, 1 - u, u)
    v = torch.where(swap, 1 - v, v)
    w = 1 - u - v
    return u[:, None] * v0 + v[:, None] * v1 + w[:, None] * v2


def leg_face_mask(dd, jnames):
    """(F,) bool tensor, True where a face's DOMINANT skinning joint is any leg joint (l_*).

    Purely a function of the template's own weights/topology -- no scan, no GT. Same
    majority-vote convention as `distal_face_mask`.
    """
    weights = torch.as_tensor(dd["weights"])
    faces = torch.as_tensor(dd["f"], dtype=torch.long)
    dom = weights.argmax(dim=1)
    vertex_leg = torch.tensor([jnames[j].startswith("l_") for j in dom.tolist()])
    face_vertex_leg = vertex_leg[faces]
    return face_vertex_leg.sum(dim=1) >= 2


def sample_target_distal_protected(meshes, faces, n_sample, leg_mask, dist_mask, distal_quota,
                                    template_verts, generator=None):
    """Rank 5 / Section 11 C5 correction to Intervention C's `distal_quota` (see
    diagnostics/overnight_20260818/OVERNIGHT_REPORT.md, Rank 5): `sample_target_stratified`
    reallocates budget from ALL non-distal faces -- including antenna, head, body -- toward
    distal leg faces, which is the documented mechanism behind Intervention C's antenna
    regression (Section 4, Section 11 H3-vs-H7). Here `distal_quota` is applied ONLY inside a
    LEG sub-budget; non-leg anatomy (body/head/mandible/antenna) is drawn area-weighted from its
    own protected pool whose SIZE is fixed at the template's baseline leg-area fraction,
    independent of `distal_quota` by construction. So non-leg sampling density cannot regress
    no matter how large `distal_quota` is -- if H7 (Section 6) was purely a target-side budget
    theft artifact, THIS sampler should show distal improvement with no antenna regression; if
    the antenna still regresses, H7 is confirmed as a fundamental constraint, not a sampler
    artifact (this is the falsifier the master prompt specifies).

    template_verts: (V,3) the TEMPLATE's rest-pose vertices, used ONLY to compute a fixed
    leg-area fraction once (deterministic, no per-specimen leakage).
    """
    leg_ids = leg_mask.nonzero(as_tuple=True)[0]
    nonleg_ids = (~leg_mask).nonzero(as_tuple=True)[0]
    leg_areas = _face_areas(template_verts, faces[leg_ids])
    total_areas = _face_areas(template_verts, faces)
    leg_area_frac = float(leg_areas.sum() / total_areas.sum().clamp_min(1e-12))

    n_leg = int(round(n_sample * leg_area_frac))
    n_nonleg = n_sample - n_leg
    n_leg_distal = int(round(n_leg * distal_quota))
    n_leg_proximal = n_leg - n_leg_distal

    leg_distal_ids = leg_ids[dist_mask[leg_ids]]
    leg_proximal_ids = leg_ids[~dist_mask[leg_ids]]

    verts_list = meshes.verts_list()
    device = verts_list[0].device
    faces = faces.to(device)
    out = []
    for verts in verts_list:
        p_nonleg = _sample_from_face_pool(verts, faces, nonleg_ids.to(device), n_nonleg, generator=generator)
        p_ld = _sample_from_face_pool(verts, faces, leg_distal_ids.to(device), n_leg_distal, generator=generator)
        p_lp = _sample_from_face_pool(verts, faces, leg_proximal_ids.to(device), n_leg_proximal, generator=generator)
        pts = torch.cat([p_nonleg, p_ld, p_lp], dim=0)
        assert pts.shape[0] == n_sample, f"expected {n_sample} points, got {pts.shape[0]}"
        out.append(pts)
    return torch.stack(out, dim=0)


def sample_leg_nonleg_split(meshes, faces, n_sample, leg_mask, template_verts, generator=None):
    """(leg_pts, nonleg_pts), each (B, n_i, 3), split by the template's own baseline leg-area
    fraction (same convention as `sample_target_distal_protected`, computed once from
    `template_verts`, no per-specimen leakage). Used by MoonshotStage's `robust_leg_only`
    (cycle2_20260819 B1 extra arm) to give leg and non-leg residuals independent robust-kernel
    treatment -- see fitter_3d/trainer_moonshot.py:MoonshotStage.forward.
    """
    leg_ids = leg_mask.nonzero(as_tuple=True)[0]
    nonleg_ids = (~leg_mask).nonzero(as_tuple=True)[0]
    leg_areas = _face_areas(template_verts, faces[leg_ids])
    total_areas = _face_areas(template_verts, faces)
    leg_area_frac = float(leg_areas.sum() / total_areas.sum().clamp_min(1e-12))
    n_leg = int(round(n_sample * leg_area_frac))
    n_nonleg = n_sample - n_leg

    verts_list = meshes.verts_list()
    device = verts_list[0].device
    faces = faces.to(device)
    leg_pts, nonleg_pts = [], []
    for verts in verts_list:
        leg_pts.append(_sample_from_face_pool(verts, faces, leg_ids.to(device), n_leg, generator=generator))
        nonleg_pts.append(_sample_from_face_pool(verts, faces, nonleg_ids.to(device), n_nonleg, generator=generator))
    return torch.stack(leg_pts, dim=0), torch.stack(nonleg_pts, dim=0)


def sample_target_stratified(meshes, faces, n_sample, dist_mask, distal_quota, generator=None):
    """(B, n_sample, 3) target points, `distal_quota` fraction from `dist_mask` faces.

    meshes: a pytorch3d Meshes batch whose per-specimen face indexing matches `faces`
    (template-indexed) exactly -- caller's responsibility (see MECHANISM_C.md leakage section).
    faces: (F,3) LongTensor, template face indices (shared across the batch).
    dist_mask: (F,) bool, from `distal_face_mask`.
    distal_quota: float in [0,1].
    """
    verts_list = meshes.verts_list()
    device = verts_list[0].device
    faces = faces.to(device)
    distal_ids = dist_mask.nonzero(as_tuple=True)[0].to(device)
    nondistal_ids = (~dist_mask).nonzero(as_tuple=True)[0].to(device)
    n_distal = int(round(n_sample * distal_quota))
    n_nondistal = n_sample - n_distal

    out = []
    for verts in verts_list:
        pd = _sample_from_face_pool(verts, faces, distal_ids, n_distal, generator=generator)
        pn = _sample_from_face_pool(verts, faces, nondistal_ids, n_nondistal, generator=generator)
        pts = torch.cat([pd, pn], dim=0)
        assert pts.shape[0] == n_sample, f"expected {n_sample} points, got {pts.shape[0]}"
        out.append(pts)
    return torch.stack(out, dim=0)
