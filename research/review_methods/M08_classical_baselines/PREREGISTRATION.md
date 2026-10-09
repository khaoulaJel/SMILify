# M08: classical non-rigid registration baselines

Written 2026-10-07, before any baseline was run on any data.

## Question

Can established non-rigid registration, with no learned correspondence, supply anatomical
correspondence that the SMILify fitter can use? A baseline, not a candidate contribution: if it fails,
that supports "the missing information is anatomical identity, not geometric registration power"; if
it works, that is more important still.

## State of the art, checked before choosing methods

- **CPD** (Myronenko & Song, TPAMI 2010) has been superseded as the classical point-set baseline by
  **BCPD** (Hirose, TPAMI 2020: variational Bayesian CPD, convergence guaranteed, joint
  rigid + non-rigid) and **GBCPD** (Hirose, TPAMI 2022: motion coherence defined by *geodesic* rather
  than Euclidean distance on the source shape). GBCPD targets our failure directly: a Euclidean
  coherence kernel couples points that are close in space (adjacent legs), a geodesic kernel only
  points that are close along the surface.
- **Optimal-Step NICP** (Amberg, Romdhani & Vetter, CVPR 2007): template-mesh deformation with a
  stiffness schedule; the reference implementation `trimesh.registration.nricp_amberg` is used.
- Implementations: official BCPD/GBCPD C code (github.com/ohirose/bcpd, MIT, commit b7b6d01),
  vendored in `third_party/bcpd`, compiled against the conda env's LAPACK/BLAS; trimesh 5.0.0.

## Methods (one shared start: template at rest, normalised exactly like the scan)

| id | method | output used |
|---|---|---|
| BCPD | BCPD, Gaussian kernel (CPD family) | per-template-vertex matched target point (`-s e`), mask = non-outlier label (`-s c`) |
| GBCPD | BCPD with the geodesic kernel on the template's own triangles (`-G geodesic,tau,triangles`) | same |
| NICP | Amberg NICP from the template mesh to the scan mesh, after rigid ICP with scale | final template vertex positions projected to the nearest scan point; mask all |

Output format is the fitter's correspondence interface (`--cse_correspondence_from`: names, verts,
mask in the normalised frame), so each baseline is consumed exactly like the CSE targets.

## Tuning protocol (fixed now)

- Grid: BCPD beta in {0.3, 1.0, 2.0} x lambda in {1, 10, 100}; GBCPD the same x tau in {0.2, 0.5};
  omega 0.1, `-ux -g0.1` (inputs roughly pre-aligned, per the BCPD README recommendation). NICP:
  trimesh default stiffness schedule, and the same schedule x 0.5 and x 2.
- **Tuning set:** 48 synthetic specimens with articulation from P0 `pose_train` fits (genera never in
  JAB, never in `pose_eval`). Never JAB, never `pose_eval`.
- **Selection criterion:** mean leg-correctness of the targets (family C, true identity). Not chamfer
  (RULES_AUDIT R6). Ties (within 0.01) go to the larger beta / stiffer schedule (smoother map).

## Evaluation

1. **Correspondence (C):** selected configurations on the S2-confirm REALPOSE set (pose_eval
   articulation) and on C0: target leg-correctness and geodesic error / sqrt(area), next to the CSE
   targets on the same specimens.
2. **Fitter use (S, G):** P48 and the S2-confirm REALPOSE set, D1_PROD with the baseline targets in
   all stages (as JAB I_cse), 3 seeds, vs D1_PROD and vs D1_PROD + CSE.
3. **JAB, once:** the better of BCPD / GBCPD on the tuning set takes the frozen "CPD" slot of the
   Holm family (PROTOCOL §3.3, m = 6 unchanged); NICP takes the "NICP" slot.

## Expected outcome (stated in advance)

On C0 (training-like pose) all three should give reasonable leg-level correspondence, since the
template starts near the answer. Under real-level articulation, Euclidean-kernel BCPD is expected to
confuse adjacent legs; GBCPD should do better; whether any beats the CSE targets is open.
