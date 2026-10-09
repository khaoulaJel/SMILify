# What we want the paper to convey, and the evidence that carries it

Draft 2026-10-08, for Khaoula's approval before anything enters the Task 3-5 pages. Fabian's frame
(negative result with a diagnosis; lead with the failure, then what survived) is kept. This page fixes
*our* message inside that frame and picks, from everything on disk including work done after his
package, the results that carry it. Each item lists source, specimen set, n, and verification status.
`[V]` = numbers read from the source file this session; `[R]` = must be re-derived before use;
`[NEW]` = run after Fabian's package (2026-10-06/08, `research/review_methods/`), Fabian decides.

## The message (five claims)

**C1. The failure is quantifiable on anatomy, not just on surfaces.** Automatic registration of the
uncleaned AntScan meshes does not reach curated quality, measured against expert joints.

**C2. A good surface fit does not certify a correct anatomy.** Surface agreement and skeletal accuracy
come apart, so surface metrics cannot be used to accept a registration.

**C3. The cause is anatomical identity under real articulation.** Preserved specimens are posed far
outside what the model and its initialisers expect; leg identity (which leg is which) is what breaks;
and the surface objective's vertex-density asymmetry pulls legs into the gaster.

**C4. What would change it: correct, dense anatomical correspondence -- and only if leg identity is
right.** Correspondence is the lever with by far the largest measured effect, but its value is
conditional on getting the leg identity right under real articulation.

**C5. Biology survives where it is robust.** Head width and the genus-level shape signal hold up
despite imperfect registration.

## Evidence per claim

### C1 -- the failure, on expert joints (Task 4 table; Task 5 opening)

| result | numbers | source | n / set | status |
|---|---|---|---|---|
| Production recipe vs expert joints | median FK joint error **25.1% of Weber's length**; no single-factor change of 11 improves it (best mean rank, Friedman p 0.011) | `diagnostics/joint_alignment_benchmark/REPORT.md` | 11 real AntScan specimens, 3 seeds, frozen pre-registration | [V] |
| Where it fails | coxa ~12, trochanter/femur ~13, body axis 24, mandible 43, antenna 55, tibia/tarsus 56% WL | same, regions | same | [V] |

Why it belongs: Fabian's "none of the strategies fits at curated quality" currently rests on surface
metrics; JAB states it on anatomy with a frozen protocol. Not in his package -> raise with him.

### C2 -- surface fit is not anatomy (Task 4 table; Task 5)

| result | numbers | source | n / set | status |
|---|---|---|---|---|
| Removing the offset/normal penalty | best surface of all arms (**0.45x chamfer**) and a worse skeleton (joint-level **x1.25** [1.15, 1.37]) | JAB REPORT, arm J | 11 real, 3 seeds | [V] |
| Correspondence on real scans: surface vs skeleton | surface F@0.01 improves **44/50** (mean +0.067, p 3e-8), but final joint error better on only **3/11** with expert GT | `anatomical_pose_init/LAB_RECORD_correspondence_20260827.md` s9; JAB arm I_cse | 50 real (bench50) / 11 real | [V] |
| No diagnostic predicts measurement error | chamfer \|r\| <= 0.34 vs trait error; even oracle correspondence/pose/shape error \|r\| ~0.41 | `morphometric_validation/RESULTS_M2_synthetic_recovery.md` | 48 synthetic, exact solution exists | [R] |

### C3 -- the diagnosis (Task 5 paragraph; Task 4 pose-init rows)

| result | numbers | source | n / set | status |
|---|---|---|---|---|
| Preservation articulation is out of range (fit-independent) | expert-skeleton leg bend vs rest **18-38 deg** (10/11 >= 26 deg) vs synthetic training max **19 deg**; survives 4x annotation noise | `research/review_methods/Q3a.../out/A7*.json` | 11 real (expert joints) vs 4000 synthetic | [V] [NEW] |
| Same at population scale | fitted articulation median **34.8 deg** (p5 20.6) across **657** scans, 161 genera, no JAB genus | `research/review_methods/P0.../RESULTS.md` | 657 real | [V] [NEW]; per-specimen fitted poses are NOT valid (gate failed) -- population level only |
| Articulation, not shape, breaks correspondence | real poses on clean synthetic ants: leg accuracy **0.745**; real shapes: **0.920** (training-like 0.908) | `research/review_methods/Q3b.../` S2-confirm | 136 synthetic, poses from 45 independent genera | [V] [NEW] |
| What breaks is leg identity | real-scan correspondence errors: wrong leg number 10-37%, wrong side 0-5% | `research/review_methods/Q3a.../RESULTS.md` A2 | 11 real (expert-skeleton proxy, gate-validated) | [V] [NEW] |
| Chamfer density asymmetry pulls legs into the gaster | joint study: depth −41% but count +16.8%, 23/27 specimens (coin flip); density asymmetry and proximity-vs-winding-number disagreement confirmed | `penetration/penetration_joint_study/FINDINGS.md` (26 Aug) | 50 real | [V] |
| Pose estimated from the surface does not beat rest pose | learned init at parity on bench50 (G1b-G1d p 0.72-0.86); geometric estimators below zero init | `anatomical_pose_init/out_ceiling_20260820_25pc/RESULTS_bench50_G1_vs_G3.md`, RESULTS_ABC_DEF | 50 real / 12 synthetic | [R] |
| ...but a correct pose does help | ground-truth-like distal pose: leg_acc **0.939** vs proximal-concentrated error 0.768 at matched 23 deg | RESULTS_ABC_DEF (one seed) | 12 synthetic | [V] -> wording: the bottleneck is *estimating* pose from the surface, not pose itself |
| Learned init on real scans | equivalent or worse; catastrophic on 2/11 | JAB arm H | 11 real | [V] |

### C4 -- what would change it (Task 5 closing sentence)

| result | numbers | source | n / set | status |
|---|---|---|---|---|
| Correspondence oracle (Fabian's number) | +0.057 seg_acc, 12/12; ceiling 0.874 vs 0.974 ground-truth mesh | `anatomical_pose_init/RESULTS_correspondence_oracle_20260825.md` | 12 synthetic | [V]; superseded for *predicted* correspondence by the n = 48 replication below |
| Predicted correspondence, properly powered | seg_acc **+0.031** (37/48, all three tests), leg +0.058 | LAB_RECORD s6, s14 | 48 synthetic | [V]; the n = 12 "+0.050 / 84-92% of oracle" numbers are superseded (your 4 Oct claim audit) |
| Size of the lever with correct correspondence | skeleton error **2.43 -> 0.89% of body length, 48/48** | `research/review_methods/Q3b.../RESULTS.md` S1 | 48 synthetic, 3 seeds | [V] [NEW] |
| Conditional on leg identity | value gone by ~20-25% wrong-leg targets; no weight rescues wrong-leg targets; off-leg errors nearly harmless; per-specimen wrong-leg fraction predicts benefit (Spearman 0.31, p 0.005) | Q3b S1, Q3d, Q2x | 48 + 66 synthetic | [V] [NEW] |
| Explains the real-scan bimodality | CSE on real: Formica 44.8 -> 14.5, Eciton 68.6 -> 47.1, but Cephalotes 23.5 -> 69.7% WL | JAB arm I_cse | 11 real | [V] |

This replaces "correspondence is necessary but not sufficient" (from 0.874 vs 0.974) with a sharper,
better-powered statement: correspondence is the largest lever measured, and it pays only when leg
identity is right, which real articulation currently prevents.

### C5 -- what survived (Task 4)

| result | numbers | source | status |
|---|---|---|---|
| Head width extraction | robust on 40/40 real; pose change 0.115%; allometry exponent 1.205 [1.154, 1.257] vs reference 1.2352 | `morphometric_validation/`, M3b | [R] |
| Genus signal in shape ratios; cross-corpus replication | Fabian's numbers (lift 3.8x, p 0.0125, gaster slenderness R 0.79, cephalic index 0.62) | `morphometrics/REPORT_MORPHOMETRICS.md`; figure code uses rigid part extents (not the defective landmark file, checked) | [R] |
| Scale cap | removes impossible anterior ratios; joint-neutral on JAB (EQUIVALENT); does not fix head-carried anatomy (X1) | khaoula_review TASK5, JAB, X1 | [R] -- word as "fixes the statistic", not "fixes the anatomy" |
| Trap-jaw separation | Fabian's claim | `morphometrics/REPORT_MORPHOMETRICS.md`, `FINAL_REPORT.md` | [R] |

## What must change in Fabian's text (tell him)

1. Task 5 "a pose initialisation cannot beat rest pose because leg configurations are
   underdetermined": a *correct* pose does beat rest pose (0.939 vs 0.768); what fails is estimating it
   from the surface. Suggested: "pose initialisations estimated from the surface do not beat rest pose".
2. Task 5 "necessary but not sufficient" from 0.874 vs 0.974: the 0.974 is the metric's own ceiling and
   includes a 0.09 coxa floor from adjacent-coxa ambiguity (C14p); the better-powered statement is C4.
3. Any n = 12 correspondence number (+0.050, 84%/92% of oracle) must not be used; the n = 48 replication
   (+0.031) is the one to carry.
4. Scale cap belongs under "what survived" only as a statistic-level fix (X1, JAB EQUIVALENT).

## What we leave out (does not serve the message, or is not solid enough)

Everything voided by the 16 Sep contamination audit; the Q3c/Q3d mechanism details beyond one sentence
(follow-up paper); classical baselines (unfinished); any per-specimen use of P0 fitted poses.
