# Holotype paper: Khaoula's work package (the document we stay focused on)

Source: Fabian's work package, dated 1 Oct 2026, re-dated 7 Oct (verbatim copy at the bottom).
This file is the single place that says what is due, what "done" means, which numbers still need
verifying, and the rules every number must satisfy. Update the tracker and the claim register as
work progresses; never delete a row, mark it.

## Where this goes

The Holotype paper (Nature Methods / Machine Intelligence). Drafts to co-authors **late October**.
The AntScan registration work enters as **one Extended Data figure + one Discussion paragraph**.
The full treatment is a possible follow-up paper. The frame Fabian set:

> A **negative result with a diagnosis**. None of the tested registration strategies fits the
> uncleaned AntScan meshes automatically at the quality the curated ant data reaches, and we can say
> precisely why. Lead with the failure, then show what survived it (scale cap, genus signal in the
> shape ratios, trap-jaw separation).

Two caveats travel with **every** number: genus is the honest taxonomic level (species = collection
lot: accession predicts species at 95.7%, shape at 31.9%), and every classification lift is lot-blind
or says it is not.

**Our message inside Fabian's frame, and the evidence that carries it:**
`PAPER_MESSAGE_AND_EVIDENCE.md` (five claims C1-C5, each with sources, n and verification status;
includes what must change in Fabian's text). Approve it before Task 3-5 writing starts.

## Deadlines and status

| # | task | due | deliverable location | status |
|---|---|---|---|---|
| 1 | Merge the weld fix; drop `tatus` + 2 Drive sync scripts from the investigation branch; push every branch incl. the write-up | **Fri 9 Oct** | PR on `fix/apply-modifiers-weld`; all branches on the remote | not started |
| 2 | Registration stage figure, one cleaned Atta worker, D1_PROD, 4 panels (init, pose, Stage_2, Stage_3) + per-stage chamfer, F@0.01, penetration count | **Mon 12 Oct** | `diagnostics/fig_registration_stages/` (300-dpi PNG + CSV) | not started |
| 3 | Registration recipe page: model file + md5, stages + iterations, every loss weight per stage, what is off/rejected and why (with evidence file), Atta-80 differences, candidate list, exact command | **Wed 14 Oct** (push as it grows) | markdown page | not started |
| 4 | Extended Data summary: corpus; T1-T9 penetration table; scale-cap A/B; GWN finding; pose initialisation; morphometrics replication; regenerate fig_genus_indices + fig_crosscorpus at 300 dpi | **Mon 19 Oct** | markdown page + 2 figures | not started |
| 5 | Discussion paragraph, 200 words: why AntScan meshes resist automatic registration, and what would change it | **Mon 26 Oct** | same page as task 4 | not started |

Fabian reviews **Monday mornings** and returns numbered comments.

## Rules (Fabian's, plus ours)

1. **No new major runs.** Re-rendering or re-scoring existing outputs is fine. New experiments wait
   until the draft is with co-authors. The `research/review_methods/` programme is **paused**
   (its queued jobs were cancelled 2026-10-08).
2. **Every number names its config, its specimen set and its seed.** No number without provenance.
3. **If a diagnostic contradicts the task-4 table, the table loses, and Fabian is told.**
4. **The penetration loss stays optional and off by default.** The correspondence network (PHASE10)
   is the follow-up paper; do not start it.
5. **One step at a time.** For each task: locate the source of every claim on disk, verify it by
   re-scoring or re-reading the raw output (not a report), record the verification, then write.
   A report is not evidence; the repo output is (standing rule, memory `feedback_repo_not_report_authority`).
6. **Contamination audit (2026-09-16) applies.** Nothing from G-series, R1-R9, A1/A3, V1/V6-V13 or
   the slide-8 rebuild enters the paper. Anything that used `landmark_indices_recalibrated.json` is
   suspect until checked.
7. **Outward actions need the user's go-ahead**: opening the PR, pushing branches, sending anything to
   Fabian. Prepare everything, then ask.

## Claim register (every number Fabian's text quotes, and its verification status)

Status: `UNVERIFIED` (taken from the package), `FOUND` (source located on disk, not yet re-derived),
`VERIFIED` (re-derived from raw output, provenance recorded), `CONFLICT` (disk disagrees -> tell Fabian).

| task | claim in the package | status | note |
|---|---|---|---|
| 4 | Corpus: 838 specimens = 757 workers + 81 clean, 188 genera | UNVERIFIED | |
| 4 | T1-T9 penetration: gaster-legs change, all-pairs change, mean and worst F delta, seeds, pre-registered bar | UNVERIFIED | sources: `diagnostics/khaoula_review/TASK*`, `PENETRATION_LOSS_SUMMARY.md` |
| 4 | T8 (BVH) and T9 (post-hoc repair) are tooling failures from CUDA faults, not negatives | UNVERIFIED | needs the actual error logs |
| 4 | T7 split: F-score up in every seed, count flat | UNVERIFIED | |
| 4 | Scale-cap A/B: lift 5.4x vs 5.3x; anterior max ratios 8.9->3.8, 10.6->4.1, 13.6->4.1 | UNVERIFIED | memory X1: the scale cap fixes the statistic, not the head-carried anatomy -- wording must not overclaim |
| 4 | GWN disagreement: r 0.947, 44 of 50 worsen gaster-to-legs | UNVERIFIED | |
| 4 | Learned init at parity on bench50 (G1b-G1d, p 0.72-0.86) | UNVERIFIED | |
| 4 | Three coherent geometric estimators below zero init: leg_acc 0.80-0.83 vs 0.87, 12 synthetic specimens | UNVERIFIED | 12 synthetic specimens must be stated |
| 4 | Morphometrics replication: lift 3.8x, p 0.0125, gaster slenderness R 0.79, cephalic index 0.62, leg ratios not replicating | UNVERIFIED | figure code (`diagnostics/morphometrics/genus_table.py`) uses `measure.part_dims` rigid part extents, **no landmark indices** -> not affected by the recalibration defect (checked 2026-10-08). Still need the run/config these numbers came from |
| 4 | fig_genus_indices, fig_crosscorpus regenerated "from the D1_PROD output you already have" | UNVERIFIED | must confirm the existing fits ARE D1_PROD (memory M3b: the 757 fits may not be on cluster) |
| 5 | Correspondence oracle: +0.057 seg_acc, 12 of 12 | FOUND | `diagnostics/anatomical_pose_init/RESULTS_correspondence_oracle_20260825.md:52` (synth_clean, n = 12). A later doc reports +0.0544 against a different baseline (C4); pin which comparison is cited |
| 5 | Perfect correspondence 0.874 vs GT mesh 0.974 | FOUND | same file, line 33 (`GT_as_fitted_ceiling` 0.974) |
| 5 | Pose init cannot beat rest pose because leg configurations are underdetermined from the surface | UNVERIFIED | check against C14p (GT pose init does help distal segments on synthetic) before using "cannot" |
| 3 | Recipe stages/weights; Atta-80 recipe identical or not | UNVERIFIED | `diagnostics/SHIPPED_RECIPE.md` (13 Aug) is the starting point; the JAB A_prod config is the frozen D1_PROD |

### Things on disk Fabian's text does not mention (raise with him, do not add silently)

- **JAB** (expert-joint benchmark, n = 11, frozen protocol): D1_PROD median joint error 25.1% of
  Weber's length; no single-factor change improves it; surface fit and skeleton dissociate. This is the
  most rigorous real-scan accuracy number we have and directly supports "none of the strategies fits
  at curated quality".
- The post-internship diagnosis (`research/review_methods/`): real articulation is outside the
  synthetic training range; correct correspondence cuts synthetic skeleton error by 63%; wrong-leg
  correspondence removes the benefit. These are new runs after Fabian's package; whether any of it
  enters the Discussion is **Fabian's call**.

## Working method per task

1. List every number and claim the task needs.
2. Find its source output (not a report). Record path, config, specimen set, seed.
3. Re-derive it from the raw output where cheap; otherwise re-read the raw file. Note agreement.
4. Only then write the sentence or table cell, with its provenance.
5. Record conflicts in the register and in a note to Fabian.

---

## Verbatim work package (1 Oct 2026, re-dated 7 Oct)

Work package Khaoula

The Holotype paper goes to Nature Methods / Machine Intelligence later this year with paper drafts
going to collaborators and co-authors in late October. Your AntScan registration work enters it as an
Extended Data figure and a Discussion paragraph, and the full treatment becomes a potential follow-up
paper. Everything below is due by Mon 26 Oct. The Atta registration figure and the recipe are needed
much earlier, because one sits in a main figure and the other feeds Methods.

**How this enters the paper.** The main paper builds its ant shape space from 60 artist-cleaned scans
plus 20 additional Atta vollenweideri scans from the scAnt pipeline, originally produced for the WOLO
study (https://github.com/FabianPlum/WOLO), where registration works well. Your work answers the
question a reviewer will ask next. What happens when you point the same pipeline at the
2,193-specimen AntScan corpus that was published without any of that cleaning? The honest answer is
that none of the registration strategies you tested fits those meshes automatically to a deformable
model at the quality the curated ant data reaches, and you have the numbers to say precisely why.
That is a result. It defines the problem the follow-up paper solves, and it keeps the main paper from
overclaiming.

So the frame is a negative result with a diagnosis. The things that did work (the scale cap, the
genus signal in the shape ratios, the trap-jaw separation) are the evidence that the registrations
carry biology even where they are imperfect. Lead with the failure, then show what survived it.

Two caveats stay attached to every number. Genus is the honest taxonomic level, because species
equals collection lot (accession predicts species at 95.7%, shape at 31.9%). And every
classification lift is lot-blind, or it says so.

**Tasks**

1. *Merge the weld fix.* Open the PR for the apply_modifiers and ray-cast robustness changes on
   fix/apply-modifiers-weld. Also drop the stray tatus file and the two Drive sync scripts from the
   investigation branch before anything else from it is merged. Then push the current state of every
   branch, including the write-up you are working on about why each technique works or fails. Your
   last push is from 26 Aug, and the paper should start from what you have now. — Done when: CI green,
   prepare_antscan_data_for_mesh_fitting.py on master has the adaptive weld distance, the 20%
   face-loss abort and the island handling, and every branch pushed. — Due Fri 9 Oct.
2. *Registration stage figure on one Atta worker.* Run D1_PROD on one of the 80 cleaned Atta scans and
   render the template at each stage (init, pose, Stage_2, Stage_3) with chamfer, F@0.01 and
   penetration count per stage underneath. Same camera, same lighting, four panels in a row. — Done
   when: a PNG at 300 dpi plus the per-stage CSV, committed under diagnostics/fig_registration_stages.
   — Due Mon 12 Oct.
3. *Registration recipe.* One page that states the default mesh-registration recipe as it stands, so
   I can lift it into Methods and a Supplementary Table. Start from diagnostics/SHIPPED_RECIPE.md
   (13 Aug) and update it with what you learned after it. Name the model file and its md5, the stages
   in order (H0_body, H1_legs, H2_joint, H3 skipped, Stage_2 coarse, Stage_3 fine) with iteration
   counts, and every loss with its weight per stage. For each term that is off or was rejected
   (penetration in all its forms, edge_mode rest, w_geocoh, w_trans, w_beta_prior, dense
   correspondence) give one line on why and the file that holds the evidence. Say whether the 80
   cleaned Atta scans use this recipe unchanged, and if not, what differs. Things that looked good
   but were never tested inside D1_PROD, such as symmetric chamfer sampling in the hierarchical stages
   (F-score up in all three seeds in TASK7), go in a short list of candidates, not into the defaults.
   No new runs for this. — Done when: a markdown page with one table of stages and weights, one table
   of what is off and why, and the exact command. I can paste it into Methods with light editing. —
   Due Wed 14 Oct.
4. *Extended Data summary of the AntScan investigation.* One markdown page, numbers only, no narrative
   of how you got there. Build it from the write-up you are already doing on why each technique works
   or fails, not as a separate document. Corpus (838 specimens, 757 workers plus 81 clean, 188
   genera). All nine penetration attempts, T1 to T9, in one table with gaster-legs change, all-pairs
   change, mean and worst F delta, seed count, and the pre-registered bar they had to pass. Report T8
   (BVH) and T9 (post-hoc repair) as tooling failures from CUDA faults, not as negative results.
   Report T7 as split, F-score up in every seed and count flat. The scale-cap A/B (lift 5.4× vs 5.3×,
   anterior max ratios 8.9 to 3.8, 10.6 to 4.1, 13.6 to 4.1). The GWN disagreement finding (r 0.947,
   44 of 50 worsen gaster to legs). Pose initialisation, with learned init at parity on bench50 (G1b
   to G1d, p 0.72 to 0.86) and the three coherent geometric estimators below zero init (leg_acc 0.80
   to 0.83 against 0.87), noting that these leg_acc numbers come from 12 synthetic specimens. The
   morphometrics replication across corpora (lift 3.8×, p 0.0125, gaster slenderness R 0.79, cephalic
   index 0.62, leg ratios not replicating). The two existing figures, fig_genus_indices and
   fig_crosscorpus, regenerated from the D1_PROD output you already have. — Done when: a page I can
   paste into Extended Data with at most light editing, plus the two figures at 300 dpi. — Due Mon 19 Oct.
5. *The diagnosis paragraph.* Two hundred words, written for the Discussion, on why published AntScan
   meshes resist automatic registration. Resolution and completeness vary, preservation poses are
   idiosyncratic, the chamfer vertex-density asymmetry pulls legs into the gaster, and a pose
   initialisation cannot beat rest pose because leg configurations are underdetermined from the
   surface alone. End with what would change it. Your correspondence-oracle result (+0.057 seg_acc,
   12 of 12) points to dense correspondence rather than better losses. Your capacity check adds the
   limit. Even perfect correspondence reaches seg_acc 0.874 against 0.974 for the ground-truth mesh,
   so correspondence is necessary but not sufficient. — Paragraph in the same markdown page as task 4.
   — Due Mon 26 Oct.

**Reporting, and what to leave alone.** Push the task 3 page as it grows, do not wait for it to be
complete. I review Monday mornings and will send back numbered comments. Every number in the page
names the config, the specimen set and the seed it came from.

No new major runs. Re-rendering or re-scoring outputs you already have is fine. New experiments wait
until the draft is with co-authors, and better until reviews are in. The penetration loss stays
optional and off by default, the correspondence network design (PHASE10) is the follow-up paper and
should not start before this is in. If you find a number in your diagnostics that contradicts
something in the task 4 table, the table loses and you tell me.

The PROTOCOL-style write-ups you did for the penetration arms are the right habit and are why this can
be an Extended Data figure at all. Keep doing that.
