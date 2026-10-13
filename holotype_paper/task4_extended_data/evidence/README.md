# Evidence for the Extended Data page

Tables behind `../README.md`, by section of the page. Recipe, model, command and the tables shared with the recipe
page are in `../../task3_registration_recipe/`; its `evidence/README.md` defines the common measures (joint-error
shift and intervals, F@0.01, surface distance, shape-prior region, intersecting triangle pairs, proximity count,
intersection-curve length, mandible asymmetry, genus accuracy).

| Section | File | Content |
|---|---|---|
| 1 | `mesh_properties.csv` | Connected components, area outside the largest component, share of open edges and face count of the scan meshes, per corpus (medians). |
| 2 | `joint_error_by_region.csv` | Default recipe on the 11 expert-annotated scans, seeds 0–2: median joint error per body region and over all joints, range over scans, seed-to-seed SD. |
| 3 | `strategies_joint_error.csv` | Each registration strategy against the default fitted in the same batch: joint-error shift, 95% interval, scans better and worse, chamfer ratio. |
| 3 | `pose_initialisation_50_scans.csv` | Learned pose initialisation against zero initialisation on 50 AntScan scans (one seed): chamfer and F-scores, paired tests. |
| 4 | `penetration_bars_per_seed.csv` | Attempts T1, T2, T4, T5 against their references in the measures and bars fixed before the runs, per seed. |
| 4 | `penetration_T6_winding_number_loss.csv` | The winding-number loss on three specimens beside the fits without a loss and with the proximity loss. |
| 4 | `penetration_T7_symmetric_sampling.csv` | Symmetric chamfer sampling against its control, 10 scans, per seed. |
| 4 | `proximity_against_winding_number.csv` | Agreement between the proximity test and the winding-number test on 50 scans. |
| 5 | `supervised_fit_capacity.csv` | Fits supervised with the expert joints of the 11 scans (seed 0), beside the unsupervised fit and a control. |
| 5 | `supervision_transfer_by_region.csv` | Joint error of each body region when unsupervised, when every other region is supervised, and when it is supervised itself. |
| 5 | `oracle_budget.csv` | Synthetic scans with real-level articulation: share of the gap to a fully informed fit closed by restoring one quantity at a time. |
| 5 | `synthetic_correspondence_target_quality.csv` | Synthetic scans: joint error with correspondence targets of which a given share points to the wrong leg. |
| 5 | `learned_correspondence_under_shift.csv` | Leg-identity accuracy of the learned correspondence under controlled shifts of shape, pose and scale. |
| 5 | `leg_articulation.csv` | Median leg bend on the expert-annotated scans and in the synthetic training set. |
| 5 | `posture_measures.csv` | Two scan-level posture measures: correlation with the expert-measured leg bend on the 11 annotated scans, medians in the two corpora, and the number of AntScan workers inside the cleaned range on all five mesh and posture measures. |
| 6 | `genus_accuracy_default_runs.csv` | Collection-lot-blind genus accuracy of two independent runs of the default on the 757 workers. |
| 6 | `species_against_collection_lot.csv` | Within-genus species classification from the fifteen indices and from the accession number alone. |
| 6 | `scale_cap.csv` | Genus accuracy and anterior scale ratios with and without the scale cap, 838 specimens. |
| 6 | `cross_corpus_genus_retrieval.csv` | Cleaned specimens assigned to a genus by their nearest AntScan genus mean, for the three seeds of the cleaned fits. |
| 6 | `cross_corpus_index_agreement.csv` | Agreement of genus means between AntScan workers and cleaned scans for fifteen indices: Pearson R with bootstrap interval, permutation p, Holm correction; three seeds of the cleaned fits. |
| 6 | `mandible_slenderness_by_side.csv` | Mandible slenderness of the trap-jaw genera measured on both sides, on the right and on the left mandible. |
| 6 | `head_width_gate.csv` | Number of workers whose head width passes a landmark-free admissibility check. |

Joint error is in percent of Weber's length (WL) unless a column says body length. Seeds are those named in the file
or on the page; where neither names one, seed 0.
