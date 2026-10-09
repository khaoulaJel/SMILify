# Task 1: what you run (commits and pushes are yours)

Everything is prepared and staged in two separate folders. Your own working tree
(`/rwthfs/rz/cluster/home/nao48500/SMILify`) is untouched.

## 0. Once: set your identity in each folder

The cluster has no git identity configured (commits fall back to `nao48500@login23-1...`).
Your earlier fix commit used `Khaoula <khaoula.jellal@um6p.ma>`.

```bash
for d in /rwthfs/rz/cluster/home/nao48500/SMILify_task1_weld_PR /rwthfs/rz/cluster/home/nao48500/SMILify_task1_investigation; do
  git -C $d config user.name "Khaoula"
  git -C $d config user.email "khaoula.jellal@um6p.ma"
done
```

## 1. Weld fix pull request (branch `fix/apply-modifiers-weld`)

Staged: one file, `custom_processing/prepare_antscan_data_for_mesh_fitting.py` = the validated version
(same program as your 30 Jul commit, ruff-formatted for CI, comments describe the mechanism). It merges
cleanly with today's master. Do **not** use `git commit -a` (a line-ending artefact in
`utilities/convert_smal_windows.ps1` must stay out).

```bash
cd /rwthfs/rz/cluster/home/nao48500/SMILify_task1_weld_PR
git commit -m "style(antscan): ruff format and document the preprocessing pipeline"
git push origin fix/apply-modifiers-weld
```

Then on GitHub: open a pull request **FabianPlum/SMILify `master` <- khaoulaJel/SMILify
`fix/apply-modifiers-weld`**, title and description from `holotype_paper/task1_weld_fix/PR_DRAFT.md`
(issue text in the same file, open it first and reference it).

## 2. Investigation branch (`feature/investigation` on your fork)

Staged on a fresh branch `investigation-sync` built from the pushed `feature/investigation` (which already
contains today's master): Fabian's three files removed, then your current code, write-ups, configs, small
result tables and referenced figures (1,689 files, largest 2.4 MB). Excluded (still on the cluster,
nothing deleted): run logs, meshes, fit arrays, scratch/tmp outputs, slide decks, presentation/ -- every
excluded file is listed with size and sha256 in `holotype_paper/push_lists/EXCLUDED_FROM_GIT.tsv`.
Tests pass (105 passed); simulated merge into master + the weld PR has no conflicts.

Two commits, so the removal Fabian asked for comes first:

```bash
cd /rwthfs/rz/cluster/home/nao48500/SMILify_task1_investigation
git commit -m "chore: remove stray tatus file and Drive sync scripts" -- tatus sync_antscan_drive.sh sync_antscan_drive_enhanced.sh
git commit -m "investigation: current state of code, write-ups and results (through 2026-10-09)"
git push origin investigation-sync:feature/investigation
```

This is a normal (fast-forward) push, not a force push.

## 3. Other branches

```bash
cd /rwthfs/rz/cluster/home/nao48500/SMILify
git push origin master          # identical to Fabian's master; syncs your fork
git push origin to-ship         # 17 commits, no large files
```

Do **not** push: `second-to-ship` (WIP snapshot with 90 files > 50 MB; content is in the investigation
push), the local `feature/investigation` (134 commits containing files up to 847 MB), `worktree-agent-*`
(empty helper branches). `feature/registration_moonshot` is already on Fabian's repo unchanged.

## 4. Known issues to mention to Fabian

- `custom_processing/prepare_antscan_data_for_mesh_fitting_test.py` (experimental variant, investigation
  only) calls `reduce_vertices_by_distance` without defining it (ruff F821).
- Research folders (`diagnostics/`, `research/`, `holotype_paper/`) do not pass the repo's ruff config
  (about 1,270 findings, mostly one-line `a; b` statements). Irrelevant for pushing; before any merge of
  the investigation branch into master they need either a lint exclusion or a cleanup -- his call.

## 5. Afterwards (optional)

```bash
cd /rwthfs/rz/cluster/home/nao48500/SMILify
git worktree remove /rwthfs/rz/cluster/home/nao48500/SMILify_task1_weld_PR
git worktree remove /rwthfs/rz/cluster/home/nao48500/SMILify_task1_investigation
```
