"""Stage the curated investigation state onto the investigation-sync worktree (master already merged in).

- changed tracked files: local diff (origin/feature/investigation -> working tree) applied with
  `git apply --3way`, so changes master made to the same files are kept
- new files: copied; if the merged tree already has a different file at that path, it is reported and
  not overwritten
- deletions (files deleted locally since origin) and Fabian's three files: `git rm`
- prepare_antscan_data_for_mesh_fitting.py: the verified clean version (identical to the weld PR), so
  merging this branch after the PR cannot conflict on it
Nothing is committed. Log: holotype_paper/push_lists/build_log.txt
"""
import os
import shutil
import subprocess

SRC = "/rwthfs/rz/cluster/home/nao48500/SMILify"
DST = "/rwthfs/rz/cluster/home/nao48500/SMILify_task1_investigation"
L = os.path.join(SRC, "holotype_paper", "push_lists")
WELD = "custom_processing/prepare_antscan_data_for_mesh_fitting.py"
CLEAN = os.path.join(SRC, "holotype_paper/task1_weld_fix/versions/clean_master.py")


def git(args, cwd, inp=None):
    return subprocess.run(["git"] + args, cwd=cwd, input=inp, capture_output=True)


def main():
    add = [l for l in open(os.path.join(L, "final_add.txt")).read().splitlines() if l]
    dele = [l for l in open(os.path.join(L, "final_delete.txt")).read().splitlines() if l]
    dele += [l for l in open(os.path.join(L, "deleted_since_remote.txt")).read().splitlines() if l]
    changed = set(l for l in open(os.path.join(L, "changed_since_remote.txt")).read().splitlines() if l)
    log, problems = [], []
    for p in add:
        if p == WELD:
            continue
        if p in changed:
            patch = git(["diff", "--binary", "origin/feature/investigation", "--", p], SRC).stdout
            r = git(["apply", "--3way", "--index", "-"], DST, inp=patch)
            ok = r.returncode == 0
            log.append(f"PATCH3WAY {'ok' if ok else 'CONFLICT'} {p}")
            if not ok:
                problems.append((p, r.stderr.decode()[-300:]))
            continue
        dst = os.path.join(DST, p)
        if os.path.exists(dst):
            same = open(dst, "rb").read() == open(os.path.join(SRC, p), "rb").read()
            if not same:
                problems.append((p, "exists in merged tree with different content; not overwritten"))
                log.append(f"SKIP-EXISTS-DIFF {p}")
                continue
        os.makedirs(os.path.dirname(dst) or DST, exist_ok=True)
        shutil.copy2(os.path.join(SRC, p), dst)
        git(["add", "-f", "--", p], DST)
        log.append(f"ADD {p}")
    for p in dele:
        r = git(["rm", "-q", "--ignore-unmatch", "--cached", "--", p], DST)
        full = os.path.join(DST, p)
        if os.path.islink(full) or os.path.isfile(full):
            os.remove(full)
        log.append(f"RM {p}")
    untracked = git(["ls-files", "--others", "--exclude-standard", "-z"], DST).stdout.decode().split("\0")
    for p in filter(None, untracked):
        full = os.path.join(DST, p)
        if os.path.islink(full) and not os.path.exists(full):
            os.remove(full)
            log.append(f"RM-DANGLING-LINK {p}")
    shutil.copy2(CLEAN, os.path.join(DST, WELD))
    git(["add", "--", WELD], DST)
    log.append(f"SET {WELD} = verified clean version (same as weld PR)")
    open(os.path.join(L, "build_log.txt"), "w").write("\n".join(log) + "\n")
    print(f"added/copied {sum(l.startswith('ADD') for l in log)}, patched {sum(l.startswith('PATCH3WAY ok') for l in log)}, "
          f"removed {sum(l.startswith('RM') for l in log)}; problems: {len(problems)}")
    for p, why in problems:
        print("  PROBLEM", p, "--", why)


if __name__ == "__main__":
    main()
