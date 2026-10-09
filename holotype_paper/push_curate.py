"""Curate the clean push: new/changed files vs origin/feature/investigation -> what Fabian sees.

Keeps: source and job files, write-ups, configs, LaTeX, small tables (<= 1 MB), and figures that a
kept write-up references by file name (or that sit in a `figures/` folder).
Drops (listed in the manifest with size + sha256, nothing deleted locally): run logs (.log/.out/.err),
anything under logs/ sbatch_logs/ specimens/, run-dump folders, the void slide8 outputs (its .py/.md
stay as audit provenance), and the three files Fabian asked to remove.
Writes holotype_paper/push_lists/{final_add.txt, final_delete.txt, EXCLUDED_FROM_GIT.tsv}.
"""
import hashlib
import os
import re

REPO = "/rwthfs/rz/cluster/home/nao48500/SMILify"
OUT = os.path.join(REPO, "holotype_paper", "push_lists")
CODE_DOC = {".py", ".sh", ".sbatch", ".slurm", ".md", ".yaml", ".yml", ".toml", ".cfg", ".tex", ".bib", ".sty",
            ".ipynb", ".html", ".ps1", ".bat", ".gitignore", ".gitattributes", ".rst", ".pkl", "",
            ".cpp", ".c", ".h", ".hpp", ".cs", ".js", ".patch", ".freeze", ".jsonl"}
TABLE = {".json", ".csv", ".tsv", ".txt"}
FIG = {".png", ".svg", ".pdf", ".jpg", ".jpeg", ".gif"}
DROP_DIRS = {"logs", "sbatch_logs", "specimens", "issue95_fullrun", "antscan_logs", "build", "CMakeFiles",
             "third_party"}
DROP_FILES = {"tatus", "sync_antscan_drive.sh", "sync_antscan_drive_enhanced.sh"}


def rd(p):
    return [l for l in open(p).read().splitlines() if l]


def main():
    os.chdir(REPO)
    new, changed, gone = rd("/w0/tmp/nao48500/delta_new.txt"), rd("/w0/tmp/nao48500/delta_changed.txt"), rd("/w0/tmp/nao48500/delta_gone.txt")
    cand = sorted(set(new) | set(changed))
    keep, drop = [], []
    for p in cand:
        parts = p.split("/")
        ext = os.path.splitext(p)[1].lower()
        size = os.path.getsize(p)
        if p in DROP_FILES:
            drop.append((p, "Fabian: remove from the investigation branch"))
        elif set(parts[:-1]) & DROP_DIRS:
            drop.append((p, "run logs / per-specimen run output"))
        elif ext in {".log", ".out", ".err"}:
            drop.append((p, "run log"))
        elif "slide8_score_can_lie" in parts and ext not in {".py", ".md"}:
            drop.append((p, "output of a series voided by the 2026-09-16 audit"))
        elif ext in CODE_DOC:
            keep.append(p)
        elif ext in TABLE:
            (keep if size <= 1_000_000 else drop).append(p if size <= 1_000_000 else (p, "table > 1 MB"))
        elif ext in FIG:
            keep.append(p)
        else:
            drop.append((p, f"type {ext or '(none)'}"))
    docs = [p for p in keep if os.path.splitext(p)[1].lower() in {".md", ".tex", ".html", ".py", ".ipynb"}]
    refs = set()
    for d in docs:
        try:
            refs.update(os.path.basename(m) for m in re.findall(r"[\w./-]+\.(?:png|svg|pdf|jpg|jpeg|gif)", open(d, errors="ignore").read()))
        except OSError:
            pass
    final = []
    for p in keep:
        if os.path.splitext(p)[1].lower() in FIG and os.path.basename(p) not in refs and "figures" not in p.split("/"):
            drop.append((p, "figure not referenced by any kept write-up"))
        else:
            final.append(p)
    deletes = sorted(set(gone) | {f for f in DROP_FILES if os.path.exists(f)})
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "final_add.txt"), "w").write("\n".join(final) + "\n")
    open(os.path.join(OUT, "final_delete.txt"), "w").write("\n".join(deletes) + "\n")
    with open(os.path.join(OUT, "EXCLUDED_FROM_GIT.tsv"), "w") as f:
        f.write("path\tbytes\tsha256\treason\n")
        for p, why in sorted(drop):
            f.write(f"{p}\t{os.path.getsize(p)}\t{hashlib.sha256(open(p, 'rb').read()).hexdigest()}\t{why}\n")
        for line in rd(os.path.join(OUT, "exclude.tsv"))[1:]:
            p, b, why, _, h = line.split("\t")
            f.write(f"{p}\t{b}\t{h}\t{why}\n")
    mb = sum(os.path.getsize(p) for p in final) / 1e6
    from collections import Counter
    print(f"commit: {len(final)} files ({mb:.1f} MB); delete: {len(deletes)}; dropped from delta: {len(drop)}")
    print("commit by ext:", Counter(os.path.splitext(p)[1].lower() or '(none)' for p in final).most_common(12))
    print("largest committed:", sorted(((os.path.getsize(p) / 1e6, p) for p in final), reverse=True)[:6])


if __name__ == "__main__":
    main()
