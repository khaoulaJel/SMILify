"""Classify every file in the current feature/investigation working tree for the clean push branch.

Rules (Fabian's work package: push the current state incl. the write-ups, so the paper starts from it;
user: lose nothing):
  INCLUDE  code / config / docs / small results / figures, <= SIZE_CAP each
  EXCLUDE  meshes, arrays, checkpoints, scratch dirs, slide decks, presentation/, core dumps,
           anything > SIZE_CAP, and the three files Fabian asked to drop
Writes holotype_paper/push_lists/{include,exclude}.txt (exclude with size + sha256) and prints totals.
"""
import hashlib
import os
import subprocess

REPO = "/rwthfs/rz/cluster/home/nao48500/SMILify"
OUT = os.path.join(REPO, "holotype_paper", "push_lists")
SIZE_CAP = 5_000_000
TEXT_EXT = {".py", ".sh", ".md", ".txt", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".json", ".csv", ".tsv",
            ".tex", ".bib", ".sty", ".sbatch", ".slurm", ".ipynb", ".html", ".ps1", ".bat", ".gitignore",
            ".gitattributes", ".log", ".rst", ".out", ".err", ".manifest", ".cpp", ".c", ".h", ".hpp", ".cs",
            ".js", ".patch", ".freeze", ".jsonl"}
FIG_EXT = {".png", ".svg", ".pdf", ".jpg", ".jpeg", ".gif"}
DATA_EXT = {".obj", ".off", ".stl", ".ply", ".glb", ".fbx", ".npz", ".npy", ".pt", ".pth", ".ckpt", ".h5",
            ".blend", ".blend1", ".pptx", ".zip", ".tar", ".gz", ".mp4", ".mov", ".tif", ".tiff", ".bin"}
EXCL_DIR_PARTS = {"tmp", "sweep_work", "__pycache__", "presentation", "runs", ".pytest_cache", "test_output"}
DROP = {"tatus", "sync_antscan_drive.sh", "sync_antscan_drive_enhanced.sh"}


def files_in_tree():
    tracked = subprocess.run(["git", "ls-files", "-z"], cwd=REPO, capture_output=True).stdout.split(b"\0")
    untracked = subprocess.run(["git", "ls-files", "-z", "--others", "--exclude-standard"], cwd=REPO,
                               capture_output=True).stdout.split(b"\0")
    ignored = subprocess.run(["git", "ls-files", "-z", "--others", "--ignored", "--exclude-standard"], cwd=REPO,
                             capture_output=True).stdout.split(b"\0")
    out = {}
    for kind, lst in (("tracked", tracked), ("untracked", untracked), ("ignored", ignored)):
        for p in lst:
            if p:
                out.setdefault(p.decode(), kind)
    return out


def classify(path, size):
    parts = set(path.split("/")[:-1])
    name = os.path.basename(path)
    ext = os.path.splitext(name)[1].lower() if "." in name else ""
    if path in DROP:
        return "exclude", "Fabian: drop from investigation branch"
    if name.startswith("core.") or name.endswith((".pyc", ".pyo")):
        return "exclude", "core dump / bytecode"
    if parts & EXCL_DIR_PARTS:
        return "exclude", f"scratch/output dir ({', '.join(sorted(parts & EXCL_DIR_PARTS))})"
    if ext in DATA_EXT:
        return "exclude", f"data file ({ext})"
    if size > SIZE_CAP:
        return "exclude", f"> {SIZE_CAP // 1_000_000} MB"
    if ext in TEXT_EXT or ext in FIG_EXT or ext == "" or ext == ".pkl":
        if ext == ".pkl" and size > 50_000_000:
            return "exclude", "model > 50 MB"
        return "include", ""
    return "exclude", f"unlisted type ({ext})"


def main():
    os.makedirs(OUT, exist_ok=True)
    inc, exc = [], []
    tot = {"include": 0, "exclude": 0}
    for path, kind in sorted(files_in_tree().items()):
        full = os.path.join(REPO, path)
        if not os.path.isfile(full) or path.startswith(("holotype_paper/push_lists/",)):
            continue
        size = os.path.getsize(full)
        cls, why = classify(path, size)
        tot[cls] += size
        if cls == "include":
            inc.append(path)
        else:
            exc.append((path, size, why, kind))
    with open(os.path.join(OUT, "include.txt"), "w") as f:
        f.write("\n".join(inc) + "\n")
    with open(os.path.join(OUT, "exclude.tsv"), "w") as f:
        f.write("path\tbytes\treason\tgit_state\tsha256\n")
        for path, size, why, kind in exc:
            h = hashlib.sha256(open(os.path.join(REPO, path), "rb").read()).hexdigest() if size < 2e9 else "too_large"
            f.write(f"{path}\t{size}\t{why}\t{kind}\t{h}\n")
    print(f"include: {len(inc)} files, {tot['include'] / 1e6:.1f} MB")
    print(f"exclude: {len(exc)} files, {tot['exclude'] / 1e9:.2f} GB")
    from collections import Counter
    print("exclude reasons:", Counter(w.split(" (")[0] for _, _, w, _ in exc).most_common(8))


if __name__ == "__main__":
    main()
