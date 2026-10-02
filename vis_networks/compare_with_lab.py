import csv
import difflib
import hashlib
import os
import subprocess
import sys
from collections import Counter

CODE_EXT = {".py", ".ps1", ".sh", ".bat", ".cfg", ".toml", ".yaml", ".yml", ".ipynb"}
SKIP_DIRS = {".git", "__pycache__", "runs", ".venv", ".conda", ".vscode"}


def code_root(top):
    """The shallowest folder holding main.py, so both trees are compared from the same level."""
    found = []
    for dirpath, dirnames, filenames in os.walk(top):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        if "main.py" in filenames:
            found.append(dirpath)
    if len(found) > 1:
        print(f"note: main.py found in {found}; using the shallowest")
    return min(found, key=lambda d: d.count(os.sep)) if found else top


def code_files(root):
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in CODE_EXT:
                full = os.path.join(dirpath, fn)
                out[os.path.relpath(full, root).replace("\\", "/")] = full
    return out


def lines(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return [ln.rstrip("\r\n") for ln in f]          # CRLF and LF count as the same


def digest(path):
    return hashlib.sha1("\n".join(lines(path)).encode("utf-8")).hexdigest()


def diff_counts(a, b):
    d = list(difflib.unified_diff(a, b, lineterm="", n=0))
    added = sum(1 for x in d if x.startswith("+") and not x.startswith("+++"))
    removed = sum(1 for x in d if x.startswith("-") and not x.startswith("---"))
    return added, removed


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: python3 compare_with_lab.py <lab folder> <your repo folder>")
    lab_root, my_root = code_root(sys.argv[1]), code_root(sys.argv[2])
    print(f"lab code root:  {lab_root}\nyour code root: {my_root}")
    for cmd in (["git", "-C", sys.argv[1], "log", "-1", "--format=lab branch head: %h, %ad", "--date=short"],
                ["git", "-C", sys.argv[1], "shortlog", "-sn", "HEAD"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
            if out:
                print(out if cmd[3] == "log" else "commits per author on the lab branch:\n" + out)
        except OSError:
            pass

    lab, mine = code_files(lab_root), code_files(my_root)
    by_hash, by_name = {}, {}
    for rel, p in lab.items():
        by_hash.setdefault(digest(p), []).append(rel)
        by_name.setdefault(os.path.basename(rel), []).append(rel)

    rows = []
    for rel in sorted(set(lab) | set(mine)):
        if rel in lab and rel in mine:
            a, b = lines(lab[rel]), lines(mine[rel])
            if a == b:
                rows.append([rel, "unchanged", len(a), len(b), 0, 0, ""])
            else:
                rows.append([rel, "modified", len(a), len(b), *diff_counts(a, b), ""])
        elif rel in mine:
            b = lines(mine[rel])
            same = by_hash.get(digest(mine[rel]), [])
            twins = [t for t in by_name.get(os.path.basename(rel), []) if t not in mine]
            if same:
                rows.append([rel, "moved, unchanged", "", len(b), 0, 0, ";".join(same)])
            elif twins:
                a = lines(lab[twins[0]])
                rows.append([rel, "moved, modified", len(a), len(b), *diff_counts(a, b), twins[0]])
            else:
                rows.append([rel, "new in yours", "", len(b), len(b), 0, ""])
        else:
            rows.append([rel, "only in lab", len(lines(lab[rel])), "", "", "", ""])

    with open("code_provenance.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "status", "lab_lines", "your_lines", "lines_added", "lines_removed", "lab_path"])
        w.writerows(rows)

    counts = Counter(r[1] for r in rows)
    print("\nfiles: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    for r in rows:
        if r[1] in ("modified", "moved, modified"):
            print(f"  {r[1]:16s} {r[0]}: +{r[4]} / -{r[5]} lines ({r[2]} -> {r[3]})")
    for r in rows:
        if r[1] == "new in yours":
            print(f"  {'new':16s} {r[0]} ({r[3]} lines)")
    print("\nSaved code_provenance.csv")


if __name__ == "__main__":
    main()
