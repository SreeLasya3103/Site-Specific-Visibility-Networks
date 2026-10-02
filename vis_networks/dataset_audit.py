"""
dataset_audit.py - what is in NewWebcams, and does the balanced subset contain
night or low-light images?

1. Census of the whole folder: images, sites, camera views, capture folders,
   date range, reported visibility values and the ten classes they fall in.
2. Checks that every image of the balanced subset (splits/*.csv) is in the folder.
3. Brightness of every subset image, measured on the region the models see (the
   top 12.81% of the frame, mostly sky, is cut off first). Writes the numbers and
   a contact sheet of the darkest images, so night images can be confirmed by eye.

Run from vis_networks/:
    python3 dataset_audit.py
Optional: --lowlight_scope all  measures every image in the folder (slower).
Results go to dataset_audit/.
"""

import argparse
import csv
import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import numpy as np
from PIL import Image

NAME = re.compile(r"^SITE\d+_ORNT\d+_VIS.+\.(png|jpg|jpeg)$", re.IGNORECASE)
SKIP_DIRS = {"references_by_view", "splits_similarity"}
CLASS_GROUPS = [{0.0, 1.0, 1.25, 1.5, 1.75}, {2.0, 2.25, 2.5}] + [{float(k)} for k in range(3, 11)]
TOP_CROP = 0.1281                       # same sky crop as dsets/webcams/common.py


def visibility(name):
    """Reported visibility in miles, parsed exactly as dsets/webcams/common.py does."""
    s = name.split("_")[2].split(".")[0].split("S")[1].split("m")[0].replace("-", ".")
    return 10.0 if s == "10+" else min(float(s), 10.0)


def vis_class(v):
    for i, g in enumerate(CLASS_GROUPS):
        if v in g:
            return i + 1
    return None


def folder_time(folder):
    for fmt in ("%Y-%m-%d-%H-%M-%S", "%m-%d-%y-%I-%M-%p"):
        try:
            return datetime.strptime(folder, fmt)
        except ValueError:
            pass
    return None


def census(root):
    images = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if NAME.match(fn):
                images.append(os.path.relpath(os.path.join(dirpath, fn), root).replace("\\", "/"))
    return images


def brightness(args):
    root, rel = args
    try:
        with Image.open(os.path.join(root, rel)) as im:
            g = im.convert("L")
            w, h = g.size
            g = g.crop((0, int(np.ceil(TOP_CROP * h)), w, h))
            g.thumbnail((240, 240))
            a = np.asarray(g, dtype=np.float32) / 255.0
        return rel, float(a.mean()), float(np.percentile(a, 95)), float((a < 0.1).mean()), w, h
    except Exception:                                          # unreadable file
        return rel, None, None, None, None, None


def contact_sheet(rows, root, out, n=48, cols=8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = rows[:n]
    r = int(np.ceil(len(rows) / cols))
    fig, axes = plt.subplots(r, cols, figsize=(cols * 2.2, r * 1.9))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, row in zip(np.atleast_1d(axes).ravel(), rows):
        with Image.open(os.path.join(root, row["path"])) as im:
            im = im.convert("RGB")
            im.thumbnail((300, 300))
            ax.imshow(im)
        ax.set_title(f"{row['mean']:.2f} | {row['visibility_mi']} mi | {row['folder'][:16]}", fontsize=6)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Census of NewWebcams and a low-light check of the subset")
    ap.add_argument("--dset_path", default=r"D:\Research - Lasya\NewWebcams")
    ap.add_argument("--splits_path", default="splits")
    ap.add_argument("--lowlight_scope", choices=("subset", "all"), default="subset")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--out", default="dataset_audit")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    lines = []
    say = lambda s="": (print(s), lines.append(s))

    # ---- 1. census of the whole folder ----------------------------------
    print("Scanning the folder (file names only)...")
    allimgs = census(args.dset_path)
    site = lambda p: os.path.basename(p).split("_")[0]
    view = lambda p: "_".join(os.path.basename(p).split("_")[:2])
    folder = lambda p: os.path.basename(os.path.dirname(p))
    times = {f: folder_time(f) for f in {folder(p) for p in allimgs}}
    ok_times = [t for t in times.values() if t]
    raw, cls, bad_names = Counter(), Counter(), 0
    for p in allimgs:
        try:
            v = visibility(os.path.basename(p))
            raw[v] += 1
            cls[vis_class(v)] += 1
        except Exception:
            bad_names += 1
    say("=== NewWebcams census ===")
    say(f"images: {len(allimgs)}   sites: {len({site(p) for p in allimgs})}   "
        f"camera views: {len({view(p) for p in allimgs})}   capture folders: {len(times)}")
    if ok_times:
        say(f"dates: {min(ok_times):%Y-%m-%d} to {max(ok_times):%Y-%m-%d}   "
            f"folders with unreadable timestamps: {sum(t is None for t in times.values())}")
    say("reported values: " + ", ".join(f"{k:g} mi: {v}" for k, v in sorted(raw.items())))
    say("classes 1-10: " + ", ".join(f"{k}: {cls[k]}" for k in range(1, 11)) +
        f"   outside the ten classes: {cls[None]}   unparseable names: {bad_names}")
    with open(os.path.join(args.out, "class_counts.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["visibility_mi", "images_in_folder"])
        w.writerows(sorted(raw.items()))

    # ---- 2. is the balanced subset inside the folder? -----------------------
    subset = {}
    for split in ("train", "validation", "test"):
        with open(os.path.join(args.splits_path, f"{split}.csv")) as f:
            for line in f:
                if line.strip():
                    subset[line.strip()] = split
    present = set(allimgs)
    missing = [p for p in subset if p not in present]
    say()
    say("=== Balanced subset ===")
    say(f"listed: {len(subset)}   found in the folder: {len(subset) - len(missing)}   missing: {len(missing)}"
        f"   sites: {len({site(p) for p in subset})}")

    # ---- 3. brightness ------------------------------------------------------
    scope = [p for p in subset if p in present] if args.lowlight_scope == "subset" else allimgs
    print(f"Measuring brightness of {len(scope)} images with {args.workers} workers...")
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, res in enumerate(ex.map(brightness, [(args.dset_path, p) for p in scope], chunksize=64), 1):
            results.append(res)
            if i % 2000 == 0:
                print(f"  {i}/{len(scope)}")
    rows = []
    for rel, m, p95, dark, w, h in results:
        if m is None:
            continue
        t = times.get(folder(rel))
        try:
            v = visibility(os.path.basename(rel))
        except Exception:
            v = None
        rows.append(dict(path=rel, split=subset.get(rel, ""), site=site(rel), visibility_mi=v,
                         vis_class=vis_class(v), folder=folder(rel),
                         hour=t.hour if t else "", width=w, height=h,
                         mean=round(m, 4), p95=round(p95, 4), dark_frac=round(dark, 4)))
    rows.sort(key=lambda r: r["mean"])
    with open(os.path.join(args.out, "lowlight.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    means = np.array([r["mean"] for r in rows])
    say()
    say(f"=== Brightness of {len(rows)} images ({args.lowlight_scope}; {len(results) - len(rows)} unreadable) ===")
    say("mean luminance below the sky crop, 0 = black, 1 = white")
    say("percentiles 0/1/5/25/50: " + " / ".join(f"{np.percentile(means, q):.3f}" for q in (0, 1, 5, 25, 50)))
    for thr in (0.05, 0.10, 0.15, 0.20):
        dark = [r for r in rows if r["mean"] < thr]
        by_class = Counter(r["vis_class"] for r in dark)
        say(f"mean < {thr:.2f}: {len(dark)} images" +
            (f", sites {len({r['site'] for r in dark})}, by class " +
             ", ".join(f"{k}: {by_class[k]}" for k in sorted(by_class)) if dark else ""))
    sizes = Counter((r["width"], r["height"]) for r in rows)
    say("image sizes (width x height): " + ", ".join(f"{w}x{h}: {n}" for (w, h), n in sizes.most_common(6)))
    hours = Counter(r["hour"] for r in rows if r["mean"] < 0.15)
    if hours:
        say("folder hour of the images with mean < 0.15: " +
            ", ".join(f"{h}h: {n}" for h, n in sorted(hours.items(), key=lambda x: str(x[0]))))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(means, bins=60, color="#4c72b0")
    ax.set_xlabel("Mean luminance below the sky crop")
    ax.set_ylabel("Images")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "lowlight_hist.png"), dpi=150)
    plt.close(fig)
    contact_sheet(rows, args.dset_path, os.path.join(args.out, "lowlight_darkest.png"))

    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved {args.out}/summary.txt, class_counts.csv, lowlight.csv, "
          f"lowlight_hist.png, lowlight_darkest.png")


if __name__ == "__main__":
    main()
