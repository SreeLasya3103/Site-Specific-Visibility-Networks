#One clear-day reference per camera view (site + orientation) for VisNet-Similarity, drawn only from images no evaluation set contains: training images of both regimes, 
#or archive images outside the dataset that share no capture event with it. Highest reported visibility wins; ties go to the brightest image. Writes reference_manifest.csv.
#Run used for the thesis (from vis_networks/): python select_references.py --dset_path "D:\Research - Lasya\NewWebcams" --splits_path "D:\Research - Lasya\NewWebcams" --output_dir "D:\Research - Lasya\NewWebcams\references_by_view" --pool_path "D:\Research - Lasya\NewWebcams"

import argparse
import csv
import argparse
import csv
import os
import random
import re
import shutil
from collections import defaultdict

IMG_EXT = ('.jpg', '.jpeg', '.png')
_VIS_RE = re.compile(r'VIS(\d+)(?:-(\d+))?\+?mi', re.IGNORECASE)


def extract_site_id(path):
    return os.path.basename(path).split("_")[0]


def extract_view_id(path):
    return "_".join(os.path.basename(path).split("_")[:2])


def extract_event(path):
    return os.path.basename(os.path.dirname(path))


def extract_visibility(path):
    m = _VIS_RE.search(os.path.basename(path))
    if not m:
        return -1
    whole = m.group(1)
    frac = m.group(2)
    return float(f"{whole}.{frac}") if frac else float(whole)


def read_split(splits_path, name):
    with open(os.path.join(splits_path, name)) as f:
        return [line.rstrip() for line in f if line.strip()]


def brightness(path):
    from torchvision import io
    return (io.decode_image(path, 'RGB') / 255.0).mean().item()


def main():
    parser = argparse.ArgumentParser(
        description="One clear-day reference per camera view, from images no evaluation set contains")
    parser.add_argument("--dset_path", required=True)
    parser.add_argument("--splits_path", required=True)
    parser.add_argument("--output_dir", required=True, help="Must be a new or empty folder")
    parser.add_argument("--pool_path", default=None,
                        help="WeatherCam images outside the dataset to draw references from")
    # These must match the per-site runs, or the per-site training portions differ.
    parser.add_argument("--min_samples", type=int, default=20)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_vis", type=float, default=10.0,
                        help="Report views whose best reference is below this visibility (mi)")
    parser.add_argument("--max_tied", type=int, default=50,
                        help="Brightness is measured on at most this many tied candidates per view")
    parser.add_argument("--allow_missing", action="store_true",
                        help="Finish even if some views get no reference")
    args = parser.parse_args()

    from trainpersite import split_site_data    # the exact per-site split the runs use

    if os.path.isdir(args.output_dir) and any(
            f.endswith(IMG_EXT) for f in os.listdir(args.output_dir)):
        raise SystemExit(f"{args.output_dir} already holds images; use a new folder so "
                         f"no per-site reference from the old design can be picked up.")
    os.makedirs(args.output_dir, exist_ok=True)

    # The same sample list trainpersite.py builds with --use_all_splits.
    train = read_split(args.splits_path, "train.csv")
    val = read_split(args.splits_path, "validation.csv")
    test = read_split(args.splits_path, "test.csv")
    all_samples = train + val + test
    pooled_train = set(train)

    by_site = defaultdict(list)
    for s in all_samples:
        by_site[extract_site_id(s)].append(s)
    per_site_train = {}
    for sid, samples in by_site.items():
        if len(samples) >= args.min_samples:
            tr, _, _ = split_site_data(samples, train_ratio=args.train_ratio,
                                       seed=args.seed, val_ratio=args.val_ratio)
            per_site_train[sid] = set(tr)

    def training_only(s):
        sid = extract_site_id(s)
        return s in pooled_train and (sid not in per_site_train or s in per_site_train[sid])

    views = sorted({extract_view_id(s) for s in all_samples})
    candidates = defaultdict(list)                  # view -> [(full path, name, source)]
    for s in all_samples:
        if training_only(s):
            candidates[extract_view_id(s)].append((os.path.join(args.dset_path, s), s, "dataset"))

    if args.pool_path:
        dataset_events = {(extract_site_id(s), extract_event(s)) for s in all_samples}
        wanted = set(views)
        n_pool = n_same_event = 0
        for root, _, files in os.walk(args.pool_path):
            for fname in sorted(files):
                if not fname.lower().endswith(IMG_EXT):
                    continue
                full = os.path.join(root, fname)
                rel = os.path.relpath(full, args.pool_path)
                if extract_view_id(rel) not in wanted:
                    continue
                if (extract_site_id(rel), extract_event(rel)) in dataset_events:
                    n_same_event += 1
                    continue
                candidates[extract_view_id(rel)].append((full, rel, "pool"))
                n_pool += 1
        print(f"Pool: {n_pool} usable images; {n_same_event} skipped for sharing a "
              f"capture event with the dataset")

    rng = random.Random(args.seed)
    manifest, missing, low = [], [], []
    for view in views:
        scored = [(full, rel, src, extract_visibility(rel))
                  for full, rel, src in candidates.get(view, [])]
        scored = [c for c in scored if c[3] >= 0]
        if not scored:
            missing.append(view)
            continue
        best_vis = max(c[3] for c in scored)
        tied = sorted((c for c in scored if c[3] == best_vis), key=lambda c: c[1])
        if len(tied) > args.max_tied:
            tied = rng.sample(tied, args.max_tied)
        chosen, chosen_b = tied[0], -1.0
        for c in tied:
            try:
                b = brightness(c[0])
            except Exception:
                continue
            if b > chosen_b:
                chosen, chosen_b = c, b
        full, rel, src, vis = chosen
        dst = os.path.join(args.output_dir, f"{view}_reference{os.path.splitext(rel)[1]}")
        shutil.copy2(full, dst)
        manifest.append({"view": view, "reference": rel, "source": src, "visibility_mi": vis,
                         "brightness": round(chosen_b, 4), "candidates": len(scored)})
        if vis < args.min_vis:
            low.append(view)

    # Audit: no dataset-sourced reference may sit in any evaluation set.
    per_site_eval = {s for s in all_samples
                     if extract_site_id(s) in per_site_train
                     and s not in per_site_train[extract_site_id(s)]}
    evaluation = set(val) | set(test) | per_site_eval
    leaked = [m["reference"] for m in manifest
              if m["source"] == "dataset" and m["reference"] in evaluation]
    if leaked:
        raise SystemExit(f"BUG: {len(leaked)} references are evaluation images: {leaked[:5]}")

    with open(os.path.join(args.output_dir, "reference_manifest.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["view", "reference", "source", "visibility_mi",
                                               "brightness", "candidates"])
        writer.writeheader()
        writer.writerows(manifest)

    n_pool_refs = sum(1 for m in manifest if m["source"] == "pool")
    print(f"\nCamera views: {len(views)}   references: {len(manifest)} "
          f"({n_pool_refs} from the pool, {len(manifest) - n_pool_refs} from the dataset)")
    print(f"Below {args.min_vis:g} mi: {len(low)}" + (f"  e.g. {low[:5]}" if low else ""))
    print(f"No reference: {len(missing)}" + (f"  e.g. {missing[:5]}" if missing else ""))
    print(f"Manifest: {os.path.join(args.output_dir, 'reference_manifest.csv')}")
    if missing and not args.allow_missing:
        raise SystemExit("Some views have no reference. Point --pool_path at more imagery, "
                         "or pass, allow_missing and exclude those views explicitly.")


if __name__ == "__main__":
    main()
