import argparse
import os
import re
import shutil
from collections import defaultdict
from torchvision import io
import torch



def extract_site_id(path):
    return os.path.basename(path).split("_")[0]


_VIS_RE = re.compile(r'VIS(\d+)(?:-(\d+))?\+?mi', re.IGNORECASE)


def extract_visibility(path):
    m = _VIS_RE.search(os.path.basename(path))
    if not m:
        return -1
    whole = m.group(1)
    frac = m.group(2)
    return float(f"{whole}.{frac}") if frac else float(whole)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dset_path", required=True)
    parser.add_argument("--splits_path", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_samples = []
    for split_name in ["train.csv", "validation.csv", "test.csv"]:
        fpath = os.path.join(args.splits_path, split_name)
        if os.path.exists(fpath):
            with open(fpath) as f:
                all_samples.extend([l.rstrip() for l in f if l.strip()])

    sites = defaultdict(list)
    for s in all_samples:
        sites[extract_site_id(s)].append(s)

    print(f"Found {len(sites)} sites across {len(all_samples)} samples")

    for site_id, samples in sorted(sites.items()):
        best_vis = -1.0
        candidates = []
        for s in samples:
            vis = extract_visibility(s)
            if vis > best_vis:
                best_vis = vis
                candidates = [s]
            elif vis == best_vis:
                candidates.append(s)

        if best_vis < 0:
            print(f"  {site_id}: no valid visibility found, skipping")
            continue

        best_brightness = -1.0
        best_sample = candidates[0]
        for s in candidates:
            try:
                img = io.decode_image(os.path.join(args.dset_path, s), 'RGB') / 255.0
                brightness = img.mean().item()
                if brightness > best_brightness:
                    best_brightness = brightness
                    best_sample = s
            except Exception:
                continue

        src = os.path.join(args.dset_path, best_sample)
        ext = os.path.splitext(best_sample)[1]
        dst = os.path.join(args.output_dir, f"{site_id}_reference{ext}")
        shutil.copy2(src, dst)
        print(f"  {site_id}: vis={best_vis}, brightness={best_brightness:.3f} -> {os.path.basename(dst)}")

    print(f"\nDone. References saved to {args.output_dir}")


if __name__ == "__main__":
    main()
