"""
check_banners.py (v2) - does the 12.81% top crop remove the text banner (with the
weather report, visibility included) that some partner cameras print across the
top of the frame?

For every subset image of the listed sites, finds where the black banner band
ends and compares that with the crop line used by dsets/webcams/common.py.

Run from vis_networks/:
    python3 check_banners.py
Writes banner_check.csv and prints a summary.
"""

import argparse
import csv
import math
import os

import numpy as np
from PIL import Image

SITES = "SITE472,SITE492,SITE375,SITE383,SITE481,SITE484,SITE480,SITE476"
TOP_CROP = 0.1281


def banner_end(path):
    """Row where the black banner band ends (0 if the frame starts without one).

    Some frames (the 2025 ones) have a thin coloured border above the banner, so
    the band is looked for in the top 5% of the frame rather than at row 0.
    """
    with Image.open(path) as im:
        a = np.asarray(im.convert("RGB"), dtype=np.int32)
    h, w, _ = a.shape
    top = a[: int(0.3 * h), int(0.1 * w): int(0.9 * w)]
    near_black = (top.sum(axis=2) < 60).mean(axis=1)             # share of near-black pixels per row
    start_zone = near_black[: max(8, int(0.05 * h))]
    if start_zone.max() < 0.3:
        return 0, h                                               # no band near the top
    start = int(np.argmax(start_zone >= 0.3))                     # first banner row
    end, gap = start + 1, 0
    for y in range(start, len(near_black)):
        if near_black[y] >= 0.05:                                 # banner rows: black background, text
            end, gap = y + 1, 0
        else:
            gap += 1
            if gap >= 4:                                          # four banner-free rows: band is over
                break
    return end, h


def main():
    ap = argparse.ArgumentParser(description="Check that the top crop removes partner-camera banners")
    ap.add_argument("--dset_path", default=r"D:\Research - Lasya\NewWebcams")
    ap.add_argument("--splits_path", default="splits")
    ap.add_argument("--sites", default=SITES)
    ap.add_argument("--out", default="banner_check.csv")
    args = ap.parse_args()

    sites = {s.strip() for s in args.sites.split(",") if s.strip()}
    paths = []
    for split in ("train", "validation", "test"):
        with open(os.path.join(args.splits_path, f"{split}.csv")) as f:
            paths += [p.strip() for p in f if p.strip() and os.path.basename(p.strip()).split("_")[0] in sites]

    rows = []
    for p in paths:
        end, h = banner_end(os.path.join(args.dset_path, p))
        crop = math.ceil(TOP_CROP * h)
        rows.append(dict(path=p, site=os.path.basename(p).split("_")[0], height=h, banner_end_row=end,
                         crop_row=crop, banner_pct=round(100 * end / h, 2), margin_rows=crop - end))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with_banner = [r for r in rows if r["banner_end_row"] > 0]
    survives = [r for r in with_banner if r["banner_end_row"] > r["crop_row"]]
    print(f"{len(rows)} subset images from {len({r['site'] for r in rows})} sites; "
          f"{len(with_banner)} carry a black banner near the top")
    for year in sorted({r["path"][:4] for r in rows}):
        yr = [r for r in rows if r["path"][:4] == year]
        print(f"  {year}: {sum(r['banner_end_row'] > 0 for r in yr)} of {len(yr)} images with a banner")
    if with_banner:
        pct = np.array([r["banner_pct"] for r in with_banner])
        print(f"banner ends at {pct.min():.1f}-{pct.max():.1f}% of the height (median {np.median(pct):.1f}%); "
              f"the crop removes the top {100 * TOP_CROP:.2f}%")
        for s in sorted({r["site"] for r in with_banner}):
            ps = [r["banner_pct"] for r in with_banner if r["site"] == s]
            print(f"  {s}: {len(ps)} images, banner up to {max(ps):.1f}%")
    print(f"images whose banner reaches below the crop line: {len(survives)}")
    for r in sorted(survives, key=lambda r: -r["banner_pct"])[:10]:
        print(f"  {r['path']}: banner to {r['banner_pct']}% (crop {100 * r['crop_row'] / r['height']:.1f}%)")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
