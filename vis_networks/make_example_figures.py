import argparse
import os
import random
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import io

from dsets.img_util import get_resize_crop_fn
from dsets.webcams import common as cmmn
from models.visnet.transform import get_transformer

mpl.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 9, "axes.titlesize": 9})

CLASS_GROUPS = [{0.0, 1.0, 1.25, 1.5, 1.75}, {2.0, 2.25, 2.5}] + [{float(k)} for k in range(3, 11)]
DIM = (280, 280)


def visibility(name):
    s = name.split("_")[2].split(".")[0].split("S")[1].split("m")[0].replace("-", ".")
    return (10.0 if s == "10+" else min(float(s), 10.0)), s           # value, label as reported


def vis_class(v):
    for i, g in enumerate(CLASS_GROUPS):
        if v in g:
            return i + 1
    return None


def capture_time(rel):
    folder = os.path.basename(os.path.dirname(rel))
    for fmt in ("%Y-%m-%d-%H-%M-%S", "%m-%d-%y-%I-%M-%p"):
        try:
            return datetime.strptime(folder, fmt)
        except ValueError:
            pass
    return None


def view_of(rel):
    return "_".join(os.path.basename(rel).split("_")[:2])


def subset(splits_path):
    out = []
    for split in ("train", "validation", "test"):
        with open(os.path.join(splits_path, f"{split}.csv")) as f:
            out += [line.strip() for line in f if line.strip()]
    return out


def pick_per_class(paths):
    """For each class, the image captured closest to 13:00."""
    best = {}
    for p in paths:
        v, raw = visibility(os.path.basename(p))
        c, t = vis_class(v), capture_time(p)
        gap = abs((t.hour + t.minute / 60) - 13.0) if t else 99
        if c and (c not in best or gap < best[c][0]):
            best[c] = (gap, p, raw, t)
    return best


def auto_views(paths, n):
    by_view = {}
    for p in paths:
        by_view.setdefault(view_of(p), []).append(p)
    full = []
    for v, ps in by_view.items():
        if len({vis_class(visibility(os.path.basename(p))[0]) for p in ps}) == 10:
            midday = sum(1 for p in ps if capture_time(p) and 11 <= capture_time(p).hour <= 15)
            full.append((midday, len(ps), v))
    full.sort(reverse=True)
    return [v for _, _, v in full[:n]], by_view


def load_tensor(root, rel):
    return io.decode_image(os.path.join(root, rel), "RGB") / 255.0        # as in cls_10full.py


def to_img(t):
    return np.clip(t.permute(1, 2, 0).numpy(), 0, 1)


def fig_classes(view, best, root, out):
    fig, axes = plt.subplots(2, 5, figsize=(6.0, 2.45))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.02, wspace=0.04, hspace=0.24)
    for ax, c in zip(axes.ravel(), range(1, 11)):
        _, p, raw, t = best[c]
        with Image.open(os.path.join(root, p)) as im:
            ax.imshow(im.convert("RGB").resize((400, 300)))     # all source sizes are about 4:3
        ax.set_title(f"Class {c}: {raw} mi", pad=2)
        if t:
            ax.text(0.02, 0.04, t.strftime("%Y-%m-%d %H:%M"), transform=ax.transAxes, fontsize=6,
                    color="white", bbox=dict(fc="black", alpha=0.45, lw=0, pad=1))
        ax.axis("off")
    fig.savefig(f"{out}.png")
    fig.savefig(f"{out}.pdf")
    plt.close(fig)


def fig_preprocessing(rel, root, out):
    raw = load_tensor(root, rel)
    h, w = raw.shape[-2:]
    cropped = cmmn.crop_margins(raw)                                     # 310 x 468
    final = get_resize_crop_fn(DIM)(cropped)                             # 280 x 280
    ch, cw = cropped.shape[-2:]
    side = (cw - ch) / 2                                                 # what the square crop drops
    fig, axes = plt.subplots(1, 3, figsize=(6.0, 1.9), gridspec_kw=dict(width_ratios=[w / h, cw / ch, 1]))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.03, wspace=0.08)
    axes[0].imshow(to_img(raw))
    top = int(np.ceil(0.1281 * h))
    axes[0].add_patch(mpl.patches.Rectangle((-0.5, -0.5), w, top, color="red", alpha=0.35, lw=0))
    axes[0].add_patch(mpl.patches.Rectangle((3.5, top - 0.5), w - 8, h - top - 4, fill=False,
                                            ec="red", lw=0.8))
    axes[0].set_title(f"(a) Original, {w} × {h}\nshaded: top 12.81% removed", pad=2)
    axes[1].imshow(to_img(cropped))
    for x in (side, cw - side):
        axes[1].axvline(x, color="red", lw=0.8, ls="--")
    axes[1].set_title(f"(b) Cropped, resized to {cw} × {ch}\ndashed: square kept", pad=2)
    axes[2].imshow(to_img(final))
    axes[2].set_title(f"(c) Network input\n{DIM[1]} × {DIM[0]}", pad=2)
    for ax in axes:
        ax.axis("off")
    fig.savefig(f"{out}.png")
    fig.savefig(f"{out}.pdf")
    plt.close(fig)


def fig_visnet_views(items, root, out):
    tr = get_transformer(DIM)
    resize = get_resize_crop_fn(DIM)
    names = ("RGB", "Pseudocolor (blue channel)", "High-pass (blue channel)")
    fig, axes = plt.subplots(len(items), 3, figsize=(6.0, 2.05 * len(items)))
    axes = np.atleast_2d(axes)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.93, bottom=0.02, wspace=0.05, hspace=0.12)
    for row, (rel, label) in enumerate(items):
        x = tr(resize(cmmn.crop_margins(load_tensor(root, rel))))          # [3 views, 3, H, W]
        for col in range(3):
            axes[row, col].imshow(to_img(x[col]))
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(names[col], pad=3)
        axes[row, 0].set_ylabel(label)
    fig.savefig(f"{out}.png")
    fig.savefig(f"{out}.pdf")
    plt.close(fig)


def fig_cameras(paths, sites, n, root, out, seed=1):
    """One class-10 frame per camera site, captured closest to 13:00, shown uncropped."""
    by_site = {}
    for p in paths:
        if vis_class(visibility(os.path.basename(p))[0]) == 10:
            by_site.setdefault(os.path.basename(p).split("_")[0], []).append(p)
    chosen = [s for s in sites if s in by_site]
    missing = [s for s in sites if s not in by_site]
    if missing:
        print(f"  no class-10 images in the subset for {', '.join(missing)}; skipped")
    rest = sorted(set(by_site) - set(chosen))
    random.Random(seed).shuffle(rest)
    chosen = (chosen + rest)[:n]
    picks = []
    for s in chosen:
        def gap(p):
            t = capture_time(p)
            return abs((t.hour + t.minute / 60) - 13.0) if t else 99
        picks.append(min(by_site[s], key=gap))
    cols = 4
    rows = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.0, 1.38 * rows + 0.1))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.02, wspace=0.04, hspace=0.24)
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, p in zip(np.atleast_1d(axes).ravel(), picks):
        with Image.open(os.path.join(root, p)) as im:
            ax.imshow(im.convert("RGB").resize((400, 300)))
        _, raw = visibility(os.path.basename(p))
        t = capture_time(p)
        site, ornt = view_of(p).split("_")
        ax.set_title(f"{site} ({ornt.replace('ORNT', '')}\u00b0)", pad=2, fontsize=8)   # camera bearing
        label = f"{raw} mi" + (t.strftime(" | %Y-%m-%d %H:%M") if t else "")
        ax.text(0.02, 0.04, label, transform=ax.transAxes, fontsize=6,
                color="white", bbox=dict(fc="black", alpha=0.45, lw=0, pad=1))
    fig.savefig(f"{out}.png")
    fig.savefig(f"{out}.pdf")
    plt.close(fig)
    return picks


def main():
    ap = argparse.ArgumentParser(description="Example-image figures for the thesis")
    ap.add_argument("--dset_path", default=r"D:\Research - Lasya\NewWebcams")
    ap.add_argument("--splits_path", default="splits")
    ap.add_argument("--views", default="", help="comma-separated SITEx_ORNTy; default: 4 best candidates")
    ap.add_argument("--figures", default="views,cameras",
                    help="which figures: views (per-view figures), cameras (10-mi cameras figure), or both")
    ap.add_argument("--cameras", default="SITE19,SITE41,SITE747,SITE492,SITE760,SITE203",
                    help="sites for the 10-mi cameras figure; the rest are filled at random (seed 1)")
    ap.add_argument("--n_cameras", type=int, default=8)
    ap.add_argument("--out", default="figures_examples")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)
    which = {f.strip() for f in args.figures.split(",")}

    paths = subset(args.splits_path)
    if "cameras" in which:
        sites = [s.strip() for s in args.cameras.split(",") if s.strip()]
        out = os.path.join(args.out, "cameras_10mi")
        picks = fig_cameras(paths, sites, args.n_cameras, args.dset_path, out)
        print(f"saved {out} (.png/.pdf):")
        for p in picks:
            print(f"      {p}")
    if "views" not in which:
        return
    candidates, by_view = auto_views(paths, 4)
    views = [v.strip() for v in args.views.split(",") if v.strip()] or candidates
    print("views with all ten classes, best first:", ", ".join(candidates))
    for v in views:
        best = pick_per_class(by_view.get(v, []))
        if len(best) < 10:
            print(f"  {v}: only {len(best)} classes in the subset, skipped")
            continue
        base = os.path.join(args.out, v)
        fig_classes(v, best, args.dset_path, f"{base}_classes")
        fig_preprocessing(best[10][1], args.dset_path, f"{base}_preprocessing")
        fig_visnet_views([(best[10][1], "Clear (10 mi class)"), (best[2][1], "Reduced (2 mi class)")],
                         args.dset_path, f"{base}_visnet_views")
        print(f"  {v}: saved {base}_classes, _preprocessing, _visnet_views (.png/.pdf)")
        for c in range(1, 11):
            print(f"      class {c:2d}: {best[c][1]}")


if __name__ == "__main__":
    main()
