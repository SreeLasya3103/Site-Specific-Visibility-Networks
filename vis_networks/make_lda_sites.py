import argparse
import csv
import os
from collections import Counter

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline

from lda_common import (ARCHS, default_splits, read_split, site_of, event_of, build_dataset,
                        build_model, embed, pooled_train_share)

mpl.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "mathtext.fontset": "stix", "font.family": "serif",
                     "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 10,
                     "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
                     "legend.fontsize": 8})


def heldout_accuracy(X, y, splits, pca_dim, seed):
    """Mean and SD of held-out accuracy over the given splits; PCA + LDA fitted on each train side."""
    accs = []
    for tr, te in splits:
        if len(np.unique(y[tr])) < len(np.unique(y)):
            continue                                       # a site absent from training cannot be recovered
        n_pca = int(min(pca_dim, len(tr) - 1, X.shape[1]))
        pipe = make_pipeline(PCA(n_components=n_pca, random_state=seed), LinearDiscriminantAnalysis())
        accs.append(pipe.fit(X[tr], y[tr]).score(X[te], y[te]))
    return float(np.mean(accs)), float(np.std(accs)), len(accs)


def main():
    ap = argparse.ArgumentParser(description="Site-identification LDA on pooled-model embeddings")
    ap.add_argument("--arch", required=True, choices=ARCHS)
    ap.add_argument("--checkpoint", required=True, help="pooled (global) best_epochNN.pt")
    ap.add_argument("--backbone", help="similarity only: the VisNet checkpoint")
    ap.add_argument("--ref_dir", help="similarity only: references_by_view folder")
    ap.add_argument("--untrained", action="store_true", help="control: random weights")
    ap.add_argument("--images", choices=("unseen", "persite_test"), default="unseen")
    ap.add_argument("--persite_predictions", default="RMEPFineTuneAndPredictionsResults/predictions.csv",
                    help="supplies the per-site test images when --images persite_test")
    ap.add_argument("--dset_path", default=r"D:\Research - Lasya\NewWebcams")
    ap.add_argument("--splits_path", help="default: splits (similarity: splits/similarity)")
    ap.add_argument("--n_sites", type=int, default=10)
    ap.add_argument("--pca_dim", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="lda_sites")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.images == "unseen":
        pool = read_split(args.splits_path or default_splits(args.arch), ["validation.csv", "test.csv"])
    else:
        with open(args.persite_predictions, newline="") as f:
            pool = list(dict.fromkeys(r["sample_path"] for r in csv.DictReader(f)))
    top = [s for s, _ in Counter(site_of(p) for p in pool).most_common(args.n_sites)]
    samples = [p for s in top for p in pool if site_of(p) == s]       # grouped by site, as earlier
    tag = f"{args.arch}{' (untrained)' if args.untrained else ''}"
    print(f"{tag}: {len(top)} sites, {len(samples)} images ({args.images}), "
          f"{pooled_train_share(samples)} of them pooled-training images")

    dataset = build_dataset(args.arch, args.dset_path, samples, args.ref_dir)
    model = build_model(args.arch, dataset[0][0].size(), args.checkpoint, args.backbone,
                        args.untrained, args.seed)
    X, _, paths = embed(args.arch, model, dataset)
    y = np.array([site_of(p) for p in paths])
    groups = np.array([event_of(p) for p in paths])
    print(f"  embeddings {X.shape}, {len(np.unique(groups))} capture events")

    parts = {"full": X}
    if args.arch == "similarity":                         # [p_curr | p_diff], 128 + 128
        parts["p_curr"], parts["p_diff"] = X[:, :X.shape[1] // 2], X[:, X.shape[1] // 2:]
    chance = 1.0 / len(top)
    rows = []
    for part, Xp in parts.items():
        grp = GroupShuffleSplit(n_splits=args.repeats, test_size=0.3, random_state=args.seed).split(Xp, y, groups)
        img = StratifiedShuffleSplit(n_splits=args.repeats, test_size=0.3, random_state=args.seed).split(Xp, y)
        acc_g, sd_g, n_g = heldout_accuracy(Xp, y, grp, args.pca_dim, args.seed)
        acc_i, sd_i, n_i = heldout_accuracy(Xp, y, img, args.pca_dim, args.seed)
        print(f"  held-out site identification from the {part} embedding ({Xp.shape[1]}-d), "
              f"chance {chance*100:.1f}%:")
        print(f"    event-grouped {acc_g*100:.1f}% +/- {sd_g*100:.1f} ({n_g} splits)")
        print(f"    image-level   {acc_i*100:.1f}% +/- {sd_i*100:.1f} ({n_i} splits)")
        rows.append([args.arch, part, args.untrained, args.seed, args.images, len(top), len(X),
                     len(np.unique(groups)), pooled_train_share(paths), Xp.shape[1], args.pca_dim,
                     round(acc_g, 4), round(sd_g, 4), n_g, round(acc_i, 4), round(sd_i, 4), round(chance, 4)])

    with open(f"{args.out}_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arch", "embedding", "untrained", "seed", "images", "n_sites", "n_images", "n_events",
                    "n_pooled_train", "embedding_dim", "pca_dim", "acc_event_grouped", "sd_event_grouped",
                    "splits_event_grouped", "acc_image_level", "sd_image_level", "chance"])
        w.writerows(rows)

    # Descriptive picture: PCA + LDA fitted on all of these images. The held-out
    # numbers above, not this plot, are the evidence.
    n_pca = int(min(args.pca_dim, X.shape[0] - 1, X.shape[1]))
    Z = LinearDiscriminantAnalysis(n_components=2).fit_transform(
        PCA(n_components=n_pca, random_state=args.seed).fit_transform(X), y)
    fig, ax = plt.subplots(figsize=(4.4, 4.3))
    fig.subplots_adjust(left=0.15, right=0.975, top=0.79, bottom=0.11)
    cmap = plt.get_cmap("tab10")
    for i, s in enumerate(top):
        m = y == s
        if m.sum() < 3:
            continue
        pts, c = Z[m], cmap(i % 10)
        ax.scatter(pts[:, 0], pts[:, 1], s=7, alpha=0.6, color=c, linewidths=0,
                   label=f"{s} (n={int(m.sum())})", zorder=3)
        vals, vecs = np.linalg.eigh(np.cov(pts, rowvar=False))
        o = vals.argsort()[::-1]; vals, vecs = vals[o], vecs[:, o]
        ax.add_patch(Ellipse(pts.mean(0), *(4 * np.sqrt(np.maximum(vals, 0))),     # 2-SD ellipse
                             angle=np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])),
                             fill=False, edgecolor=c, ls="--", lw=0.8, zorder=4))
    ax.set_xlabel("LDA direction 1"); ax.set_ylabel("LDA direction 2"); ax.tick_params(length=2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if args.label:
        ax.text(0.025, 0.975, args.label, transform=ax.transAxes, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#bbbbbb", lw=0.5), zorder=6)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.03), ncol=3, frameon=False,
              handletextpad=0.3, columnspacing=0.8, borderaxespad=0.0, markerscale=2.2)
    fig.savefig(f"{args.out}.png"); fig.savefig(f"{args.out}.pdf")
    print(f"  saved {args.out}.png/.pdf and {args.out}_summary.csv")


if __name__ == "__main__":
    main()
