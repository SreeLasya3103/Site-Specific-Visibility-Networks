import argparse
import csv
import os
import warnings

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GroupShuffleSplit

from lda_common import (ARCHS, default_splits, read_split, site_of, event_of, build_dataset,
                        build_model, embed, pooled_train_share)

# Small classes make some PCA directions nearly collinear within a class; LDA's
# SVD solver handles that, so the warning is noise here.
warnings.filterwarnings("ignore", message="Variables are collinear")

mpl.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "mathtext.fontset": "stix", "font.family": "serif",
                     "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 10,
                     "axes.labelsize": 10, "axes.titlesize": 10, "xtick.labelsize": 9,
                     "ytick.labelsize": 9})


def fold_stats(X, vis, groups, pca_dim, repeats, seed):
    """Event-grouped held-out class accuracy and Spearman rho on LDA direction 1."""
    acc, rho = [], []
    for tr, te in GroupShuffleSplit(n_splits=repeats, test_size=0.3, random_state=seed).split(X, vis, groups):
        if len(np.unique(vis[tr])) < 2:
            continue
        n_pca = int(min(pca_dim, len(tr) - 1, X.shape[1]))
        pca = PCA(n_components=n_pca, random_state=seed).fit(X[tr])
        lda = LinearDiscriminantAnalysis().fit(pca.transform(X[tr]), vis[tr])
        acc.append(lda.score(pca.transform(X[te]), vis[te]))
        z_tr = lda.transform(pca.transform(X[tr]))[:, 0]
        z_te = lda.transform(pca.transform(X[te]))[:, 0]
        sign = np.sign(spearmanr(z_tr, vis[tr])[0]) or 1.0   # an LDA axis has no inherent sign
        if len(np.unique(vis[te])) > 1:
            rho.append(sign * spearmanr(z_te, vis[te])[0])
    return np.mean(acc), np.std(acc), np.mean(rho), np.std(rho), len(acc)


def main():
    ap = argparse.ArgumentParser(description="Per-site LDA of pooled-model embeddings, labelled by visibility")
    ap.add_argument("--arch", required=True, choices=ARCHS)
    ap.add_argument("--checkpoint", required=True, help="pooled (global) best_epochNN.pt")
    ap.add_argument("--backbone", help="similarity only: the VisNet checkpoint")
    ap.add_argument("--ref_dir", help="similarity only: references_by_view folder")
    ap.add_argument("--sites", default="SITE22,SITE43")
    ap.add_argument("--labels", default="SITE22 (best),SITE43 (worst)")
    ap.add_argument("--dset_path", default=r"D:\Research - Lasya\NewWebcams")
    ap.add_argument("--splits_path", help="default: splits (similarity: splits/similarity)")
    ap.add_argument("--pca_dim", type=int, default=10,
                    help="kept small: SITE43 has about 50 training images per split")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="lda_onesite")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    sites = [s.strip() for s in args.sites.split(",")]
    titles = [t.strip() for t in args.labels.split(",")]
    if len(titles) != len(sites):
        raise SystemExit("ERROR: --labels needs one entry per site")
    all_samples = read_split(args.splits_path or default_splits(args.arch),
                             ["train.csv", "validation.csv", "test.csv"])

    results, model = {}, None
    for site in sites:
        samples = [p for p in all_samples if site_of(p) == site]
        if not samples:
            raise SystemExit(f"ERROR: no images for {site}")
        dataset = build_dataset(args.arch, args.dset_path, samples, args.ref_dir)
        if model is None:
            model = build_model(args.arch, dataset[0][0].size(), args.checkpoint, args.backbone,
                                seed=args.seed)
        X, vis, paths = embed(args.arch, model, dataset)
        groups = np.array([event_of(p) for p in paths])
        acc, acc_sd, rho, rho_sd, n = fold_stats(X, vis, groups, args.pca_dim, args.repeats, args.seed)
        _, counts = np.unique(vis, return_counts=True)
        base = counts.max() / counts.sum()
        n_pca = int(min(args.pca_dim, len(X) - 1, X.shape[1]))
        Z = LinearDiscriminantAnalysis(n_components=min(2, len(counts) - 1)).fit_transform(
            PCA(n_components=n_pca, random_state=args.seed).fit_transform(X), vis)
        if Z.shape[1] == 1:
            Z = np.column_stack([Z[:, 0], np.zeros(len(Z))])
        results[site] = dict(Z=Z, vis=vis, n=len(X), events=len(np.unique(groups)), k=len(counts),
                             trained=pooled_train_share(paths), acc=acc, acc_sd=acc_sd, base=base,
                             rho=rho, rho_sd=rho_sd, folds=n)
        r = results[site]
        print(f"{site}: {r['n']} images ({r['trained']} pooled-training), {r['events']} events, "
              f"{r['k']} classes | held-out acc {acc*100:.1f}% +/- {acc_sd*100:.1f} "
              f"(majority {base*100:.1f}%) | held-out Spearman rho {rho:+.2f} +/- {rho_sd:.2f} ({n} splits)")

    with open(f"{args.out}_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arch", "site", "n_images", "n_pooled_train", "n_events", "n_classes", "pca_dim",
                    "acc_heldout", "acc_sd", "majority_rate", "spearman_rho_heldout", "rho_sd", "splits"])
        for s in sites:
            r = results[s]
            w.writerow([args.arch, s, r["n"], r["trained"], r["events"], r["k"], args.pca_dim,
                        round(r["acc"], 4), round(r["acc_sd"], 4), round(r["base"], 4),
                        round(r["rho"], 4), round(r["rho_sd"], 4), r["folds"]])

    fig, axes = plt.subplots(1, len(sites), figsize=(3.3 * len(sites) + 0.7, 3.1))
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=0.085, right=0.88, top=0.90, bottom=0.145, wspace=0.32)
    norm = mpl.colors.Normalize(vmin=1, vmax=10)
    for ax, s, title in zip(axes, sites, titles):
        r = results[s]
        ax.scatter(r["Z"][:, 0], r["Z"][:, 1], c=r["vis"], cmap="viridis", norm=norm, s=16,
                   alpha=0.85, linewidth=0.3, edgecolor="#303030", zorder=3)
        ax.set_title(title); ax.set_xlabel("LDA direction 1"); ax.set_ylabel("LDA direction 2")
        ax.tick_params(length=2); ax.set_axisbelow(True); ax.grid(True, color="#e4e4e4", linewidth=0.4)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=axes.tolist(),
                      pad=0.02, fraction=0.045, ticks=[2, 4, 6, 8, 10])
    cb.set_label("Visibility class (mi)", labelpad=3); cb.ax.tick_params(length=2); cb.outline.set_linewidth(0.5)
    fig.savefig(f"{args.out}.png"); fig.savefig(f"{args.out}.pdf")
    print(f"saved {args.out}.png/.pdf and {args.out}_summary.csv")


if __name__ == "__main__":
    main()
