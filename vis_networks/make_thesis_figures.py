import argparse
import csv
import glob
import os
from collections import Counter, defaultdict
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from thesis_common import (CONFIGS, arrays, collection_rows, event_of, label, leakage_free, load_all,
                           pooled_splits, site_of, vis_class)

OUT = "thesis_figures"          

mpl.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.unicode_minus": True,
})
WIDTH = 6.0                                     # text width of the ETD template, inches
COL = {"VisNet": "#1b6ca8", "RMEP": "#c0392b", "VisNet-Similarity": "#2e8b57"}


def save(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("saved", name)

#chapter4

def fig_classes(repo):
    sp = pooled_splits(repo)
    allp = sp["train"] + sp["validation"] + sp["test"]
    sub = Counter(vis_class(p) for p in allp)
    folder = Counter(int(r["vis_class"]) for r in collection_rows(repo))
    x = np.arange(1, 11)
    fig, ax = plt.subplots(figsize=(WIDTH, 2.6))
    ax.bar(x - 0.2, [folder[c] for c in x], 0.4, color="#9aa5b1", label=f"Full collection ({sum(folder.values()):,} images)")
    ax.bar(x + 0.2, [sub[c] for c in x], 0.4, color="#1b6ca8", label=f"Balanced subset ({sum(sub.values()):,} images)")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(mpl.ticker.FixedLocator([1000, 2000, 5000, 10000, 20000, 50000]))
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_ylim(1000, 80000)
    ax.set_xticks(x)
    ax.set_xlabel("Visibility class (nominal miles)")
    ax.set_ylabel("Images (log scale)")
    ax.legend(frameon=False, loc="upper left")
    for c in (9, 10):
        ax.text(c - 0.2, folder[c] * 1.12, f"{folder[c]:,}", ha="center", va="bottom", fontsize=8)
    save(fig, "fig4_class_distribution")


def fig_time_and_sites(repo):
    sp = pooled_splits(repo)
    allp = sp["train"] + sp["validation"] + sp["test"]

    def month(p):
        f = event_of(p)
        for fm in ("%Y-%m-%d-%H-%M-%S", "%m-%d-%y-%I-%M-%p"):
            try:
                return datetime.strptime(f, fm).strftime("%Y-%m")
            except ValueError:
                pass
    groups = [("Classes 1–3 (1–3 mi)", range(1, 4), "#c0392b"),
              ("Classes 4–8 (4–8 mi)", range(4, 9), "#e5a03a"),
              ("Classes 9–10 (9–10+ mi)", range(9, 11), "#1b6ca8")]
    months = sorted({month(p) for p in allp})
    counts = {g: Counter(month(p) for p in allp if vis_class(p) in rng) for g, rng, _ in groups}
    fig, ax = plt.subplots(figsize=(WIDTH, 2.5))
    bottom = np.zeros(len(months))
    for g, _, c in groups:
        v = np.array([counts[g][m] for m in months])
        ax.bar(range(len(months)), v, 0.75, bottom=bottom, color=c, label=g)
        bottom += v
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels([datetime.strptime(m, "%Y-%m").strftime("%b\n%Y") if i == 0 or m.endswith("-01")
                        else datetime.strptime(m, "%Y-%m").strftime("%b") for i, m in enumerate(months)])
    ax.set_ylabel("Images")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    save(fig, "fig4_months")

    per = Counter(site_of(p) for p in allp)
    cls = defaultdict(set)
    for p in allp:
        cls[site_of(p)].add(vis_class(p))
    n = np.array(list(per.values()))
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 2.5), gridspec_kw=dict(width_ratios=[1.35, 1]))
    bins = np.arange(0, 300, 10)
    a.hist(n, bins=bins, color="#1b6ca8", edgecolor="white", linewidth=0.5)
    a.axvline(20, color="black", ls="--", lw=0.9)
    a.annotate(f"{(n >= 20).sum()} sites with\n20 or more images", xy=(20, 30), xytext=(95, 45),
               fontsize=8, va="center", arrowprops=dict(arrowstyle="->", lw=0.7))
    a.set_xlabel("Images per site")
    a.set_ylabel("Sites")
    a.set_title("(a) Images per site", loc="left")
    k = Counter(len(v) for v in cls.values())
    b.bar(range(1, 11), [k[i] for i in range(1, 11)], 0.75, color="#1b6ca8")
    b.set_xticks(range(1, 11))
    b.set_xlabel("Visibility classes present at the site")
    b.set_ylabel("Sites")
    b.set_title("(b) Classes per site", loc="left")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig4_sites")


#chapter6

def read_metrics(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return (np.array([int(r["epoch"]) for r in rows]), np.array([float(r["accuracy"]) for r in rows]),
            np.array([float(r["loss"]) for r in rows]))


def fig_curves(repo):
    runs = [("VisNet", "VisNetGlobalRunCSvs", 19), ("RMEP", "RMEPGlobalrunCSVs", 37),
            ("VisNet-Similarity", "VisNetSimilarity", 4)]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.4))
    for name, folder, best in runs:
        e, tr_acc, tr_loss = read_metrics(os.path.join(repo, folder, "train_metrics.csv"))
        e2, va_acc, va_loss = read_metrics(os.path.join(repo, folder, "validation_metrics.csv"))
        assert np.argmin(va_loss) + 1 == best, (name, np.argmin(va_loss) + 1)
        c = COL[name]
        axes[0].plot(e, 100 * tr_acc, color=c, ls="--", lw=1)
        axes[0].plot(e2, 100 * va_acc, color=c, lw=1.2, label=name)
        axes[0].plot(best, 100 * va_acc[best - 1], "o", color=c, ms=4)
        ax = axes[2] if name == "VisNet-Similarity" else axes[1]
        ax.plot(e, tr_loss, color=c, ls="--", lw=1)
        ax.plot(e2, va_loss, color=c, lw=1.2)
        ax.plot(best, va_loss[best - 1], "o", color=c, ms=4)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("(a) Accuracy", loc="left")
    axes[1].set_ylabel("Cross-entropy loss")
    axes[1].set_title("(b) Loss: VisNet, RMEP", loc="left")
    axes[2].set_ylabel("Composite loss")
    axes[2].set_title("(c) Loss: VisNet-Similarity", loc="left")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_xlim(0, 81)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=COL[n], lw=1.2, label=n) for n in COL]
    h += [Line2D([], [], color="gray", ls="--", lw=1, label="training"),
          Line2D([], [], color="gray", lw=1.2, label="validation"),
          Line2D([], [], color="gray", marker="o", ls="", ms=4, label="selected epoch")]
    fig.legend(handles=h, ncol=6, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               fontsize=8, handlelength=1.6, columnspacing=1.0)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig6_training_curves")


def per_site(preds, name, paths, key):
    by = defaultdict(list)
    for p in paths:
        by[site_of(p)].append(p)
    out = {}
    for s, ps in by.items():
        t, q = arrays(preds, name, ps)
        if key == "logmae":
            out[s] = np.mean(np.abs(np.log10(t) - np.log10(q)))
    return out


def fig_persite_box(repo):
    preds, common = load_all(repo)
    lf = leakage_free(repo, common)
    order = [("Pooled", "Pooled"), ("Scratch", "Scratch"), ("Fine-tuned", "Fine-tuned")]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.9))
    pos, data, cols = [], [], []
    for gi, (reg, _) in enumerate(order):
        for mi, model in enumerate(COL):
            r = "Per-site heads" if (model == "VisNet-Similarity" and reg == "Scratch") else reg
            v = per_site(preds, label(model, r), lf, "logmae")
            data.append(list(v.values()))
            pos.append(gi * 4 + mi)
            cols.append(COL[model])
    bp = ax.boxplot(data, positions=pos, widths=0.7, patch_artist=True, showfliers=True,
                    flierprops=dict(marker=".", ms=3, alpha=0.5), medianprops=dict(color="black", lw=1.2))
    for b, c in zip(bp["boxes"], cols):
        b.set_facecolor(c)
        b.set_alpha(0.75)
    ax.set_xticks([1, 5, 9])
    ax.set_xticklabels(["Pooled", "Per-site from scratch", "Per-site fine-tuned"])
    ax.set_ylabel(r"Per-site log-MAE, $\log_{10}$")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, alpha=0.75, label=n) for n, c in COL.items()], frameon=False,
              ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    save(fig, "fig6_persite_logmae")


def fig_vs_images_classes(repo):
    preds, common = load_all(repo)
    sel = [("RMEP", "Scratch"), ("RMEP", "Fine-tuned")]
    by = defaultdict(list)
    for p in common:
        by[site_of(p)].append(p)
    sites = sorted(by)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.7), sharey=True)
    for ax, (m, r) in zip(axes, sel):
        name = label(m, r)
        d = preds[name]
        n = np.array([int(d[by[s][0]]["n_train"]) for s in sites])
        k = np.array([len({float(d[p]["true_vis"]) for p in by[s]}) for s in sites])
        y = np.array([np.mean(np.abs(np.log10(arrays(preds, name, by[s])[0]) -
                                     np.log10(arrays(preds, name, by[s])[1]))) for s in sites])
        sc = ax.scatter(n, y, c=k, cmap="viridis", s=12, vmin=1, vmax=10, edgecolor="none", alpha=0.85)
        ax.set_xscale("log")
        ax.set_xlabel("Training images at the site (log scale)")
        ax.set_title(f"({'a' if r == 'Scratch' else 'b'}) RMEP, {r.lower()}", loc="left")
    axes[0].set_ylabel(r"Per-site log-MAE, $\log_{10}$")
    cb = fig.colorbar(sc, ax=axes, fraction=0.04, pad=0.02)
    cb.set_label("Classes in the site's test set")
    save(fig, "fig6_logmae_vs_images")

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.6), sharey=True)
    for ax, (m, r) in zip(axes, sel):
        name = label(m, r)
        d = preds[name]
        k = np.array([len({float(d[p]["true_vis"]) for p in by[s]}) for s in sites])
        y = np.array([np.mean(np.abs(np.log10(arrays(preds, name, by[s])[0]) -
                                     np.log10(arrays(preds, name, by[s])[1]))) for s in sites])
        ks = sorted(set(k))
        ax.boxplot([y[k == kk] for kk in ks], positions=ks, widths=0.6, showfliers=True,
                   flierprops=dict(marker=".", ms=3, alpha=0.5), medianprops=dict(color=COL["RMEP"], lw=1.2))
        for kk in ks:
            ax.text(kk, -0.03, f"{int((k == kk).sum())}", ha="center", va="top", fontsize=7, color="gray",
                    transform=ax.get_xaxis_transform())
        ax.set_xlabel("Classes in the site's test set")
        ax.set_title(f"({'a' if r == 'Scratch' else 'b'}) RMEP, {r.lower()}", loc="left")
        ax.xaxis.labelpad = 10
    axes[0].set_ylabel(r"Per-site log-MAE, $\log_{10}$")
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig6_logmae_vs_classes")


def fig_confusion(repo):
    preds, common = load_all(repo)
    lf = leakage_free(repo, common)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.9))
    for ax, r in zip(axes, ("Pooled", "Fine-tuned")):
        t, q = arrays(preds, label("RMEP", r), lf)
        m = np.zeros((10, 10))
        for a, b in zip(t.astype(int), q.astype(int)):
            m[a - 1, b - 1] += 1
        rown = m / np.maximum(m.sum(1, keepdims=True), 1)
        im = ax.imshow(rown, cmap="Blues", vmin=0, vmax=0.6)
        ax.set_xticks(range(10))
        ax.set_xticklabels(range(1, 11))
        ax.set_yticks(range(10))
        ax.set_yticklabels(range(1, 11))
        ax.set_xlabel("Predicted class")
        ax.set_title(f"({'a' if r == 'Pooled' else 'b'}) RMEP, {r.lower()}", loc="left")
        for i in range(10):
            for j in range(10):
                if m[i, j]:
                    ax.text(j, i, f"{int(m[i, j])}", ha="center", va="center", fontsize=6,
                            color="white" if rown[i, j] > 0.35 else "black")
        ax.spines[:].set_visible(True)
    axes[0].set_ylabel("Reported class")
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("Share of the row")
    save(fig, "fig6_confusion_rmep")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".", help="the vis_networks folder (default: the current folder)")
    ap.add_argument("--out", default="thesis_figures", help="output folder (default: thesis_figures)")
    args = ap.parse_args()
    global OUT
    OUT = args.out
    os.makedirs(OUT, exist_ok=True)
    fig_classes(args.repo)
    fig_time_and_sites(args.repo)
    fig_curves(args.repo)
    fig_persite_box(args.repo)
    fig_vs_images_classes(args.repo)
    fig_confusion(args.repo)


if __name__ == "__main__":
    main()
