"""
make_fig2_logmae.py

Per-site distribution of log-space mean absolute error.

Replaces the per-site R^2 figure. R^2 divides by each site's own label
variance over as few as two test images, so it is unstable here and no longer
separates the pooled and fine-tuned regimes. log-MAE has no such denominator
and weights a one-mile error at 1 mi far more heavily than at 10 mi, which is
the behaviour that matters operationally.

Every configuration is scored on the images common to all six result folders,
so the bars are comparable.

Run from vis_networks/ :

    python make_fig2_logmae.py

Writes fig2_logmae.png and .pdf beside the script.
"""

import csv
import os
import statistics
from collections import defaultdict

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ICASSP: nine point type or larger, Times-like, Type 42 fonts.
mpl.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "mathtext.fontset": "stix",
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.labelsize": 9,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
})

BASE = "."
OUT = "fig2_logmae"

CONFIGS = [
    ("VisNet", "Scratch",    "VisNetPerSiteAndPredictionsResults"),
    ("RMEP",   "Scratch",    "RMEPPerSiteAndPredictionsResults"),
    ("VisNet", "Fine-tuned", "VisNetFineTuneAndPredictionsResults"),
    ("RMEP",   "Fine-tuned", "RMEPFineTuneAndPredictionsResults"),
]
# Pooled folders are read only to define the common image set.
POOLED = ["VisNetGlobalOnPerSiteTest", "RMEPGlobalOnPerSiteTest"]

# Bands, worst first so the stack reads bottom-up from worst to best.
BANDS = [
    (0.30, np.inf, r"$>0.30$",       "#e8e8e8", "black"),
    (0.22, 0.30,   r"$0.22$–$0.30$", "#b5b5b5", "black"),
    (0.15, 0.22,   r"$0.15$–$0.22$", "#7a7a7a", "white"),
    (0.00, 0.15,   r"$<0.15$",       "#3a3a3a", "white"),
]

BAR_W = 0.52


def resolve(folder):
    """Locate a results folder case-insensitively (they were written on Windows)."""
    direct = os.path.join(BASE, folder, "predictions.csv")
    if os.path.exists(direct):
        return direct
    for entry in os.listdir(BASE):
        if entry.lower() == folder.lower():
            return os.path.join(BASE, entry, "predictions.csv")
    raise FileNotFoundError(f"{folder}/predictions.csv not found")


def load(folder):
    return {r["sample_path"]: r for r in csv.DictReader(open(resolve(folder), newline=""))}


tables = {f: load(f) for _, _, f in CONFIGS}
tables.update({f: load(f) for f in POOLED})
common = set.intersection(*[set(t) for t in tables.values()])
print(f"{len(common)} images common to all {len(tables)} configurations")

# Per-site log-MAE for each configuration.
per_site = {}
for model, regime, folder in CONFIGS:
    by = defaultdict(list)
    for path in common:
        row = tables[folder][path]
        by[row["site_id"]].append((float(row["true_vis"]), float(row["pred_vis"])))
    per_site[(model, regime)] = {
        site: float(np.mean([abs(np.log10(t) - np.log10(p)) for t, p in v]))
        for site, v in by.items()
    }

n_sites = len(next(iter(per_site.values())))
sizes = [len(v) for v in
         [[p for p in common if p.split("/")[1].split("_")[0] == s] for s in list(per_site.values())[0]]]

counts = np.zeros((len(BANDS), len(CONFIGS)), dtype=int)
for j, key in enumerate([(m, r) for m, r, _ in CONFIGS]):
    for e in per_site[key].values():
        for i, (lo, hi, *_) in enumerate(BANDS):
            if lo <= e < hi:
                counts[i, j] += 1
                break

fig, ax = plt.subplots(figsize=(3.39, 2.35))
fig.subplots_adjust(left=0.175, right=0.995, top=0.775, bottom=0.185)

x = np.arange(len(CONFIGS))
bottoms = np.zeros(len(CONFIGS))
for i, (lo, hi, label, face, txt) in enumerate(BANDS):
    ax.bar(x, counts[i], BAR_W, bottom=bottoms, color=face, label=label,
           edgecolor="white", linewidth=0.6, zorder=2)
    for j in range(len(CONFIGS)):
        if counts[i, j] >= 14:                      # only label segments with room
            ax.text(x[j], bottoms[j] + counts[i, j] / 2, str(counts[i, j]),
                    ha="center", va="center", color=txt,
                    fontsize=8, zorder=4)
    bottoms += counts[i]

ax.set_ylabel("Number of sites")
ax.set_xticks(x)
ax.set_xticklabels([m for m, _, _ in CONFIGS])
ax.set_xlim(-0.55, len(CONFIGS) - 0.45)
ax.set_ylim(0, n_sites * 1.02)
ax.tick_params(axis="x", length=0)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)

# Regime labels under the architecture names.
for xc, regime in ((0.5, "Scratch"), (2.5, "Fine-tuned")):
    ax.text(xc, -0.115, regime, transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=9)
ax.plot([1.5, 1.5], [-0.135, -0.015], transform=ax.get_xaxis_transform(),
        color="#999999", linewidth=0.5, clip_on=False)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc="lower center",
          bbox_to_anchor=(0.5, 1.01), ncol=4, frameon=False,
          handlelength=1.1, handletextpad=0.45, columnspacing=0.9,
          borderaxespad=0.0)

fig.savefig(f"{OUT}.png", dpi=300)
fig.savefig(f"{OUT}.pdf")
print(f"saved {OUT}.png and {OUT}.pdf")

print(f"\n{'configuration':<22}" + "".join(f"{b[2]:>12}" for b in BANDS)
      + f"{'median':>9}{'sites':>7}")
for j, (m, r, _) in enumerate(CONFIGS):
    med = statistics.median(per_site[(m, r)].values())
    print(f"{m + ' ' + r:<22}" + "".join(f"{counts[i, j]:>12d}" for i in range(len(BANDS)))
          + f"{med:>9.3f}{sum(counts[:, j]):>7d}")
