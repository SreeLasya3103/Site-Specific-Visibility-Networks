import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

BASE = "."
OUT = os.path.join(BASE, "fig2_r2dist.png")

LABEL_SMALL = True
MERGE_TOP_TIER = False
FS_IN = 7.5
FS_OUT = 7.0
BAR_W = 0.52

mpl.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300,
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
})

CONFIGS = [
    ("VisNet\nScratch",    "VisNetPersiteAndPredictionsResults"),
    ("RMEP\nScratch",      "RMEPPerSiteAndPredictionsResults"),
    ("VisNet\nFine-tuned", "VisNetFineTuneAndPredictionsResults"),
    ("RMEP\nFine-tuned",   "RMEPFineTuneAndPredictionsResults"),
]

TIERS = [
    (-np.inf, 0.00,   r"Poor ($R^2 \leq 0$)",     "#e6e6e6", "black", "<=0"),
    (0.00,    0.50,   r"Fair ($0$–$0.5$)",        "#b5b5b5", "black", "0-0.5"),
    (0.50,    0.89,   r"Good ($0.5$–$0.89$)",     "#767676", "white", "0.5-0.89"),
    (0.89,    np.inf, r"Excellent ($>0.89$)",     "#242424", "white", ">0.89"),
]

if MERGE_TOP_TIER:
    TIERS = TIERS[:2] + [(0.50, np.inf, r"Good ($R^2 > 0.5$)", "#5a5a5a", "white", ">0.5")]


def resolve(folder):
    direct = os.path.join(BASE, folder, "persite_results.csv")
    if os.path.exists(direct):
        return direct
    for entry in os.listdir(BASE):
        if entry.lower() == folder.lower():
            return os.path.join(BASE, entry, "persite_results.csv")
    raise FileNotFoundError(f"{folder}/persite_results.csv not found under {os.path.abspath(BASE)}")


def site_r2(folder):
    df = pd.read_csv(resolve(folder))
    df = df[df["site_id"] != "OVERALL"]
    return df["R2"].dropna().values


data = [site_r2(f) for _, f in CONFIGS]
x = np.arange(len(CONFIGS))

counts = np.zeros((len(TIERS), len(CONFIGS)), dtype=int)
for i, (lo, hi, _, _, _, _) in enumerate(TIERS):
    for j, v in enumerate(data):
        counts[i, j] = int(np.sum(v > lo) if np.isinf(hi) else np.sum((v > lo) & (v <= hi)))

fig, ax = plt.subplots(figsize=(3.4, 2.7), constrained_layout=True)
ax.set_axisbelow(True)
ax.yaxis.grid(True, color="#d0d0d0", linewidth=0.4)

segments = []                      # (bar index, y_bottom, height, text colour)
bottoms = np.zeros(len(CONFIGS))
for i, (_, _, lab, face, txt, _) in enumerate(TIERS):
    ax.bar(x, counts[i], BAR_W, bottom=bottoms, color=face, label=lab,
           edgecolor="white", linewidth=0.6, zorder=3)
    for j in range(len(CONFIGS)):
        if counts[i, j] > 0:
            segments.append((j, bottoms[j], int(counts[i, j]), txt))
    bottoms = bottoms + counts[i]

ax.set_ylabel("Number of sites")
ax.set_xticks(x)
ax.set_xticklabels([lab for lab, _ in CONFIGS])
ax.tick_params(axis="x", length=0)
ax.set_xlim(-0.55, len(CONFIGS) - 1 + 0.75)   # headroom for the outside labels
ax.set_ylim(0, bottoms.max() * 1.14)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)

handles, labels = ax.get_legend_handles_labels()
handles, labels = handles[::-1], labels[::-1]          # darkest (top of stack) first
ncol = 2 if len(TIERS) == 4 else 3
if ncol == 2:                                          # matplotlib fills column-major;
    order = [0, 2, 1, 3]                               # reorder so it reads left-to-right
    handles = [handles[i] for i in order]
    labels = [labels[i] for i in order]
ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.02),
          ncol=ncol, frameon=False, columnspacing=1.0, handlelength=1.3,
          handletextpad=0.5, borderaxespad=0.0)

fig.canvas.draw()
ax_pt = ax.get_window_extent().height * 72.0 / fig.dpi
y0, y1 = ax.get_ylim()
pt_per_site = ax_pt / (y1 - y0)
min_inside = FS_IN * 1.5 / pt_per_site

outside = {}
for j, base, height, txt in segments:
    if height >= min_inside:
        ax.text(x[j], base + height / 2.0, str(height), ha="center", va="center",
                fontsize=FS_IN, color=txt, zorder=4)
    else:
        outside.setdefault(j, []).append((base + height / 2.0, height))

if LABEL_SMALL:
    sep = FS_OUT * 1.45 / pt_per_site
    for j, items in outside.items():
        items.sort()
        ys = [c for c, _ in items]
        for k in range(1, len(ys)):
            ys[k] = max(ys[k], ys[k - 1] + sep)
        offset = (sum(c for c, _ in items) - sum(ys)) / len(ys)   # recentre the group
        ys = [y + offset for y in ys]
        for (cy, n), ly in zip(items, ys):
            # Start the leader just inside the bar so it visibly touches the band
            # it refers to, and leave a clear gap before the digits.
            x0 = x[j] + BAR_W / 2.0 - 0.04
            x1 = x0 + 0.17
            ax.plot([x0, x1], [cy, ly], color="#555555", linewidth=0.5, zorder=5)
            # The white backing box keeps the y-grid from running through the number.
            ax.text(x1 + 0.05, ly, str(n), ha="left", va="center",
                    fontsize=FS_OUT, color="black", zorder=6,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.0))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=300, bbox_inches="tight")
fig.savefig(os.path.splitext(OUT)[0] + ".pdf", bbox_inches="tight")
print("saved", OUT, "and", os.path.splitext(OUT)[0] + ".pdf")

print(f"\n{'configuration':<16}" + "".join(f"{t[5]:>10}" for t in TIERS) + f"{'sites':>8}")
for j, (lab, _) in enumerate(CONFIGS):
    name = lab.replace("\n", " ")
    print(f"{name:<16}" + "".join(f"{counts[i, j]:>10d}" for i in range(len(TIERS)))
          + f"{len(data[j]):>8d}")
