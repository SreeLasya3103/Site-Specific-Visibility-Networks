import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

BASE = "."
OUT = os.path.join(BASE, "fig2_r2dist.png")

LABEL_SMALL = True  
MERGE_TOP_TIER = False

FS_IN = 9.0
FS_OUT = 9.0
BAR_W = 0.52

mpl.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "mathtext.fontset": "stix",
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.labelsize": 9,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
})

CONFIGS = [
    ("VisNet", "Scratch",    "VisNetPersiteAndPredictionsResults"),
    ("RMEP",   "Scratch",    "RMEPPerSiteAndPredictionsResults"),
    ("VisNet", "Fine-tuned", "VisNetFineTuneAndPredictionsResults"),
    ("RMEP",   "Fine-tuned", "RMEPFineTuneAndPredictionsResults"),
]

TIERS = [
    (0.30, np.inf, r"$>0.30$",      "#e6e6e6", "black", ">.30"),
    (0.15, 0.30,   r"$0.15$–$0.30$", "#b5b5b5", "black", ".15-.30"),
    (0.05, 0.15,   r"$0.05$–$0.15$", "#767676", "white", ".05-.15"),
    (-np.inf, 0.05, r"$\leq 0.05$",  "#242424", "white", "<=.05"),
]

if MERGE_TOP_TIER:
    TIERS = [(-np.inf, 0.00, r"$R^2 \leq 0$", "#e6e6e6", "black", "<=0"),
             (0.00, 0.50, r"$0$–$0.5$", "#b5b5b5", "black", "0-0.5"),
             (0.50, np.inf, r"$R^2 > 0.5$", "#5a5a5a", "white", ">0.5")]


def resolve(folder):
    direct = os.path.join(BASE, folder, "persite_results.csv")
    if os.path.exists(direct):
        return direct
    for entry in os.listdir(BASE):
        if entry.lower() == folder.lower():
            return os.path.join(BASE, entry, "persite_results.csv")
    raise FileNotFoundError(f"{folder}/persite_results.csv not found under {os.path.abspath(BASE)}")


def load(folder):
    d = pd.read_csv(resolve(folder).replace("persite_results", "predictions"))
    g = d.groupby("site_id").apply(
        lambda x: np.abs(np.log10(x.true_vis) - np.log10(x.pred_vis)).mean(),
        include_groups=False)
    return g.values, None


loaded = [load(f) for _, _, f in CONFIGS]
data = [d for d, _ in loaded]
idents = [i for _, i in loaded]
x = np.arange(len(CONFIGS))

counts = np.zeros((len(TIERS), len(CONFIGS)), dtype=int)
for i, (lo, hi, _, _, _, _) in enumerate(TIERS):
    for j, v in enumerate(data):
        counts[i, j] = int(np.sum(v > lo) if np.isinf(hi) else np.sum((v > lo) & (v <= hi)))

fig, ax = plt.subplots(figsize=(3.39, 3.15))
fig.subplots_adjust(left=0.165, right=0.995, top=0.855, bottom=0.175)
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
ax.set_xticklabels([arch for arch, _, _ in CONFIGS])
ax.tick_params(axis="x", length=0)

for xc, regime in ((0.5, "Scratch"), (2.5, "Fine-tuned")):
    ax.text(xc, -0.105, regime, transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=9, clip_on=False)
ax.plot([1.5, 1.5], [-0.135, -0.015], transform=ax.get_xaxis_transform(),
        color="#999999", linewidth=0.5, clip_on=False)

ax.set_xlim(-0.55, len(CONFIGS) - 1 + 0.75)   # headroom for the outside labels
ax.set_ylim(0, bottoms.max() * 1.14)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)

handles, labels = ax.get_legend_handles_labels()
handles, labels = handles[::-1], labels[::-1]         
ncol = 2 if len(TIERS) == 4 else 3
if len(TIERS) == 4:
    order = [0, 2, 1, 3]                             
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
            x0 = x[j] + BAR_W / 2.0 - 0.04
            x1 = x0 + 0.17
            ax.plot([x0, x1], [cy, ly], color="#555555", linewidth=0.5, zorder=5)
            ax.text(x1 + 0.05, ly, str(n), ha="left", va="center",
                    fontsize=FS_OUT, color="black", zorder=6,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.0))

out_dir = os.path.dirname(OUT)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
fig.savefig(OUT, dpi=300)
fig.savefig(os.path.splitext(OUT)[0] + ".pdf")
print("saved", OUT, "and", os.path.splitext(OUT)[0] + ".pdf")

ARCHIVED = {
    "VisNet Scratch":    (-0.2974, 33.1, 2.252),
    "RMEP Scratch":      (-0.3406, 33.6, 2.271),
    "VisNet Fine-tuned": (0.1932, 40.7, 1.653),
    "RMEP Fine-tuned":   (0.2986, 43.0, 1.515),
}

print(f"\n{'configuration':<18}" + "".join(f"{t[5]:>10}" for t in TIERS)
      + f"{'sites':>7}{'OVERALL R2':>12}{'Acc':>7}{'MAE':>7}   identity")
for j, (arch, regime, _) in enumerate(CONFIGS):
    name = f"{arch} {regime}"
    row = (f"{name:<18}" + "".join(f"{counts[i, j]:>10d}" for i in range(len(TIERS)))
           + f"{len(data[j]):>7d}")
    if idents[j] is None:
        print(row + f"{'-':>12}{'-':>7}{'-':>7}   no OVERALL row")
        continue
    r2, acc, mae = idents[j]
    expected = ARCHIVED.get(name)
    if expected is None:
        tag = ""
    elif (abs(r2 - expected[0]) < 0.01 and abs(acc - expected[1]) < 0.2
          and abs(mae - expected[2]) < 0.01):
        tag = "matches archive"
    else:
        tag = f"MISMATCH (archive: R2={expected[0]:.4f}, {expected[1]:.1f}%, MAE={expected[2]:.3f})"
    print(row + f"{r2:>12.4f}{acc:>7.1f}{mae:>7.3f}   {tag}")
