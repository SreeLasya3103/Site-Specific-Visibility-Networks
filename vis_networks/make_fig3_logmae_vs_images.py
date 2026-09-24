import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "mathtext.fontset": "stix",
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.labelsize": 9,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
})

BASE = "."
REF = "RMEPFineTuneAndPredictionsResults"
# The two sites that failed to converge under VisNet are excluded from every
# configuration, so this is drawn on the same 253 sites as Fig. 2.
SCRATCH = "VisNetPerSiteAndPredictionsResults"
CMAP = "viridis"
OUT = "fig3_logmae_vs_images"


def resolve(folder, name):
    direct = os.path.join(BASE, folder, name)
    if os.path.exists(direct):
        return direct
    for entry in os.listdir(BASE):
        if entry.lower() == folder.lower():
            return os.path.join(BASE, entry, name)
    raise FileNotFoundError(f"{folder}/{name} not found")


rows = [r for r in csv.DictReader(open(resolve(REF, "persite_results.csv"), newline=""))
        if r["site_id"] != "OVERALL"]
keep = {r["site_id"] for r in csv.DictReader(open(resolve(SCRATCH, "persite_results.csv"), newline=""))
        if r["site_id"] != "OVERALL"}
rows = [r for r in rows if r["site_id"] in keep]

err, classes = defaultdict(list), defaultdict(set)
for p in csv.DictReader(open(resolve(REF, "predictions.csv"), newline="")):
    t, q = float(p["true_vis"]), float(p["pred_vis"])
    err[p["site_id"]].append(abs(np.log10(t) - np.log10(q)))
    classes[p["site_id"]].add(t)

site = [r["site_id"] for r in rows]
n_train = np.array([int(r["n_train"]) for r in rows], float)
log_mae = np.array([float(np.mean(err[s])) for s in site])
n_cls = np.array([len(classes[s]) for s in site], float)

corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])
print(f"{len(rows)} sites")
print(f"r(n_train, log-MAE)   = {corr(n_train, log_mae):+.2f}")
print(f"r(n_train, n_classes) = {corr(n_train, n_cls):+.2f}")
print(f"r(n_classes, log-MAE) = {corr(n_cls, log_mae):+.2f}")
for k in (2, 8):
    m = n_cls == k
    print(f"{k} classes (n={int(m.sum()):>2}): mean log-MAE {log_mae[m].mean():.3f}")
m = n_train == 20
print(f"exactly 20 training images (n={int(m.sum())}): "
      f"log-MAE {log_mae[m].min():.3f}--{log_mae[m].max():.3f}")

fig, ax = plt.subplots(figsize=(3.39, 2.55))
fig.subplots_adjust(left=0.165, right=0.86, top=0.965, bottom=0.185)

sc = ax.scatter(n_train, log_mae, c=n_cls, cmap=CMAP, vmin=1, vmax=10,
                s=17, linewidth=0.45, edgecolor="#303030", zorder=3)

b, a = np.polyfit(n_train, log_mae, 1)
xs = np.array([n_train.min(), n_train.max()])
ax.plot(xs, b * xs + a, color="#333333", linewidth=0.9, linestyle="--", zorder=4)

ax.set_xlabel("Training images at the site")
ax.set_ylabel("Per-site log-MAE")
ax.set_xlim(0, n_train.max() * 1.04)
ax.set_ylim(-0.02, log_mae.max() * 1.06)
ax.set_axisbelow(True)
ax.yaxis.grid(True, color="#dcdcdc", linewidth=0.4)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

cb = fig.colorbar(sc, ax=ax, pad=0.03, fraction=0.055, ticks=[2, 4, 6, 8, 10])
cb.set_label("Visibility classes", fontsize=8, labelpad=2)
cb.ax.tick_params(labelsize=8, length=2)
cb.outline.set_linewidth(0.5)

fig.savefig(f"{OUT}.png", dpi=300)
fig.savefig(f"{OUT}.pdf")
print(f"saved {OUT}.png and {OUT}.pdf")
