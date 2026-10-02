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
SCRATCH = "VisNetPerSiteAndPredictionsResults"
CMAP = "viridis"   # set to "Greys" for a black-and-white build
OUT = "fig4_logmae_vs_classes"


def resolve(folder, name):
    direct = os.path.join(BASE, folder, name)
    if os.path.exists(direct):
        return direct
    for entry in os.listdir(BASE):
        if entry.lower() == folder.lower():
            return os.path.join(BASE, entry, name)
    raise FileNotFoundError(f"{folder}/{name} not found")


keep = {r["site_id"] for r in csv.DictReader(open(resolve(SCRATCH, "persite_results.csv"), newline=""))
        if r["site_id"] != "OVERALL"}
err, classes = defaultdict(list), defaultdict(set)
for p in csv.DictReader(open(resolve(REF, "predictions.csv"), newline="")):
    if p["site_id"] not in keep:
        continue
    t, q = float(p["true_vis"]), float(p["pred_vis"])
    err[p["site_id"]].append(abs(np.log10(t) - np.log10(q)))
    classes[p["site_id"]].add(t)

site = sorted(err)
log_mae = np.array([float(np.mean(err[s])) for s in site])
n_cls = np.array([len(classes[s]) for s in site], float)

print(f"{len(site)} sites   r(n_classes, log-MAE) = {np.corrcoef(n_cls, log_mae)[0,1]:+.2f}\n")
ks = sorted({int(k) for k in n_cls})
data = [log_mae[n_cls == k] for k in ks]
print(f"{'classes':>8}{'sites':>7}{'median':>9}{'IQR':>16}")
for k, d in zip(ks, data):
    print(f"{k:>8}{len(d):>7}{np.median(d):>9.3f}"
          f"   {np.percentile(d,25):.3f}--{np.percentile(d,75):.3f}")

fig, ax = plt.subplots(figsize=(3.39, 2.35))
fig.subplots_adjust(left=0.165, right=0.99, top=0.975, bottom=0.185)

bp = ax.boxplot(data, positions=ks, widths=0.62, showfliers=False,
                patch_artist=True, medianprops=dict(color="black", linewidth=1.1),
                boxprops=dict(edgecolor="#404040", linewidth=0.6),
                whiskerprops=dict(color="#404040", linewidth=0.6),
                capprops=dict(color="#404040", linewidth=0.6))

# Same viridis ramp Fig. 3 uses for visibility classes, so the colour means
# the same thing in both figures.
cmap = plt.get_cmap(CMAP)
norm = mpl.colors.Normalize(vmin=1, vmax=10)
for patch, k in zip(bp["boxes"], ks):
    patch.set_facecolor(cmap(norm(k)))
    patch.set_alpha(0.85)

rng = np.random.default_rng(0)
for k, d in zip(ks, data):
    ax.scatter(k + rng.uniform(-0.2, 0.2, len(d)), d, s=6, color="#111111",
               alpha=0.45, linewidth=0, zorder=3)

ax.set_xlabel("Visibility classes observed at the site")
ax.set_ylabel("Per-site log-MAE")
ax.set_xticks(ks)
ax.set_xticklabels([str(k) for k in ks])
ax.set_xlim(min(ks) - 0.7, max(ks) + 0.7)
ax.set_ylim(-0.02, log_mae.max() * 1.06)
ax.set_axisbelow(True)
ax.yaxis.grid(True, color="#dcdcdc", linewidth=0.4)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

fig.savefig(f"{OUT}.png", dpi=300)
fig.savefig(f"{OUT}.pdf")
print(f"\nsaved {OUT}.png and {OUT}.pdf")
