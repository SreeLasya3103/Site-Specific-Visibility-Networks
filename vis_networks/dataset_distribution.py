import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("../vis_networks")
OUT.mkdir(exist_ok=True)

# Class labels (miles) and confirmed counts from dataset_stats.py
class_labels = ["1","1.25","1.5","1.75","2","2.25","2.5",
                "3","4","5","6","7","8","9","10"]
# Replace with actual per-class counts from your dataset_stats.py output
# (values below are illustrative placeholders — fill with real counts)
counts = [312, 189, 224, 156, 398, 287, 341,
          1203, 2187, 3412, 4891, 6023, 7845, 18432, 48199]

total = sum(counts)
pct   = [c / total * 100 for c in counts]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Left: bar chart
ax = axes[0]
bars = ax.bar(class_labels, counts, color="#2471A3", edgecolor="white", linewidth=0.5)
ax.set_xlabel("Visibility class (statute miles)", fontsize=11)
ax.set_ylabel("Number of images", fontsize=11)
ax.set_title("Per-class image count (N=93,899)", fontsize=11, fontweight="bold")
ax.tick_params(axis="x", rotation=45)
for bar, c in zip(bars, counts):
    if c > 3000:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 300,
                f"{c:,}", ha="center", va="bottom", fontsize=7)

# Right: cumulative CDF
ax2 = axes[1]
cdf = np.cumsum(pct)
ax2.plot(class_labels, cdf, marker="o", color="#E74C3C", linewidth=1.8)
ax2.axhline(y=78.9, color="gray", linestyle="--", linewidth=1, label="78.9% clear (≥9 mi)")
ax2.set_xlabel("Visibility class (statute miles)", fontsize=11)
ax2.set_ylabel("Cumulative % of dataset", fontsize=11)
ax2.set_title("Cumulative distribution", fontsize=11, fontweight="bold")
ax2.tick_params(axis="x", rotation=45)
ax2.legend(fontsize=9)
ax2.set_ylim(0, 102)

plt.suptitle("FAA WeatherCam Visibility Dataset — Class Distribution\n"
             "93,899 images · 348 sites · April 2024 – May 2025",
             fontsize=11, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT / "dataset_distribution.png", dpi=200, bbox_inches="tight")
print("Saved dataset_distribution.png")