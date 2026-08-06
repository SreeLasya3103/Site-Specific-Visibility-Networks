import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))

GLOBAL_CSV   = os.path.join(BASE, "VisNetSiteEvaluationResults/site_specific_validation_epoch0.csv")
FINETUNE_CSV = os.path.join(BASE, "VisNetFineTuneAndPredictionsResults/persite_results.csv")

BUCKETS = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 999)]
BUCKET_LABELS = ["< 20", "20–40", "40–60", "60–80", "80+"]


def load_and_merge():
    g = pd.read_csv(GLOBAL_CSV)
    g = g[g["site_id"] != "OVERALL"][["site_id", "R2", "MAE_visibility", "RMSE_visibility"]]
    g.columns = ["site_id", "global_R2", "global_MAE", "global_RMSE"]

    f = pd.read_csv(FINETUNE_CSV)
    f = f[f["site_id"] != "OVERALL"][
        ["site_id", "n_total", "n_train", "n_test", "R2", "MAE_visibility", "RMSE_visibility"]
    ]
    f.columns = ["site_id", "n_total", "n_train", "n_test", "ft_R2", "ft_MAE", "ft_RMSE"]

    df = pd.merge(f, g, on="site_id", how="inner")
    df["delta_MAE"]  = df["ft_MAE"]  - df["global_MAE"]
    df["delta_RMSE"] = df["ft_RMSE"] - df["global_RMSE"]
    df["delta_R2"]   = df["ft_R2"]   - df["global_R2"]
    return df


def bucket_label(n):
    for lo, hi, label in zip(
        [b[0] for b in BUCKETS],
        [b[1] for b in BUCKETS],
        BUCKET_LABELS
    ):
        if lo <= n < hi:
            return label
    return BUCKET_LABELS[-1]


def plot_all(df, save):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Fine-Tune Improvement vs Training Sample Count\n(VisNet Fine-tune vs VisNet Global)",
        fontsize=13, fontweight="bold"
    )

    ax = axes[0, 0]
    ax.scatter(df["n_train"], df["delta_MAE"],
               alpha=0.55, edgecolors="white", linewidth=0.4,
               c=df["delta_MAE"], cmap="RdYlGn_r", s=40)
    ax.axhline(0, color="black", linewidth=1, linestyle="--", label="No change")
    ax.set_xlabel("Training samples per site", fontsize=10)
    ax.set_ylabel("ΔMAE  (fine-tune − global, miles)", fontsize=10)
    ax.set_title("VisNet Global vs Fine-Tune MAE per Site", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    n_improved = (df["delta_MAE"] < 0).sum()
    ax.text(0.97, 0.05, f"{n_improved}/{len(df)} sites improved",
            transform=ax.transAxes, ha="right", fontsize=8, color="darkgreen")

    ax = axes[0, 1]
    ax.scatter(df["n_train"], df["delta_RMSE"],
               alpha=0.55, edgecolors="white", linewidth=0.4,
               c=df["delta_RMSE"], cmap="RdYlGn_r", s=40)
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Training samples per site", fontsize=10)
    ax.set_ylabel("ΔRMSE  (fine-tune − global, miles)", fontsize=10)
    ax.set_title("RMSE Improvement vs Sample Count", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1, 0]
    norm = plt.Normalize(df["n_train"].min(), df["n_train"].max())
    sc = ax.scatter(df["global_MAE"], df["ft_MAE"],
                    c=df["n_train"], cmap="viridis", alpha=0.6,
                    edgecolors="white", linewidth=0.4, s=45, norm=norm)
    lo = min(df["global_MAE"].min(), df["ft_MAE"].min()) - 0.1
    hi = max(df["global_MAE"].max(), df["ft_MAE"].max()) + 0.1
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="No change")
    plt.colorbar(sc, ax=ax, label="n_train")
    ax.set_xlabel("Global model MAE (miles)", fontsize=10)
    ax.set_ylabel("Fine-tune MAE (miles)", fontsize=10)
    ax.set_title("Global vs Fine-Tune MAE per Site", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1, 1]
    df["bucket"] = df["n_train"].apply(bucket_label)
    groups = [df[df["bucket"] == lbl]["delta_MAE"].values for lbl in BUCKET_LABELS]
    valid_labels = [lbl for lbl, g in zip(BUCKET_LABELS, groups) if len(g) > 0]
    groups = [g for g in groups if len(g) > 0]

    bp = ax.boxplot(groups, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2))
    colors_box = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(valid_labels)))
    for patch, color in zip(bp["boxes"], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xticks(range(1, len(valid_labels) + 1))
    ax.set_xticklabels(valid_labels, fontsize=9)
    ax.set_xlabel("Training samples per site", fontsize=10)
    ax.set_ylabel("ΔMAE  (fine-tune − global, miles)", fontsize=10)
    ax.set_title("MAE Improvement by Sample Count Bucket", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    counts = df.groupby("bucket").size().reindex(valid_labels, fill_value=0)
    for i, (lbl, count) in enumerate(zip(valid_labels, counts)):
        ax.text(i + 1, ax.get_ylim()[0] + 0.05, f"n={count}",
                ha="center", va="bottom", fontsize=7, color="gray")

    plt.tight_layout()
    if save:
        p = os.path.join(BASE, "improvement_vs_samples.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved → {p}")
    else:
        plt.show()
    plt.close()


def print_threshold_table(df):
    df["bucket"] = df["n_train"].apply(bucket_label)
    print("\nMean MAE improvement (negative = better) by training sample count:")
    print(f"{'Bucket':<12} {'Sites':>6} {'Mean ΔMAE':>12} {'% improved':>12}")
    print("-" * 46)
    for lbl in BUCKET_LABELS:
        sub = df[df["bucket"] == lbl]
        if len(sub) == 0:
            continue
        mean_delta = sub["delta_MAE"].mean()
        pct = (sub["delta_MAE"] < 0).mean() * 100
        print(f"{lbl:<12} {len(sub):>6} {mean_delta:>12.3f} {pct:>11.0f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    df = load_and_merge()
    print(f"Loaded {len(df)} sites with both global and fine-tune results.")
    print_threshold_table(df)
    plot_all(df, save=args.save)


if __name__ == "__main__":
    main()