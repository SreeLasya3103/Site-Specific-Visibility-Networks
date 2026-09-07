import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE = os.path.dirname(os.path.abspath(__file__))

EXPERIMENTS = [
    {
        "label": "VisNet\nGlobal",
        "short": "VisNet Global",
        "csv":   "VisNetSiteEvaluationResults/site_specific_validation_epoch0.csv",
        "type":  "global",
    },
    {
        "label": "RMEP\nGlobal",
        "short": "RMEP Global",
        "csv":   "RMEPSiteEvaluationResults/site_specific_validation_epoch0.csv",
        "type":  "global",
    },
    {
        "label": "VisNet\nScratch",
        "short": "VisNet Scratch",
        "csv":   "VisNetPersiteAndPredictionsResults/persite_results.csv",
        "type":  "scratch",
    },
    {
        "label": "RMEP\nScratch",
        "short": "RMEP Scratch",
        "csv":   "RMEPPerSiteAndPredictionsResults/persite_results.csv",
        "type":  "scratch",
    },
    {
        "label": "VisNet\nFine-tune",
        "short": "VisNet Fine-tune",
        "csv":   "VisNetFineTuneAndPredictionsResults/persite_results.csv",
        "type":  "finetune",
    },
    {
        "label": "RMEP\nFine-tune",
        "short": "RMEP Fine-tune",
        "csv":   "RMEPFineTuneAndPredictionsResults/persite_results.csv",
        "type":  "finetune",
    },
    {
        "label": "VisNet\nSimilarity",
        "short": "VisNet-Similarity",
        "csv":   "VisNetSimilarityPersiteFineTuneResults/persite_results.csv",
        "type":  "similarity",
    },
]

VEIA_R2 = 0.89

TYPE_COLORS = {
    "global":     "#5b9bd5",
    "scratch":    "#ed7d31",
    "finetune":   "#70ad47",
    "similarity": "#9e5db3",
}


def load_csv(rel_path):
    path = os.path.join(BASE, rel_path)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def overall_row(df):
    rows = df[df["site_id"] == "OVERALL"]
    return rows.iloc[0] if len(rows) else None


def per_site_r2(df):
    sub = df[df["site_id"] != "OVERALL"].copy()
    if "R2" not in sub.columns:
        return np.array([])
    return sub["R2"].dropna().values


def resolve_experiments(experiments):
    resolved = []
    for exp in experiments:
        df = load_csv(exp["csv"])
        if df is None:
            print(f"  WARNING: CSV not found — {exp['csv']}")
            continue
        row = overall_row(df)
        if row is None:
            print(f"  WARNING: No OVERALL row — {exp['csv']}")
            continue
        n_test = int(row.get("n_samples", row.get("n_test", 0)))
        resolved.append({
            **exp,
            "n_test": n_test,
            "r2":        float(row["R2"]),
            "acc":       float(row["accuracy"]),
            "acc2":      float(row["accuracy_within_2"]),
            "mae":       float(row["MAE_visibility"]),
            "r2_values": per_site_r2(df),
        })
    return resolved


def print_table(exps):
    sep = "-" * 82
    print(f"\n{sep}")
    print(f"{'Experiment':<22} {'Type':<12} {'R²':>7} {'Acc':>8} {'±2 Acc':>8} {'MAE':>7} {'N test':>8}")
    print(sep)
    for e in exps:
        print(
            f"{e['short']:<22} {e['type']:<12} "
            f"{e['r2']:>7.3f} "
            f"{e['acc']*100:>7.1f}% "
            f"{e['acc2']*100:>7.1f}% "
            f"{e['mae']:>7.3f} "
            f"{e['n_test']:>8,}"
        )
    print(sep)
    print(f"  {'VEIA (reference)':<21} {'traditional':<12} {VEIA_R2:>7.3f}{'—':>9}{'—':>9}{'—':>8}{'—':>9}")
    print(f"{sep}\n")


def plot_metrics(exps, save):
    labels = [e["label"] for e in exps]
    colors = [TYPE_COLORS[e["type"]] for e in exps]
    x = np.arange(len(exps))

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle("Experiment Comparison — All Metrics", fontsize=14, fontweight="bold")

    panels = [
        (axes[0, 0], [e["r2"]   for e in exps], "R²",            True),
        (axes[0, 1], [e["acc"]  for e in exps], "Exact Accuracy", False),
        (axes[1, 0], [e["acc2"] for e in exps], "±2 Accuracy",   False),
        (axes[1, 1], [e["mae"]  for e in exps], "MAE (miles)",   False),
    ]

    for ax, vals, title, show_veia in panels:
        bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.6, zorder=3)
        for bar, val in zip(bars, vals):
            offset = 0.01 if val >= 0 else -0.06
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=7, fontweight="bold",
            )
        if show_veia:
            ax.axhline(VEIA_R2, color="red", linestyle="--", linewidth=1.4,
                       label=f"VEIA  R²={VEIA_R2}", zorder=4)
            ax.axhline(0, color="gray", linestyle=":", linewidth=0.8, zorder=2)
            ax.legend(fontsize=8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

    legend_patches = [
        mpatches.Patch(color=c, label=t.capitalize())
        for t, c in TYPE_COLORS.items()
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4,
               title="Experiment type", fontsize=9, title_fontsize=9,
               bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    if save:
        p = os.path.join(BASE, "experiment_metrics.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved → {p}")
    else:
        plt.show()
    plt.close()


def plot_r2_distribution(exps, save):
    tiers = [
        (0.89,  float("inf"), "VEIA-level  (R²>0.89)", "#2ecc71"),
        (0.50,  0.89,         "Good        (0.5–0.89)", "#82e0aa"),
        (0.00,  0.50,         "Fair        (0–0.5)",    "#f9e79f"),
        (float("-inf"), 0.00, "Poor        (R²<0)",     "#e74c3c"),
    ]

    eligible = [e for e in exps if len(e["r2_values"]) > 0]
    if not eligible:
        print("No per-site R² data available.")
        return

    x = np.arange(len(eligible))
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.suptitle("Per-Site R² Distribution by Experiment", fontsize=13, fontweight="bold")

    bottoms = np.zeros(len(eligible))
    for lo, hi, tier_label, color in tiers:
        pcts = []
        for e in eligible:
            vals = e["r2_values"]
            count = np.sum(vals > lo) if hi == float("inf") \
                    else np.sum((vals > lo) & (vals <= hi))
            pcts.append(count / len(vals) * 100)
        pcts = np.array(pcts)
        bars = ax.bar(x, pcts, bottom=bottoms, color=color,
                      label=tier_label, edgecolor="white", linewidth=0.5)
        for bar, pct in zip(bars, pcts):
            if pct > 4:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{pct:.0f}%",
                    ha="center", va="center", fontsize=8,
                    fontweight="bold", color="black",
                )
        bottoms += pcts

    ax.set_xticks(x)
    ax.set_xticklabels([e["short"] for e in eligible], fontsize=8.5)
    ax.set_ylabel("% of sites", fontsize=10)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", fontsize=8, title="R² tier", title_fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if save:
        p = os.path.join(BASE, "experiment_r2_distribution.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved → {p}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true",
                        help="Save plots as PNG instead of showing them")
    args = parser.parse_args()

    print("Loading experiment CSVs...")
    exps = resolve_experiments(EXPERIMENTS)
    print_table(exps)
    plot_metrics(exps, save=args.save)
    plot_r2_distribution(exps, save=args.save)


if __name__ == "__main__":
    main()