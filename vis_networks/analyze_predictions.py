import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))

EXPERIMENTS = {
    "VisNet Global":   {
        "summary_csv":     "VisNetSiteEvaluationResults/site_specific_validation_epoch0.csv",
        "predictions_csv": None,
    },
    "VisNet Fine-tune": {
        "summary_csv":     "VisNetFineTuneAndPredictionsResults/persite_results.csv",
        "predictions_csv": "VisNetFineTuneAndPredictionsResults/predictions.csv",
    },
    "RMEP Fine-tune":  {
        "summary_csv":     "RMEPFineTuneAndPredictionsResults/persite_results.csv",
        "predictions_csv": "RMEPFineTuneAndPredictionsResults/predictions.csv",
    },
}

BANDS = [
    ("Short  (< 3 mi)",  0.0, 3.0),
    ("Medium (3–6 mi)",  3.0, 6.0),
    ("Far    (> 6 mi)",  6.0, 99.0),
]


def rmse(true, pred):
    return np.sqrt(np.mean((np.array(true) - np.array(pred)) ** 2))

def mae(true, pred):
    return np.mean(np.abs(np.array(true) - np.array(pred)))

def mape(true, pred):
    t, p = np.array(true), np.array(pred)
    mask = t > 0
    return np.mean(np.abs(t[mask] - p[mask]) / t[mask]) * 100

def load_predictions(rel_path):
    path = os.path.join(BASE, rel_path)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    required = {"site_id", "true_vis", "pred_vis"}
    if not required.issubset(df.columns):
        print(f"  WARNING: {rel_path} missing columns {required - set(df.columns)}")
        return None
    return df

def load_summary(rel_path):
    path = os.path.join(BASE, rel_path)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def compute_range_metrics(df):
    rows = []
    for label, lo, hi in BANDS:
        mask = (df["true_vis"] >= lo) & (df["true_vis"] < hi)
        sub = df[mask]
        if len(sub) < 2:
            rows.append({"band": label, "n": len(sub),
                         "RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan})
            continue
        rows.append({
            "band":  label,
            "n":     len(sub),
            "RMSE":  rmse(sub["true_vis"], sub["pred_vis"]),
            "MAE":   mae(sub["true_vis"],  sub["pred_vis"]),
            "MAPE":  mape(sub["true_vis"], sub["pred_vis"]),
        })
    rows.append({
        "band":  "Overall",
        "n":     len(df),
        "RMSE":  rmse(df["true_vis"], df["pred_vis"]),
        "MAE":   mae(df["true_vis"],  df["pred_vis"]),
        "MAPE":  mape(df["true_vis"], df["pred_vis"]),
    })
    return pd.DataFrame(rows)


def plot_range_breakdown(exp_metrics: dict, save):
    band_labels = [b[0] for b in BANDS] + ["Overall"]
    metrics = ["RMSE", "MAE", "MAPE"]
    ylabels = ["RMSE (miles)", "MAE (miles)", "MAPE (%)"]

    n_exps = len(exp_metrics)
    x = np.arange(len(band_labels))
    width = 0.8 / n_exps

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Error by Visibility Range", fontsize=13, fontweight="bold")
    colors = plt.cm.Set2(np.linspace(0, 1, n_exps))

    for ax_idx, (metric, ylabel) in enumerate(zip(metrics, ylabels)):
        ax = axes[ax_idx]
        for j, (exp_name, mdf) in enumerate(exp_metrics.items()):
            offsets = x + (j - n_exps / 2 + 0.5) * width
            vals = mdf.set_index("band").reindex(band_labels)[metric].values
            bars = ax.bar(offsets, vals, width=width * 0.9,
                          label=exp_name, color=colors[j], alpha=0.85,
                          edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, vals):
                if not np.isnan(val) and val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.02,
                            f"{val:.2f}", ha="center", va="bottom",
                            fontsize=6.5, rotation=0)
        ax.set_xticks(x)
        ax.set_xticklabels(band_labels, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(metric, fontsize=11, fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if save:
        p = os.path.join(BASE, "range_breakdown.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved → {p}")
    else:
        plt.show()
    plt.close()


def print_range_table(exp_metrics):
    print("\n" + "=" * 80)
    print("ERROR BY VISIBILITY RANGE")
    print("=" * 80)
    for exp_name, mdf in exp_metrics.items():
        print(f"\n  {exp_name}")
        print(f"  {'Band':<22} {'N':>6} {'RMSE':>8} {'MAE':>8} {'MAPE %':>8}")
        print("  " + "-" * 56)
        for _, row in mdf.iterrows():
            rmse_s = f"{row['RMSE']:.3f}" if not np.isnan(row['RMSE']) else "  n/a"
            mae_s  = f"{row['MAE']:.3f}"  if not np.isnan(row['MAE'])  else "  n/a"
            mape_s = f"{row['MAPE']:.1f}" if not np.isnan(row['MAPE']) else "  n/a"
            print(f"  {row['band']:<22} {int(row['n']):>6} {rmse_s:>8} {mae_s:>8} {mape_s:>8}")


def select_case_study_sites(finetune_summary, global_summary, n=6):
    ft = finetune_summary[finetune_summary["site_id"] != "OVERALL"].copy()
    gl = global_summary[global_summary["site_id"] != "OVERALL"].copy()

    gl_renamed = gl[["site_id", "MAE_visibility", "R2"]].rename(
        columns={"MAE_visibility": "global_MAE", "R2": "global_R2"}
    )
    merged = ft.merge(gl_renamed, on="site_id", how="inner")
    merged["delta_MAE"] = merged["MAE_visibility"] - merged["global_MAE"]
    merged["n_train_col"] = merged.get("n_train", merged.get("n_samples", 0))

    selected = {}

    veia = merged[merged["R2"] > 0.89].nsmallest(1, "MAE_visibility")
    if len(veia):
        selected["Beats VEIA (R²>0.89)"] = veia.iloc[0]["site_id"]

    best_imp = merged.nsmallest(1, "delta_MAE")
    selected["Biggest improvement"] = best_imp.iloc[0]["site_id"]

    high_data = merged[merged["n_train_col"] >= 60].nsmallest(1, "MAE_visibility")
    if len(high_data):
        selected["High data + good result"] = high_data.iloc[0]["site_id"]

    low_data = merged[merged["n_train_col"] < 20].nlargest(1, "delta_MAE")
    if len(low_data):
        selected["Low data (< 20 train)"] = low_data.iloc[0]["site_id"]

    poor = merged[merged["R2"] < 0].nlargest(1, "MAE_visibility")
    if len(poor):
        selected["Still poor (R²<0)"] = poor.iloc[0]["site_id"]

    medium = merged[
        (merged["n_train_col"] >= 30) & (merged["n_train_col"] < 60)
        & (~merged["site_id"].isin(selected.values()))
    ].nsmallest(1, "delta_MAE")
    if len(medium):
        selected["Medium data"] = medium.iloc[0]["site_id"]

    seen = set()
    deduped = {}
    for label, sid in selected.items():
        if sid not in seen:
            deduped[label] = sid
            seen.add(sid)
    return deduped


def plot_case_studies(site_selection, ft_preds, global_preds, ft_summary, save):
    sites = list(site_selection.items())
    n = len(sites)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4.5))
    fig.suptitle("Case Study Sites — Predicted vs True Visibility",
                 fontsize=13, fontweight="bold")
    axes = axes.flatten()
    vis_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    for idx, (label, site_id) in enumerate(sites):
        ax = axes[idx]
        if ft_preds is not None and site_id in ft_preds["site_id"].values:
            sub_ft = ft_preds[ft_preds["site_id"] == site_id]
        else:
            sub_ft = None

        if sub_ft is not None and len(sub_ft) > 0:
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(sub_ft))
            ax.scatter(
                sub_ft["true_vis"] + jitter,
                sub_ft["pred_vis"] + jitter,
                alpha=0.6, s=35, color="#70ad47", edgecolors="white",
                linewidth=0.4, label="Fine-tune", zorder=3
            )
            ft_mae_val  = mae(sub_ft["true_vis"], sub_ft["pred_vis"])
            ft_mape_val = mape(sub_ft["true_vis"], sub_ft["pred_vis"])

        lo, hi = 0.5, 10.5
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.5, zorder=2)
        ax.fill_between([lo, hi], [lo - 2, hi - 2], [lo + 2, hi + 2],
                        alpha=0.08, color="gray", label="±2 mi band")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xticks(vis_values)
        ax.set_yticks(vis_values)
        ax.set_xticklabels([str(int(v)) for v in vis_values], fontsize=7)
        ax.set_yticklabels([str(int(v)) for v in vis_values], fontsize=7)
        ax.set_xlabel("True visibility (mi)", fontsize=8)
        ax.set_ylabel("Predicted visibility (mi)", fontsize=8)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

        if ft_summary is not None:
            site_row = ft_summary[ft_summary["site_id"] == site_id]
            n_train = int(site_row["n_train"].values[0]) if len(site_row) else "?"
            ft_r2   = float(site_row["R2"].values[0])   if len(site_row) else float("nan")
        else:
            n_train, ft_r2 = "?", float("nan")

        title_lines = [f"{site_id}  —  {label}",
                       f"n_train={n_train}  |  FT R²={ft_r2:.3f}"]
        if sub_ft is not None and len(sub_ft) > 0:
            title_lines.append(f"FT MAE={ft_mae_val:.2f} mi  |  MAPE={ft_mape_val:.1f}%")
        ax.set_title("\n".join(title_lines), fontsize=8, fontweight="bold")
        ax.legend(fontsize=7, loc="upper left")

    for idx in range(len(sites), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    if save:
        p = os.path.join(BASE, "case_studies.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved → {p}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    exp_preds = {}
    exp_summaries = {}
    missing = []

    for name, paths in EXPERIMENTS.items():
        summary = load_summary(paths["summary_csv"])
        exp_summaries[name] = summary
        if paths["predictions_csv"] is None:
            missing.append(name)
            continue
        preds = load_predictions(paths["predictions_csv"])
        if preds is None:
            missing.append(name)
        else:
            exp_preds[name] = preds
            print(f"  Loaded {len(preds):,} predictions for '{name}'")

    if missing:
        print(f"\n  NOTE: No predictions.csv found for: {', '.join(missing)}")
        print("  → Re-run trainpersite.py with the modified version to generate them.\n")

    if not exp_preds:
        print("\nNo prediction files found. Please re-run trainpersite.py first.")
        return

    print("\nComputing range breakdown metrics...")
    exp_metrics = {name: compute_range_metrics(preds) for name, preds in exp_preds.items()}
    print_range_table(exp_metrics)
    plot_range_breakdown(exp_metrics, save=args.save)

    ft_name = "VisNet Fine-tune"
    gl_name = "VisNet Global"

    if ft_name in exp_preds and gl_name in exp_summaries:
        print("\nSelecting case study sites...")
        ft_summary = load_summary(EXPERIMENTS[ft_name]["summary_csv"])
        ft_summary = ft_summary[ft_summary["site_id"] != "OVERALL"]
        gl_summary = exp_summaries[gl_name]
        gl_summary = gl_summary[gl_summary["site_id"] != "OVERALL"]

        site_sel = select_case_study_sites(ft_summary, gl_summary)
        print("  Selected sites:")
        for label, sid in site_sel.items():
            print(f"    {label:<30} → {sid}")

        plot_case_studies(
            site_sel,
            ft_preds=exp_preds[ft_name],
            global_preds=exp_preds.get(gl_name),
            ft_summary=ft_summary,
            save=args.save,
        )
    else:
        print(f"\nSkipping case studies — need '{ft_name}' predictions.")


if __name__ == "__main__":
    main()