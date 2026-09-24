"""
bootstrap_ci.py

Paired bootstrap confidence intervals over the leakage-free evaluation set.

Every configuration is resampled with the SAME indices, so the differences are
paired. Images are restricted to those outside the pooled training split, which
is the set Table 1 reports; --all evaluates the full per-site held-out split
instead, which is the basis for the per-site distributions in Fig. 2.

The resampling unit is selectable. Images within a camera site are correlated,
so --unit site (resample whole sites) is the conservative choice and is what a
reviewer will ask for; --unit image is the conventional one. Report whichever
you use, and say which in the paper.

Usage, from vis_networks/:

    python bootstrap_ci.py
    python bootstrap_ci.py --unit site
    python bootstrap_ci.py --all --unit site
"""

import argparse
import csv
import os
from collections import defaultdict

import numpy as np

# Reference configuration that every other one is compared against.
REFERENCE = "RMEP Fine-tuned"

CONFIGS = [
    ("VisNet Pooled",     "VisNetGlobalOnPerSiteTest"),
    ("RMEP Pooled",       "RMEPGlobalOnPerSiteTest"),
    ("VisNet Scratch",    "VisNetPerSiteAndPredictionsResults"),
    ("RMEP Scratch",      "RMEPPerSiteAndPredictionsResults"),
    ("VisNet Fine-tuned", "VisNetFineTuneAndPredictionsResults"),
    ("RMEP Fine-tuned",   "RMEPFineTuneAndPredictionsResults"),
]

POOLED_TRAIN = os.path.join("runs", "VisNetSimilarity", "train.csv")


def resolve(folder):
    """Locate a results folder case-insensitively (they were written on Windows)."""
    direct = os.path.join(folder, "predictions.csv")
    if os.path.exists(direct):
        return direct
    for entry in os.listdir("."):
        if entry.lower() == folder.lower():
            return os.path.join(entry, "predictions.csv")
    raise FileNotFoundError(f"{folder}/predictions.csv not found")


def load(folder, keep):
    """True/predicted visibility and site id, ordered by sample path.

    Sorting by path is what makes the configurations comparable: a shared
    bootstrap index must address the same image in every configuration.
    """
    rows = [r for r in csv.DictReader(open(resolve(folder), newline=""))
            if keep(r["sample_path"])]
    rows.sort(key=lambda r: r["sample_path"])
    return (np.array([float(r["true_vis"]) for r in rows]),
            np.array([float(r["pred_vis"]) for r in rows]),
            np.array([r["site_id"] for r in rows]),
            [r["sample_path"] for r in rows])


def r2(t, p):
    denom = ((t - t.mean()) ** 2).sum()
    return np.nan if denom == 0 else 1.0 - ((t - p) ** 2).sum() / denom


def ci(a):
    return np.percentile(a, 2.5), np.percentile(a, 97.5)


def main():
    ap = argparse.ArgumentParser(description="Paired bootstrap CIs for Table 1")
    ap.add_argument("--all", action="store_true",
                    help="Use the full per-site held-out split instead of the "
                         "leakage-free subset")
    ap.add_argument("--unit", choices=("image", "site"), default="image",
                    help="Resampling unit (default: image)")
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.all:
        keep, tag = (lambda p: True), "full per-site held-out split"
    else:
        train = set(l.strip() for l in open(POOLED_TRAIN) if l.strip())
        keep, tag = (lambda p: p not in train), "outside the pooled training split"

    data = {name: load(folder, keep) for name, folder in CONFIGS}
    if REFERENCE not in data:
        raise SystemExit(f"ERROR: reference {REFERENCE!r} not among the configurations")

    paths = data[REFERENCE][3]
    n = len(paths)
    for name, (_, _, _, p) in data.items():
        if p != paths:
            raise SystemExit(
                f"ERROR: {name} does not cover the same images as {REFERENCE} "
                f"({len(p)} vs {n}) -- the bootstrap cannot be paired")

    sites = data[REFERENCE][2]
    print(f"{n} images, {len(set(sites))} sites, {tag}")
    print(f"{args.reps} replicates, resampling unit: {args.unit}\n")

    rng = np.random.default_rng(args.seed)
    if args.unit == "image":
        idx = [rng.integers(0, n, size=n) for _ in range(args.reps)]
    else:
        uniq = np.unique(sites)
        pos = {s: np.where(sites == s)[0] for s in uniq}
        idx = [np.concatenate([pos[uniq[j]]
                               for j in rng.integers(0, len(uniq), size=len(uniq))])
               for _ in range(args.reps)]

    boot = {}
    print(f"{'configuration':<18}{'R2':>8}  {'95% CI':<18}{'MAE':>7}  {'95% CI':<18}")
    for name, (t, p, _, _) in data.items():
        br2 = np.empty(args.reps)
        bmae = np.empty(args.reps)
        for b, i in enumerate(idx):
            br2[b] = r2(t[i], p[i])
            bmae[b] = np.abs(t[i] - p[i]).mean()
        boot[name] = (br2, bmae)
        lo, hi = ci(br2)
        mlo, mhi = ci(bmae)
        print(f"{name:<18}{r2(t, p):8.3f}  [{lo:6.3f}, {hi:6.3f}]  "
              f"{np.abs(t - p).mean():7.3f}  [{mlo:.3f}, {mhi:.3f}]")

    print(f"\nPaired MAE difference against {REFERENCE} "
          f"(positive = {REFERENCE} is better):")
    print(f"{'configuration':<18}{'dMAE':>8}  {'95% CI':<18}{'p':>9}")
    for name in data:
        if name == REFERENCE:
            continue
        d = boot[name][1] - boot[REFERENCE][1]
        # Two-sided percentile bootstrap: the proportion of replicates in which
        # the sign of the difference reverses. With R replicates the smallest
        # reportable value is 2/R; 0 means no replicate crossed.
        pval = 2 * min(np.mean(d <= 0), np.mean(d >= 0))
        lo, hi = ci(d)
        shown = f"{pval:.4f}" if pval > 0 else f"<{2/args.reps:.4f}"
        print(f"{name:<18}{d.mean():+8.3f}  [{lo:6.3f}, {hi:6.3f}]  {shown:>9}")
    print(f"\np is a two-sided paired percentile bootstrap, not a parametric test.")


if __name__ == "__main__":
    main()
