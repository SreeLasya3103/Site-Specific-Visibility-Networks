#score_rmep_similarity.py - RMEP-Similarity on the thesis evaluation set, without changing
#any existing table: the evaluation set is still built from the original nine
#configurations, and the new comparisons get their own Holm correction.
import argparse
import os

from make_thesis_tables import logmae, mae
from thesis_common import (arrays, label, leakage_free, load_all, load_predictions, metrics,
                           paired_bootstrap, site_of)

NEW = [
    ("RMEP-Similarity", "Pooled", "RMEPSimilarityGlobalOnPerSiteTest"),
    ("RMEP-Similarity", "Per-site heads", "RMEPSimilarityPerSiteAndPredictionsResults"),
    ("RMEP-Similarity", "Fine-tuned", "RMEPSimilarityFineTuneAndPredictionsResults"),
]

COMPARISONS = [
    ("Adaptation: pooled - fine-tuned", ("RMEP-Similarity", "Pooled"), ("RMEP-Similarity", "Fine-tuned")),
    ("Initialization: per-site heads - fine-tuned", ("RMEP-Similarity", "Per-site heads"), ("RMEP-Similarity", "Fine-tuned")),
    ("Backbone, pooled: RMEP-Sim - VisNet-Sim", ("RMEP-Similarity", "Pooled"), ("VisNet-Similarity", "Pooled")),
    ("Backbone, fine-tuned: RMEP-Sim - VisNet-Sim", ("RMEP-Similarity", "Fine-tuned"), ("VisNet-Similarity", "Fine-tuned")),
    ("Reference, pooled: RMEP-Sim - RMEP", ("RMEP-Similarity", "Pooled"), ("RMEP", "Pooled")),
    ("Reference, fine-tuned: RMEP-Sim - RMEP", ("RMEP-Similarity", "Fine-tuned"), ("RMEP", "Fine-tuned")),
]


def holm(pvals, alpha=0.05):
    out, stop = {}, False
    for rank, (i, p) in enumerate(sorted(enumerate(pvals), key=lambda z: z[1])):
        ok = not stop and p <= alpha / (len(pvals) - rank)
        stop = stop or not ok
        out[i] = ok
    return [out[i] for i in range(len(pvals))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".", help="the vis_networks folder (default: the current folder)")
    ap.add_argument("--out", default="rmep_similarity_numbers.txt")
    args = ap.parse_args()

    lines = []
    def note(s):
        print(s)
        lines.append(s)

    # The evaluation set exactly as in Table 6.1 (from the original nine configurations).
    preds, common = load_all(args.repo)
    lf = leakage_free(args.repo, common)
    note(f"evaluation set: {len(lf)} images, {len({site_of(p) for p in lf})} sites")

    for m, r, folder in NEW:
        d = load_predictions(args.repo, folder)
        missing = [p for p in lf if p not in d]
        if missing:
            raise SystemExit(f"ERROR: {folder} has no prediction for {len(missing)} evaluation "
                             f"image(s), e.g. {missing[:3]}. The runs must cover the same images "
                             f"as VisNet-Similarity.")
        preds[label(m, r)] = d

    A = {name: arrays(preds, name, lf) for name in preds}
    t = A[label("RMEP", "Fine-tuned")][0]
    for name in A:   # every configuration must agree on the true labels
        if not (A[name][0] == t).all():
            raise SystemExit(f"ERROR: true labels differ for {name}")

    for m, r, _ in NEW:
        M = metrics(*A[label(m, r)])
        note(f"{label(m, r)}: acc {M['acc']:.1f} w2 {M['within2']:.1f} w1 {M['within1']:.1f} "
             f"MAE {M['mae']:.3f} RMSE {M['rmse']:.3f} logMAE {M['logmae']:.3f}")

    res = []
    for name, a, b in COMPARISONS:
        pa, pb = A[label(*a)][1], A[label(*b)][1]
        # A two-sided p above 1 only happens when every resample gives the same difference.
        r1, r2 = (paired_bootstrap(t, pa, pb, fn) for fn in (mae, logmae))
        res.append((name, r1[:3] + (min(r1[3], 1.0),), r2[:3] + (min(r2[3], 1.0),)))
    sig = holm([r1[3] for _, r1, _ in res])
    note(f"paired bootstrap (10,000 resamples, seed 42); Holm over these {len(res)} MAE comparisons only")
    for (name, r1, r2), s in zip(res, sig):
        note(f"{name}: dMAE {r1[0]:+.3f} [{r1[1]:+.3f}, {r1[2]:+.3f}] p={r1[3]:.4f}{' *' if s else ''}; "
             f"dlog {r2[0]:+.4f} [{r2[1]:+.4f}, {r2[2]:+.4f}] p={r2[3]:.4f}")

    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
