import argparse
import csv
import os
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
from scipy import stats

from thesis_common import (CONFIGS, arrays, collection_rows, event_of, label, leakage_free, load_all,
                           load_persite, load_predictions, metrics, paired_bootstrap,
                           persite_splits, pooled_splits, raw_label, read_list, site_of,
                           view_of, vis_class)



from decimal import Decimal, ROUND_HALF_UP


def pct(s, d="0.1"):
    """Percentage from a stored fraction string, rounded half-up (as a reader would)."""
    return str((Decimal(str(s)) * 100).quantize(Decimal(d), rounding=ROUND_HALF_UP))


def dec(s, d="0.01"):
    return str(Decimal(str(s)).quantize(Decimal(d), rounding=ROUND_HALF_UP))


def sgn(x, nd=3):
    """Signed number in math mode, with a true minus sign and no negative zero."""
    v = f"{abs(x):.{nd}f}"
    if float(v) == 0:
        return f"${v}$"
    return f"${'+' if x > 0 else '-'}{v}$"


def sgns(s, d="0.01"):
    v = Decimal(str(s)).quantize(Decimal(d), rounding=ROUND_HALF_UP)
    if v == 0:
        return f"${abs(v)}$"
    return f"${'+' if v > 0 else '-'}{abs(v)}$"


OUT = "thesis_tables"          # set from --out in main()
LOG = []


def note(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s)


def write(name, body):
    with open(os.path.join(OUT, name), "w") as f:
        f.write(body)


def fmt_p(p, reps=10000):
    return f"$<$0.001" if p < 0.001 else f"{p:.3f}"


def mae(t, p):
    return np.mean(np.abs(t - p))


def logmae(t, p):
    return np.mean(np.abs(np.log10(t) - np.log10(p)))


def dataset_tables(repo):
    sp = pooled_splits(repo)
    allp = sp["train"] + sp["validation"] + sp["test"]
    raw = Counter(raw_label(p) for p in allp)
    coll = collection_rows(repo)
    folder = Counter(float(r["visibility_mi"]) for r in coll)
    note(f"collection without nested duplicates: {len(coll)} images, {len({r['site'] for r in coll})} sites, "
         f"{len({view_of(r['path']) for r in coll})} views, "
         f"{len({(event_of(r['path']), site_of(r['path'])) for r in coll})} capture events, "
         f"{len({event_of(r['path']) for r in coll})} harvest folders")
    note("collection counts by value:", sorted(folder.items()))
    ps_all = persite_splits(repo)
    sub20 = [p for v in ps_all.values() for x in v for p in x]
    note(f"sites with >= 20 images: {len(sub20)} images, {len(ps_all)} sites, "
         f"{len({view_of(p) for p in sub20})} views, {len({(event_of(p), site_of(p)) for p in sub20})} events, "
         f"{len({event_of(p) for p in sub20})} harvest folders")

    groups = [("1", ["1", "1.25", "1.5", "1.75"], [0.0, 1.0, 1.25, 1.5, 1.75]),
              ("2", ["2", "2.25", "2.5"], [2.0, 2.25, 2.5])]
    groups += [(str(k), [str(k)], [float(k)]) for k in range(3, 10)]
    rows = []
    for c, keys, fkeys in groups:
        sub = sum(raw[k] for k in keys)
        fol = sum(folder.get(k, 0) for k in fkeys)
        if c == "1":
            vals = "1, 1\\textonequarter, 1\\textonehalf, 1\\textthreequarters"
            detail = " / ".join(f"{raw[k]:,}" for k in keys)
        elif c == "2":
            vals = "2, 2\\textonequarter, 2\\textonehalf"
            detail = " / ".join(f"{raw[k]:,}" for k in keys)
        else:
            vals, detail = c, f"{sub:,}"
        rows.append((c, vals, detail, sub, fol))
    over = sorted((k for k in raw if k not in ("10+",) and float(k) > 10), key=float)
    sub10 = raw["10+"] + sum(raw[k] for k in over)
    rows.append(("10", "10+ and reports above 10",
                 f"{raw['10+']:,} / {sum(raw[k] for k in over):,}", sub10, folder[10.0]))
    note("subset reports above 10 mi:", {k: raw[k] for k in over})
    zero = folder.get(0.0, 0)
    body = []
    for c, vals, detail, sub, fol in rows:
        fol_s = f"{fol:,}" + ("\\textsuperscript{a}" if c == "1" and zero else "")
        body.append(f"{c} & {vals} & {detail} & {fol_s} \\\\")
    body.append("\\midrule")
    body.append(f"Total & & {sum(r[3] for r in rows):,} & {sum(r[4] for r in rows):,} \\\\")
    write("t4_classes_body.tex", "\n".join(body) + "\n")
    note("class table:", [(r[0], r[3], r[4]) for r in rows])

    # structure
    sites = Counter(site_of(p) for p in allp)
    views = Counter(view_of(p) for p in allp)
    events = Counter((event_of(p), site_of(p)) for p in allp)
    v = np.array(list(sites.values()))
    vps = Counter(len([x for x in views if x.split("_")[0] == s]) for s in sites)
    note(f"subset: {len(allp)} images, {len(sites)} sites, {len(views)} views, {len(events)} events, "
         f"{len(set(event_of(p) for p in allp))} capture folders")
    note(f"images per site: median {np.median(v):.0f}, mean {v.mean():.1f}, min {v.min()}, max {v.max()}; "
         f">=20 images: {(v >= 20).sum()} sites, {v[v >= 20].sum()} images")
    note("views per site:", sorted(vps.items()))
    fe = Counter(events.values())
    note("frames per event:", sorted(fe.items()))
    where = {p: s for s, ps in sp.items() for p in ps}
    ev = defaultdict(set)
    for p in allp:
        ev[(event_of(p), site_of(p))].add(where[p])
    strad = [e for e, s in ev.items() if len(s) > 1]
    note(f"events straddling pooled splits: {len(strad)}; images {sum(events[e] for e in strad)} "
         f"({100 * sum(events[e] for e in strad) / len(allp):.1f}%)")
    cps = Counter(len({vis_class(p) for p in allp if site_of(p) == s}) for s in sites)
    note("classes per site:", sorted(cps.items()))

    def ts(p):
        f = event_of(p)
        for fm in ("%Y-%m-%d-%H-%M-%S", "%m-%d-%y-%I-%M-%p"):
            try:
                return datetime.strptime(f, fm)
            except ValueError:
                pass
    months = Counter(ts(p).strftime("%Y-%m") for p in allp)
    note("images per month:", sorted(months.items()))
    hours = Counter(ts(p).hour for p in allp)
    note("images per hour (folder time, US Eastern):", sorted(hours.items()))

    # splits table
    ps = persite_splits(repo)
    n_tr = sum(len(x[0]) for x in ps.values())
    n_va = sum(len(x[1]) for x in ps.values())
    n_te = sum(len(x[2]) for x in ps.values())
    tot = n_tr + n_va + n_te
    med = lambda i: np.median([len(x[i]) for x in ps.values()])
    note(f"per-site: {len(ps)} sites, {tot} images; train/val/test {n_tr}/{n_va}/{n_te} "
         f"= {100*n_tr/tot:.1f}/{100*n_va/tot:.1f}/{100*n_te/tot:.1f}%; medians "
         f"{med(0):.0f}/{med(1):.0f}/{med(2):.0f}; median site total "
         f"{np.median([sum(len(y) for y in x) for x in ps.values()]):.0f}")
    te_all = [p for x in ps.values() for p in x[2]]
    train = set(sp["train"])
    val = set(sp["validation"])
    note(f"per-site test images in pooled train: {sum(p in train for p in te_all)} "
         f"({100*sum(p in train for p in te_all)/len(te_all):.1f}%), pooled val "
         f"{sum(p in val for p in te_all)}, pooled test {sum(p in set(sp['test']) for p in te_all)}")
    # event integrity of the per-site split
    bad = 0
    for s, (tr, va, te) in ps.items():
        e = [{event_of(p) for p in x} for x in (tr, va, te)]
        bad += len(e[0] & e[1]) + len(e[0] & e[2]) + len(e[1] & e[2])
    note(f"per-site events shared between train/val/test: {bad}")
    return ps


#results

ROW_NAMES = {
    ("VisNet", "Pooled"): "VisNet", ("RMEP", "Pooled"): "RMEP",
    ("VisNet-Similarity", "Pooled"): "VisNet-Similarity",
    ("VisNet", "Scratch"): "VisNet", ("RMEP", "Scratch"): "RMEP",
    ("VisNet-Similarity", "Per-site heads"): "VisNet-Similarity\\textsuperscript{a}",
    ("VisNet", "Fine-tuned"): "VisNet", ("RMEP", "Fine-tuned"): "RMEP",
    ("VisNet-Similarity", "Fine-tuned"): "VisNet-Similarity",
}
REGIME_TITLES = {"Pooled": "Pooled (one model for all sites)",
                 "Scratch": "Per-site, trained from scratch",
                 "Per-site heads": None,
                 "Fine-tuned": "Per-site, fine-tuned from the pooled model"}


def main_table(repo, ps):
    preds, common = load_all(repo)
    lf = leakage_free(repo, common)
    sp = pooled_splits(repo)
    note(f"evaluation set: {len(lf)} images, {len({site_of(p) for p in lf})} sites; "
         f"pooled validation {sum(p in set(sp['validation']) for p in lf)}, pooled test "
         f"{sum(p in set(sp['test']) for p in lf)}")
    A = {label(m, r): arrays(preds, label(m, r), lf) for m, r, _ in CONFIGS}
    t = A[label("RMEP", "Fine-tuned")][0]
    body, last = [], None
    for m, r, _ in CONFIGS:
        reg = "Scratch" if r == "Per-site heads" else r
        if reg != last:
            if last is not None:
                body.append("\\midrule")
            body.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{REGIME_TITLES[reg]}}}}} \\\\")
            last = reg
        M = metrics(*A[label(m, r)])
        note(f"T6.1 {label(m, r)}: acc {M['acc']:.1f} w2 {M['within2']:.1f} w1 {M['within1']:.1f} "
             f"MAE {M['mae']:.3f} RMSE {M['rmse']:.3f} logMAE {M['logmae']:.3f} extMAE {M['extmae']:.3f}")
        body.append(f"\\quad {ROW_NAMES[(m, r)]} & {M['acc']:.1f} & {M['within2']:.1f} & "
                    f"{M['mae']:.3f} & {M['rmse']:.3f} & {M['logmae']:.3f} \\\\")
    # baselines
    maj = {}
    for s, (tr, va, te) in ps.items():
        c = Counter(vis_class(p) for p in tr + va)
        top = max(c.values())
        maj[s] = min(k for k, v in c.items() if v == top)          # ties -> lower class
    q = np.array([float(maj[site_of(p)]) for p in lf])
    M = metrics(t, q)
    note(f"T6.1 baseline majority (train+val, ties->lower): acc {M['acc']:.1f} w2 {M['within2']:.1f} "
         f"MAE {M['mae']:.3f} RMSE {M['rmse']:.3f} logMAE {M['logmae']:.3f}")
    alt = []
    for portion in ("tr", "trva"):
        for rule in ("lower", "higher", "first"):
            mj = {}
            for s, (tr, va, te) in ps.items():
                src = tr if portion == "tr" else tr + va
                c = Counter(vis_class(p) for p in src)
                top = max(c.values())
                tied = [k for k, v in c.items() if v == top]
                mj[s] = {"lower": min(tied), "higher": max(tied), "first": tied[0]}[rule]
            qq = np.array([float(mj[site_of(p)]) for p in lf])
            alt.append(mae(t, qq))
            note(f"   majority variant {portion}/{rule}: MAE {mae(t, qq):.3f} logMAE {logmae(t, qq):.3f}")
    note(f"   majority baseline MAE range over variants: {min(alt):.3f}-{max(alt):.3f}")
    c55 = np.full_like(t, 5.5)
    M2 = metrics(t, c55)
    note(f"T6.1 baseline constant 5.5: MAE {M2['mae']:.3f} RMSE {M2['rmse']:.3f} logMAE {M2['logmae']:.3f}")
    body.append("\\midrule")
    body.append("\\multicolumn{6}{l}{\\emph{Baselines that ignore the image}} \\\\")
    body.append(f"\\quad Site's majority class\\textsuperscript{{b}} & {M['acc']:.1f} & {M['within2']:.1f} & "
                f"{M['mae']:.3f} & {M['rmse']:.3f} & {M['logmae']:.3f} \\\\")
    body.append(f"\\quad Constant 5.5 mi\\textsuperscript{{c}} & -- & -- & "
                f"{M2['mae']:.3f} & {M2['rmse']:.3f} & {M2['logmae']:.3f} \\\\")
    write("t6_main_body.tex", "\n".join(body) + "\n")

    # rankings by metric (for the extinction-space remark)
    names = [label(m, r) for m, r, _ in CONFIGS]
    for key in ("mae", "logmae", "extmae"):
        order = sorted(names, key=lambda n: metrics(*A[n])[key])
        note(f"ranking by {key}: " + " < ".join(order))

    # paired bootstrap
    P = lambda m, r: A[label(m, r)][1]
    comps = [
        ("Adaptation (pooled $-$ fine-tuned)", None),
        ("VisNet", ("VisNet", "Pooled", "VisNet", "Fine-tuned")),
        ("RMEP", ("RMEP", "Pooled", "RMEP", "Fine-tuned")),
        ("Similarity", ("VisNet-Similarity", "Pooled", "VisNet-Similarity", "Fine-tuned")),
        ("Initialization (scratch $-$ fine-tuned)", None),
        ("VisNet", ("VisNet", "Scratch", "VisNet", "Fine-tuned")),
        ("RMEP", ("RMEP", "Scratch", "RMEP", "Fine-tuned")),
        ("Similarity\\textsuperscript{a}", ("VisNet-Similarity", "Per-site heads", "VisNet-Similarity", "Fine-tuned")),
        ("Architecture, pooled", None),
        ("RMEP $-$ VisNet", ("RMEP", "Pooled", "VisNet", "Pooled")),
        ("Similarity $-$ VisNet", ("VisNet-Similarity", "Pooled", "VisNet", "Pooled")),
        ("Architecture, fine-tuned", None),
        ("VisNet $-$ RMEP", ("VisNet", "Fine-tuned", "RMEP", "Fine-tuned")),
        ("Similarity $-$ RMEP", ("VisNet-Similarity", "Fine-tuned", "RMEP", "Fine-tuned")),
        ("Similarity $-$ VisNet", ("VisNet-Similarity", "Fine-tuned", "VisNet", "Fine-tuned")),
    ]
    res = []
    for name, c in comps:
        if c is None:
            res.append((name, None))
            continue
        a, b = P(c[0], c[1]), P(c[2], c[3])
        r1 = paired_bootstrap(t, a, b, mae)
        r2 = paired_bootstrap(t, a, b, logmae)
        res.append((name, (r1, r2)))
        note(f"T6.2 {c[0]} {c[1]} - {c[2]} {c[3]}: dMAE {r1[0]:+.3f} [{r1[1]:+.3f}, {r1[2]:+.3f}] p={r1[3]:.4f}; "
             f"dlog {r2[0]:+.4f} [{r2[1]:+.4f}, {r2[2]:+.4f}] p={r2[3]:.4f}")
    # Holm over the eleven MAE comparisons
    ps_ = [(i, x[1][0][3]) for i, x in enumerate(res) if x[1] is not None]
    m_ = len(ps_)
    holm = {}
    stop = False
    for rank, (i, p) in enumerate(sorted(ps_, key=lambda z: z[1])):
        thr = 0.05 / (m_ - rank)
        if not stop and p <= thr:
            holm[i] = True
        else:
            stop = True
            holm[i] = False
    note(f"Holm over {m_} MAE comparisons: significant = "
         f"{[res[i][0] for i in holm if holm[i]]}")
    body = []
    for i, (name, r) in enumerate(res):
        if r is None:
            if i:
                body.append("\\midrule")
            body.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{name}}}}} \\\\")
            continue
        (o, lo, hi, p), (o2, lo2, hi2, p2) = r
        mark = "\\textsuperscript{*}" if holm[i] else ""
        body.append(f"\\quad {name} & {sgn(o)}{mark} & [{sgn(lo)}, {sgn(hi)}] & {fmt_p(p)} & "
                    f"{sgn(o2)} & {fmt_p(p2)} \\\\")
    write("t6_bootstrap_body.tex", "\n".join(body) + "\n")
    return preds, common, lf


def persite_behaviour(repo, ps, preds, common, lf):
    """Constant predictors, selected epochs, and per-site log-MAE."""
    body = []
    for m, r, folder in CONFIGS:
        own = load_predictions(repo, folder)
        per = defaultdict(set)
        cnt = Counter()
        for p, row in own.items():
            per[row["site_id"]].add(float(row["pred_vis"]))
            cnt[row["site_id"]] += 1
        multi = [s for s in per if cnt[s] > 1]
        const = [s for s in multi if len(per[s]) == 1]
        majority = 0
        for s in const:
            tr, va, te = ps[s]
            c = Counter(vis_class(p) for p in tr)
            top = max(c.values())
            if int(next(iter(per[s]))) in [k for k, v in c.items() if v == top]:
                majority += 1
        # per-site log-MAE on the evaluation set
        d = preds[label(m, r)]
        e = defaultdict(list)
        for p in lf:
            e[site_of(p)].append(abs(np.log10(float(d[p]["true_vis"])) - np.log10(float(d[p]["pred_vis"]))))
        v = np.array([np.mean(x) for x in e.values()])
        if r == "Pooled":
            be = "--"
        else:
            rows = load_persite(repo, folder)
            be = f"{np.median([int(x['best_epoch']) for x in rows]):.0f}"
            ep1 = 100 * np.mean([int(x['best_epoch']) == 1 for x in rows])
            capped = sum(int(x["epochs_trained"]) == 60 for x in rows)
            note(f"T6.3 {label(m, r)}: best epoch median {be}, best=1 at {ep1:.0f}% of sites, "
                 f"{capped} sites ran all 60 epochs")
        note(f"T6.3 {label(m, r)}: constant at {len(const)}/{len(per)} sites "
             f"({len(multi)} with >1 test image; {majority} = a training-majority class); "
             f"per-site log-MAE on eval set: median {np.median(v):.3f} over {len(v)} sites")
        body.append(f"{ROW_NAMES[(m, r)]} & {r if r != 'Per-site heads' else 'Scratch'} & "
                    f"{len(const)} / {len(per)} & {be} & {np.median(v):.3f} \\\\")
        if r == "Pooled" and m == "VisNet-Similarity" or r == "Per-site heads":
            body.append("\\midrule")
    write("t6_persite_body.tex", "\n".join(body) + "\n")


def correlations(repo, preds, common):
    """Per-site metrics vs training images (n) and test classes (k), 253 sites."""
    names = [(m, r) for m, r, _ in CONFIGS if r != "Pooled"]
    keys = [("acc", "Accuracy"), ("within2", "$\\pm$2"), ("mae", "MAE"), ("rmse", "RMSE"), ("logmae", "log-MAE")]
    out = {}
    for m, r in names:
        d = preds[label(m, r)]
        by = defaultdict(list)
        for p in common:
            by[site_of(p)].append(p)
        sites = sorted(by)
        n = np.array([int(d[by[s][0]]["n_train"]) for s in sites], float)
        k = np.array([len({float(d[p]["true_vis"]) for p in by[s]}) for s in sites], float)
        rows = {}
        for key, _ in keys:
            y = np.array([metrics(*arrays(preds, label(m, r), by[s]))[key] for s in sites])
            rn = np.corrcoef(n, y)[0, 1]
            rho = stats.spearmanr(n, y)[0]
            rk = np.corrcoef(k, y)[0, 1]
            rnk = np.corrcoef(n, k)[0, 1]
            partial = (rn - rk * rnk) / np.sqrt((1 - rk ** 2) * (1 - rnk ** 2))
            rows[key] = (rn, rho, rk, partial)
        out[(m, r)] = rows
        note(f"T6.4 {label(m, r)} ({len(sites)} sites, r(n,k)={np.corrcoef(n, k)[0, 1]:+.2f}): " +
             "; ".join(f"{kk} {v[0]:+.2f}/{v[1]:+.2f}/{v[2]:+.2f}/{v[3]:+.2f}" for kk, v in rows.items()))
    # main-text table: accuracy and log-MAE
    body = []
    for m, r in names:
        a, l = out[(m, r)]["acc"], out[(m, r)]["logmae"]
        reg = "Scratch" if r in ("Scratch", "Per-site heads") else "Fine-tuned"
        short = ROW_NAMES[(m, r)].replace("VisNet-Similarity", "Similarity")
        body.append(f"{short} & {reg} & " + " & ".join(sgn(x, 2) for x in a) + " & " +
                    " & ".join(sgn(x, 2) for x in (l[0], l[2], l[3])) + " \\\\")
    write("t6_corr_body.tex", "\n".join(body) + "\n")
    # appendix table: all metrics
    body = []
    for m, r in names:
        reg = "Scratch" if r in ("Scratch", "Per-site heads") else "Fine-tuned"
        first = True
        for key, kname in keys:
            v = out[(m, r)][key]
            lead = f"{ROW_NAMES[(m, r)]} ({reg.lower()})" if first else ""
            body.append(f"{lead} & {kname} & " + " & ".join(sgn(x, 2) for x in v) + " \\\\")
            first = False
        body.append("\\midrule")
    write("tC_corr_full_body.tex", "\n".join(body[:-1]) + "\n")


def lda_tables(repo):
    base = os.path.join(repo, "lda_results")

    def acc(f, emb="full"):
        with open(os.path.join(base, f)) as fh:
            for r in csv.DictReader(fh):
                if r["embedding"] == emb:
                    return r
    rows = []
    for arch, name in (("visnet", "VisNet"), ("rmep", "RMEP")):
        tr = acc(f"sites_{arch}_summary.csv")
        un = [acc(f"sites_{arch}_untrained{s}_summary.csv") for s in ("", "_seed1", "_seed2")]
        ua = [float(u["acc_event_grouped"]) for u in un]
        note(f"T7.1 {name}: trained {float(tr['acc_event_grouped']):.3f} ± {float(tr['sd_event_grouped']):.3f} "
             f"(image-level {float(tr['acc_image_level']):.3f}); untrained seeds 42/1/2 "
             + "/".join(f"{x:.3f}" for x in ua) + f" mean {np.mean(ua):.3f}; dim {tr['embedding_dim']}, "
             f"pca {tr['pca_dim']}, images {tr['n_images']}, events {tr['n_events']}, "
             f"pooled-train {tr['n_pooled_train']}")
        rows.append(f"{name} & {int(tr['embedding_dim']):,} & {pct(tr['acc_event_grouped'])} $\\pm$ "
                    f"{pct(tr['sd_event_grouped'])} & "
                    + " / ".join(pct(u["acc_event_grouped"]) for u in un) + " \\\\")
    for emb, nm in (("full", "VisNet-Similarity, $[\\mathbf{p}_{\\text{curr}}, \\mathbf{p}_{\\text{diff}}]$"),
                    ("p_curr", "\\quad current image only, $\\mathbf{p}_{\\text{curr}}$"),
                    ("p_diff", "\\quad difference only, $\\mathbf{p}_{\\text{diff}}$")):
        r = acc("sites_similarity_summary.csv", emb)
        note(f"T7.1 Similarity {emb}: {float(r['acc_event_grouped']):.3f} ± {float(r['sd_event_grouped']):.3f} "
             f"(image-level {float(r['acc_image_level']):.3f}), dim {r['embedding_dim']}")
        rows.append(f"{nm} & {r['embedding_dim']} & {pct(r['acc_event_grouped'])} $\\pm$ "
                    f"{pct(r['sd_event_grouped'])} & -- \\\\")
    write("t7_sites_body.tex", "\n".join(rows) + "\n")

    rows = []
    for arch, name in (("rmep", "RMEP"), ("visnet", "VisNet"), ("similarity", "VisNet-Similarity")):
        with open(os.path.join(base, f"onesite_{arch}_summary.csv")) as fh:
            d = {r["site"]: r for r in csv.DictReader(fh)}
        cells = []
        for s in ("SITE22", "SITE43"):
            r = d[s]
            note(f"T7.2 {name} {s}: rho {float(r['spearman_rho_heldout']):+.2f} ± {float(r['rho_sd']):.2f}, "
                 f"acc {100*float(r['acc_heldout']):.1f} ± {100*float(r['acc_sd']):.1f}, majority "
                 f"{100*float(r['majority_rate']):.1f}; n {r['n_images']}, pooled-train {r['n_pooled_train']}, "
                 f"events {r['n_events']}, classes {r['n_classes']}")
            cells.append(f"{sgns(r['spearman_rho_heldout'])} $\\pm$ {dec(r['rho_sd'])} & "
                         f"{pct(r['acc_heldout'])}")
        rows.append(f"{name} & " + " & ".join(cells) + " \\\\")
    write("t7_onesite_body.tex", "\n".join(rows) + "\n")


def pooled_overall(repo):
    body = []
    for name, pre in (("VisNet", "VisNet"), ("RMEP", "RMEP"), ("VisNet-Similarity", "Similarity")):
        cells = []
        for split in ("validation", "test"):
            f = os.path.join(repo, f"{pre}SiteEvaluationResults{split.capitalize()}",
                             f"site_specific_{split}_epoch0.csv")
            with open(f) as fh:
                rows = {r["site_id"]: r for r in csv.DictReader(fh)}
            o = rows["OVERALL"]
            per = [r for k, r in rows.items() if k != "OVERALL"]
            note(f"T6.0 {name} pooled {split}: n {o['n_samples']}, acc {pct(o['accuracy'])}, "
                 f"w2 {pct(o['accuracy_within_2'])}, MAE {float(o['MAE_visibility']):.3f}, "
                 f"RMSE {float(o['RMSE_visibility']):.3f}; per-site rows {len(per)} sites, "
                 f"{sum(int(r['n_samples']) for r in per)} images")
            cells.append(f"{pct(o['accuracy'])} & {float(o['MAE_visibility']):.3f}")
        body.append(f"{name} & " + " & ".join(cells) + " \\\\")
    write("t6_pooled_body.tex", "\n".join(body) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".", help="the vis_networks folder (default: the current folder)")
    ap.add_argument("--out", default="thesis_tables", help="output folder (default: thesis_tables)")
    args = ap.parse_args()
    global OUT
    OUT = args.out
    os.makedirs(OUT, exist_ok=True)
    ps = dataset_tables(args.repo)
    preds, common, lf = main_table(args.repo, ps)
    persite_behaviour(args.repo, ps, preds, common, lf)
    correlations(args.repo, preds, common)
    lda_tables(args.repo)
    pooled_overall(args.repo)
    with open(os.path.join(OUT, "numbers.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
