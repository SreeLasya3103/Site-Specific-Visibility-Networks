#thesis_common.py - shared loaders for the thesis figure and table scripts.
import csv
import os
import random
from collections import Counter, defaultdict

import numpy as np

# (model, regime, folder) for the twelve configurations, in table order: grouped by
# regime, and within a regime RMEP, RMEP-Similarity, VisNet, VisNet-Similarity.
CONFIGS = [
    ("RMEP", "Pooled", "RMEPGlobalOnPerSiteTest"),
    ("RMEP-Similarity", "Pooled", "RMEPSimilarityGlobalOnPerSiteTest"),
    ("VisNet", "Pooled", "VisNetGlobalOnPerSiteTest"),
    ("VisNet-Similarity", "Pooled", "SimilarityGlobalOnPerSiteTest"),
    ("RMEP", "Scratch", "RMEPPerSiteAndPredictionsResults"),
    ("RMEP-Similarity", "Per-site heads", "RMEPSimilarityPerSiteAndPredictionsResults"),
    ("VisNet", "Scratch", "VisNetPerSiteAndPredictionsResults"),
    ("VisNet-Similarity", "Per-site heads", "SimilarityPerSiteAndPredictionsResults"),
    ("RMEP", "Fine-tuned", "RMEPFineTuneAndPredictionsResults"),
    ("RMEP-Similarity", "Fine-tuned", "RMEPSimilarityFineTuneAndPredictionsResults"),
    ("VisNet", "Fine-tuned", "VisNetFineTuneAndPredictionsResults"),
    ("VisNet-Similarity", "Fine-tuned", "SimilarityFineTuneAndPredictionsResults"),
]
NETWORKS = ["RMEP", "RMEP-Similarity", "VisNet", "VisNet-Similarity"]


def regime_of(r): #Per-site heads are the Similarity networks' version of training from scratch.
    return "Scratch" if r == "Per-site heads" else r
CLASS_GROUPS = [{0.0, 1.0, 1.25, 1.5, 1.75}, {2.0, 2.25, 2.5}] + [{float(k)} for k in range(3, 11)]


def label(model, regime):
    return f"{model} {regime}"


def read_list(repo, name):
    with open(os.path.join(repo, "splits", name)) as f:
        return [l.strip() for l in f if l.strip()]


def pooled_splits(repo):
    return {s: read_list(repo, f"{s}.csv") for s in ("train", "validation", "test")}


def site_of(path):
    return os.path.basename(path).split("_")[0]


def view_of(path):
    return "_".join(os.path.basename(path).split("_")[:2])


def event_of(path):
    return os.path.dirname(path)


def raw_label(path): #Reported visibility as written in the file name ('10+' kept as text).
    return os.path.basename(path).split("_")[2].split(".")[0].split("S")[1].split("m")[0].replace("-", ".")


def vis_value(path):
    s = raw_label(path)
    return min(10.0 if s == "10+" else float(s), 10.0)


def vis_class(path):
    v = vis_value(path)
    for i, g in enumerate(CLASS_GROUPS):
        if v in g:
            return i + 1
    return None


def load_predictions(repo, folder):
    with open(os.path.join(repo, folder, "predictions.csv"), newline="") as f:
        return {r["sample_path"]: r for r in csv.DictReader(f)}


def load_persite(repo, folder):
    with open(os.path.join(repo, folder, "persite_results.csv"), newline="") as f:
        return [r for r in csv.DictReader(f) if r["site_id"].startswith("SITE")]


def split_site_data(samples, train_ratio=0.8, seed=42, val_ratio=0.2):
    rng = random.Random(seed)
    events = defaultdict(list)
    for s in samples:
        events[os.path.dirname(s)].append(s)
    by_class = defaultdict(list)
    for event_id, frames in events.items():
        basename = os.path.basename(frames[0])
        try:
            vis_part = basename.split('_')[2].split('.')[0]
            vis_part = vis_part.replace('VIS', '').replace('mi', '').replace('-', '.')
            cls_key = float(vis_part)
        except Exception:
            cls_key = "unknown"
        by_class[cls_key].append(sorted(frames))
    train_events, test_events = [], []
    for cls, cls_events in by_class.items():
        rng.shuffle(cls_events)
        n_train = max(1, int(len(cls_events) * train_ratio))
        train_events.extend(cls_events[:n_train])
        test_events.extend(cls_events[n_train:])
    if len(test_events) == 0 and len(train_events) > 1:
        rng.shuffle(train_events)
        test_events.append(train_events.pop())
    rng.shuffle(train_events)
    rng.shuffle(test_events)
    n_val = int(len(train_events) * val_ratio)
    if len(train_events) - n_val < 1:
        n_val = 0
    val_events = train_events[:n_val]
    train_events = train_events[n_val:]
    flatten = lambda groups: [f for g in groups for f in g]
    return flatten(train_events), flatten(val_events), flatten(test_events)


def persite_splits(repo, min_samples=20):
    sp = pooled_splits(repo)
    allp = sp["train"] + sp["validation"] + sp["test"]
    by_site = defaultdict(list)
    for p in allp:
        by_site[site_of(p)].append(p)
    out = {}
    for s in sorted(by_site):
        if len(by_site[s]) < min_samples:
            continue
        tr, va, te = split_site_data(by_site[s])
        if len(te) < 2:
            continue
        out[s] = (tr, va, te)
    return out


def class_value(c):
    return float(c)


def metrics(t, p):
    t = np.asarray(t, float)
    p = np.asarray(p, float)
    return dict(
        acc=100 * np.mean(t == p),
        within2=100 * np.mean(np.abs(t - p) <= 2),
        within1=100 * np.mean(np.abs(t - p) <= 1),
        mae=np.mean(np.abs(t - p)),
        rmse=np.sqrt(np.mean((t - p) ** 2)),
        logmae=np.mean(np.abs(np.log10(t) - np.log10(p))),
        extmae=np.mean(np.abs(3.912 / p - 3.912 / t)),
    )


def load_all(repo):
    preds = {label(m, r): load_predictions(repo, f) for m, r, f in CONFIGS}
    common = sorted(set.intersection(*(set(d) for d in preds.values())))
    return preds, common


def leakage_free(repo, paths):
    train = set(read_list(repo, "train.csv"))
    return [p for p in paths if p not in train]


def arrays(preds, name, paths):
    d = preds[name]
    t = np.array([float(d[p]["true_vis"]) for p in paths])
    q = np.array([float(d[p]["pred_vis"]) for p in paths])
    return t, q


def paired_bootstrap(t, pa, pb, fn, reps=10000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(t)
    obs = fn(t, pa) - fn(t, pb)
    d = np.empty(reps)
    for b in range(reps):
        i = rng.integers(0, n, size=n)
        d[b] = fn(t[i], pa[i]) - fn(t[i], pb[i])
    p = 2 * min(np.mean(d <= 0), np.mean(d >= 0))
    return obs, np.percentile(d, 2.5), np.percentile(d, 97.5), p


def collection_rows(repo):
    with open(os.path.join(repo, "dataset_audit_all", "lowlight.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if not r["path"].replace("\\", "/").startswith("NewWebcams/")]

