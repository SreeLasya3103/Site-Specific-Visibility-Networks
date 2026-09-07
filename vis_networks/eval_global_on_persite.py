"""
eval_global_on_persite.py

Evaluate a globally trained checkpoint on the EXACT pooled per-site test set
used by the per-site experiments (N = 3,232), so that the global and per-site
regimes become strictly comparable on identical data.

The sample list is read from any per-site predictions.csv; all per-site runs
share the same test samples, so the choice of source file does not matter.

Reads config.py from the current directory, exactly like main.py and
runsiteeval.py, so the active config selects the model and dataset.

Usage (run from vis_networks/, with the matching config in place):

    Copy-Item config_visnet.py config.py -Force
    python3 eval_global_on_persite.py ^
        --checkpoint "runs/VisNetGlobal/best_epoch19.pt" ^
        --predictions "VisNetFineTuneAndPredictionsResults/predictions.csv" ^
        --output_dir VisNetGlobalOnPerSiteTest

Outputs, in the same schema as the other result folders:
    persite_results.csv   per-site + OVERALL metrics
    predictions.csv       per-sample predictions (for paired bootstrap)
"""

import argparse
import csv
import importlib.util
import os
import sys
from collections import defaultdict
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader


def load_config():
    spec = importlib.util.spec_from_file_location(
        "config", os.path.join(os.getcwd(), "config.py")
    )
    cfg = importlib.util.module_from_spec(spec)
    cfg.__package__ = "vis_networks"
    sys.modules["config"] = cfg
    spec.loader.exec_module(cfg)
    return SimpleNamespace(**cfg.CONFIG)


def extract_site_id(path):
    return os.path.basename(path).split("_")[0]


def compute_metrics(pred_classes, true_classes, class_values):
    """Identical formulas to trainpersite.compute_metrics."""
    n = pred_classes.numel()
    acc = (pred_classes == true_classes).float().mean().item()
    acc2 = (torch.abs(pred_classes - true_classes) <= 2).float().mean().item()

    pred_vis = class_values[pred_classes]
    true_vis = class_values[true_classes]

    ss_res = torch.sum((true_vis - pred_vis) ** 2)
    ss_tot = torch.sum((true_vis - true_vis.mean()) ** 2)
    mae = torch.mean(torch.abs(pred_vis - true_vis)).item()
    if ss_tot.item() == 0.0:
        r2 = 1.0 if mae == 0.0 else float("nan")
    else:
        r2 = 1.0 - (ss_res / ss_tot).item()
    rmse = torch.sqrt(torch.mean((pred_vis - true_vis) ** 2)).item()
    return n, acc, acc2, r2, mae, rmse


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate a global checkpoint on the pooled per-site test set"
    )
    ap.add_argument("--checkpoint", required=True, help="Path to global best_epochNN.pt")
    ap.add_argument("--predictions", required=True,
                    help="Any per-site predictions.csv (supplies the sample list)")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    cf = load_config()
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- sample list from the per-site test set -------------------------
    with open(args.predictions, newline="") as f:
        wanted = [r["sample_path"] for r in csv.DictReader(f)]
    # de-duplicate, preserve order
    seen, samples = set(), []
    for s in wanted:
        if s not in seen:
            seen.add(s)
            samples.append(s)
    print(f"Loaded {len(samples)} test samples from {args.predictions}")

    # ---- dataset / loader (config-driven, mirrors runsiteeval.py) -------
    transformer = (lambda x: x)
    if getattr(cf.model_module, "CUSTOM_TRANSFORM", None) is not None:
        transformer = cf.model_module.CUSTOM_TRANSFORM(**cf.transform_params)

    DsetClass = cf.dset_module.DsetClass
    dim = cf.dset_params.get("dim", (280, 280))
    n_channels = cf.dset_params.get("n_channels", 3)
    extra = {k: v for k, v in cf.dset_params.items() if k not in ("dim", "n_channels")}

    dataset = DsetClass(cf.dset_path, list(samples), dim, n_channels, transformer, **extra)
    kept = list(dataset.sample_list)          # dataset drops unparseable labels
    if len(kept) != len(samples):
        print(f"WARNING: {len(samples) - len(kept)} sample(s) dropped (unparseable label)")

    num_workers = 0 if os.name == "nt" else min(getattr(cf, "n_workers", 2), 2)
    loader = DataLoader(dataset, batch_size=cf.batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    # ---- model ---------------------------------------------------------
    ModelClass = cf.model_module.Model
    class_names = DsetClass.CLASS_NAMES
    num_classes = len(class_names)

    probe = next(iter(loader))[0][0]
    model = ModelClass(num_classes, probe.size(), **cf.model_params)
    sd = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    missing = model.load_state_dict(sd, strict=False)
    model(probe.unsqueeze(0))                 # initialise LazyLinear layers

    n_loaded = sum(1 for k in sd if k in dict(model.named_parameters())
                   or k in dict(model.named_buffers()))
    print(f"Checkpoint: {len(sd)} tensors, {n_loaded} matched into the model")
    if n_loaded == 0:
        raise SystemExit("ERROR: no checkpoint tensors matched the model - wrong checkpoint?")

    device = "cuda" if cf.cuda and torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # ---- inference -----------------------------------------------------
    all_pred, all_true, all_paths = [], [], []
    with torch.inference_mode():
        for data, labels, paths in loader:
            out = model(data.to(device))
            out = out[0] if isinstance(out, tuple) else out
            all_pred.append(torch.argmax(torch.softmax(out.float(), 1), 1).cpu())
            all_true.append(torch.argmax(labels, 1).cpu())
            all_paths += list(paths)

    pred = torch.cat(all_pred)
    true = torch.cat(all_true)
    class_values = torch.tensor([float(c) for c in class_names], dtype=torch.float32)

    # ---- per-site + overall metrics ------------------------------------
    by_site = defaultdict(list)
    for i, p in enumerate(all_paths):
        by_site[extract_site_id(p)].append(i)

    rows = []
    for site in sorted(by_site):
        idx = torch.tensor(by_site[site])
        n, acc, acc2, r2, mae, rmse = compute_metrics(pred[idx], true[idx], class_values)
        rows.append([site, n, 0, n, 0, 0, acc, acc2, r2, mae, rmse, 0.0, 0.0, 0])

    n, acc, acc2, r2, mae, rmse = compute_metrics(pred, true, class_values)
    rows.append(["OVERALL", n, 0, n, 0, 0, acc, acc2, r2, mae, rmse, 0.0, 0.0, 0])

    header = ["site_id", "n_total", "n_train", "n_test", "best_epoch", "epochs_trained",
              "accuracy", "accuracy_within_2", "R2", "MAE_visibility", "RMSE_visibility",
              "train_loss", "test_loss", "epoch"]
    with open(os.path.join(args.output_dir, "persite_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    with open(os.path.join(args.output_dir, "predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["site_id", "n_train", "sample_path", "true_vis", "pred_vis"])
        for i, p in enumerate(all_paths):
            w.writerow([extract_site_id(p), 0, p,
                        class_values[true[i]].item(), class_values[pred[i]].item()])

    print("\n" + "=" * 74)
    print(f"{'GLOBAL MODEL ON POOLED PER-SITE TEST SET':^74}")
    print("=" * 74)
    print(f"  checkpoint : {args.checkpoint}")
    print(f"  sites      : {len(by_site)}")
    print(f"  N          : {n}")
    print(f"  R2         : {r2:.4f}")
    print(f"  Accuracy   : {acc:.4f}")
    print(f"  +-2 Acc    : {acc2:.4f}")
    print(f"  MAE  (mi)  : {mae:.4f}")
    print(f"  RMSE (mi)  : {rmse:.4f}")
    print("=" * 74)
    print(f"\nSaved to {args.output_dir}/")


if __name__ == "__main__":
    main()
