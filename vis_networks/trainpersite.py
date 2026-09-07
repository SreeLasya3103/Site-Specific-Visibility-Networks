import argparse
import os
import sys
import importlib.util
import random
from collections import defaultdict
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import torchvision.transforms as tf


def load_config():
    spec = importlib.util.spec_from_file_location(
        "config", os.path.join(os.getcwd(), "config.py")
    )
    cfg = importlib.util.module_from_spec(spec)
    cfg.__package__ = "vis_networks"
    sys.modules["config"] = cfg
    spec.loader.exec_module(cfg)
    return SimpleNamespace(**cfg.CONFIG)


def extract_site_id(path: str) -> str:
    """Extract site ID from sample path."""
    return os.path.basename(path).split("_")[0]


def group_by_site(sample_list: list) -> dict:
    """Group sample paths by site ID."""
    groups = defaultdict(list)
    for s in sample_list:
        groups[extract_site_id(s)].append(s)
    return dict(groups)


def split_site_data(samples: list, train_ratio: float = 0.8, seed: int = 42):
    """Split a site's samples into train/test sets, stratified by visibility class."""
    rng = random.Random(seed)

    # Try stratified split by visibility class
    by_class = defaultdict(list)
    for s in samples:
        basename = os.path.basename(s)
        try:
            vis_part = basename.split('_')[2].split('.')[0]
            vis_part = vis_part.replace('VIS', '').replace('mi', '').replace('-', '.')
            cls_key = float(vis_part)
        except Exception:
            cls_key = "unknown"
        by_class[cls_key].append(s)


    train_samples, test_samples = [], []

    for cls, cls_samples in by_class.items():
        rng.shuffle(cls_samples)
        n_train = max(1, int(len(cls_samples) * train_ratio))
        train_samples.extend(cls_samples[:n_train])
        test_samples.extend(cls_samples[n_train:])

    # Ensure test set is not empty
    if len(test_samples) == 0 and len(train_samples) > 1:
        rng.shuffle(train_samples)
        test_samples.append(train_samples.pop())

    rng.shuffle(train_samples)
    rng.shuffle(test_samples)

    return train_samples, test_samples

def make_loader(cf, samples, transformer, batch_size, shuffle=True, drop_last=False):
    DsetClass = cf.dset_module.DsetClass
    dim = cf.dset_params.get("dim", (280, 280))
    n_channels = cf.dset_params.get("n_channels", 3)
    extra = {k: v for k, v in cf.dset_params.items() if k not in ("dim", "n_channels")}

    dataset = DsetClass(cf.dset_path, list(samples), dim, n_channels, transformer, **extra)
    if len(dataset) == 0:
        return None

    bs = min(batch_size, len(dataset))
    num_workers = 0 if os.name == "nt" else min(cf.n_workers, 2)

    return DataLoader(
        dataset,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last and len(dataset) >= bs,
    )

def create_fresh_model(cf, sample_tensor, train_loader=None, seed=42):
    """Create a fresh model instance, optionally computing normalization stats from training data."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    ModelClass = cf.model_module.Model
    DsetClass = cf.dset_module.DsetClass
    num_classes = len(DsetClass.CLASS_NAMES) if hasattr(DsetClass, "CLASS_NAMES") else 1

    # Compute normalization stats from this site's training data
    mean, std = None, None
    if train_loader is not None and cf.normalize:
        dims = sample_tensor.size()
        n = 0
        running_mean = torch.zeros(dims[:-2])
        for data, _, _ in train_loader:
            batch_sum = torch.sum(data, (0, -2, -1))
            count = data.size(0) * data.size(-2) * data.size(-1)
            running_mean = (n * running_mean + batch_sum) / (n + count)
            n += count
        running_variance = torch.zeros(dims[:-2])
        for data, _, _ in train_loader:
            dev = data - running_mean.view(1, *running_mean.size(), 1, 1)
            running_variance += torch.sum(torch.square(dev), (0, -2, -1)) / (n - 1)
        std = torch.sqrt(running_variance)
        mean = running_mean

    model = ModelClass(num_classes, sample_tensor.size(), mean, std, **cf.model_params)
    model(sample_tensor.unsqueeze(0))
    return model


def load_pretrained_model(cf, model_path, sample_tensor):
    """Load a pretrained global model for fine-tuning."""
    ModelClass = cf.model_module.Model
    DsetClass = cf.dset_module.DsetClass
    num_classes = len(DsetClass.CLASS_NAMES) if hasattr(DsetClass, "CLASS_NAMES") else 1

    model = ModelClass(num_classes, sample_tensor.size(), **cf.model_params)
    state_dict = torch.load(model_path, weights_only=True, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)

    if "example_input" in state_dict:
        model.register_buffer("example_input", state_dict["example_input"])
    if "example_output" in state_dict:
        model.register_buffer("example_output", state_dict["example_output"])

    return model

def compute_metrics(outputs, targets, class_names):
    outputs = outputs.cpu()
    targets = targets.cpu()

    class_values = torch.tensor([float(c) for c in class_names], dtype=torch.float32)

    sm = torch.softmax(outputs, dim=1)
    pred_classes = torch.argmax(sm, dim=1)
    true_classes = torch.argmax(targets, dim=1)

    # Exact accuracy
    acc = (pred_classes == true_classes).float().mean().item()

    # ±2 accuracy
    diff = torch.abs(pred_classes - true_classes)
    acc_within_2 = (diff <= 2).float().mean().item()

    # Map to visibility values
    pred_vis = class_values[pred_classes]
    true_vis = class_values[true_classes]

    # R²
    ss_res = torch.sum((true_vis - pred_vis) ** 2)
    ss_tot = torch.sum((true_vis - true_vis.mean()) ** 2)

    if ss_tot.item() == 0.0:
        mae_val = torch.mean(torch.abs(pred_vis - true_vis)).item()
        r2 = 1.0 if mae_val == 0.0 else float("nan")
    else:
        r2 = 1.0 - (ss_res / ss_tot).item()

    # MAE and RMSE
    mae = torch.mean(torch.abs(pred_vis - true_vis)).item()
    rmse = torch.sqrt(torch.mean((pred_vis - true_vis) ** 2)).item()

    return {
        "accuracy": acc,
        "accuracy_within_2": acc_within_2,
        "R2": r2,
        "MAE_visibility": mae,
        "RMSE_visibility": rmse,
    }

def train_one_site(
    model,
    train_loader,
    test_loader,
    loss_fn,
    class_names,
    device,
    epochs,
    lr,
    weight_decay,
    patience,
    use_amp=False,
):
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )
    scaler = torch.amp.GradScaler(device, enabled=use_amp)

    best_test_loss = float("inf")
    best_test_metrics = None
    best_epoch = 0
    patience_counter = 0
    history = []
    best_outputs = None   # raw logits from best epoch
    best_targets = None   # one-hot targets from best epoch

    for epoch in range(1, epochs + 1):
        # --- TRAIN ---
        model.train()
        train_loss = 0.0
        n_batches = 0

        for data, labels, _ in train_loader:
            data = data.to(device)
            labels = labels.to(device)

            with torch.autocast(device, torch.bfloat16, enabled=use_amp):
                output = model(data)
                loss = loss_fn(output, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            train_loss += loss.item()
            n_batches += 1

        train_loss /= max(n_batches, 1)
        scheduler.step()

        # --- TEST ---
        model.eval()
        all_outputs, all_targets = [], []
        test_loss = 0.0
        n_test_batches = 0

        with torch.inference_mode():
            for data, labels, _ in test_loader:
                data = data.to(device)
                labels = labels.to(device)

                with torch.autocast(device, torch.bfloat16, enabled=use_amp):
                    output = model(data)
                    loss = loss_fn(output, labels)

                test_loss += loss.item()
                n_test_batches += 1
                _cls = output[0] if isinstance(output, tuple) else output
                all_outputs.append(_cls.cpu())
                all_targets.append(labels.cpu())

        test_loss /= max(n_test_batches, 1)
        outputs = torch.cat(all_outputs, 0)
        targets = torch.cat(all_targets, 0)

        met = compute_metrics(outputs, targets, class_names)
        met["train_loss"] = train_loss
        met["test_loss"] = test_loss
        met["epoch"] = epoch
        history.append(met)

        # Early stopping on test loss
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_test_metrics = met.copy()
            best_epoch = epoch
            patience_counter = 0
            best_outputs = outputs.clone()   # save raw logits
            best_targets = targets.clone()   # save one-hot targets
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    return best_test_metrics, best_epoch, history, best_outputs, best_targets

def main():
    parser = argparse.ArgumentParser(
        description="Per Site Training"
    )
    parser.add_argument(
        "--pretrained_path",
        default=None,
        help="Path to pretrained global model (.pt). If provided, fine-tunes instead of training from scratch.",
    )
    parser.add_argument("--epochs", type=int, default=60, help="Max epochs per site")
    parser.add_argument(
        "--min_samples", type=int, default=20, help="Min samples per site to train"
    )
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: from config)")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train/test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output_dir", default="persite_results", help="Output directory"
    )
    parser.add_argument(
        "--use_all_splits",
        action="store_true",
        default=False,
        help="Pool train+val+test splits before per-site splitting",
    )
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load config
    if args.pretrained_path:
        print("Per-Site Fine-Tuning (from pretrained global model)")
    else:
        print("Per-Site Training")

    cf = load_config()
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(args.output_dir)

    # Setup transforms
    augmenter = tf.Compose(cf.augment_list) if cf.augment_list else (lambda x: x)
    transformer = lambda x: x
    if (
        hasattr(cf.model_module, "CUSTOM_TRANSFORM")
        and cf.model_module.CUSTOM_TRANSFORM is not None
    ):
        transformer = cf.model_module.CUSTOM_TRANSFORM(**cf.transform_params)

    train_transformer = lambda x: transformer(augmenter(x))

    device = "cuda" if cf.cuda and torch.cuda.is_available() else "cpu"
    class_names = cf.dset_module.DsetClass.CLASS_NAMES
    lr = args.lr if args.lr else cf.optimizer_params.get("lr", 1e-5)
    weight_decay = cf.optimizer_params.get("weight_decay", 1e-4)

    print(f"\nDevice:          {device}")
    print(f"Classes:         {class_names}")
    print(f"Epochs/site:     {args.epochs}")
    print(f"Min samples:     {args.min_samples}")
    print(f"Early stopping:  patience={args.patience}")
    print(f"Learning rate:   {lr}")
    print(f"Train ratio:     {args.train_ratio}")
    print(f"Pretrained:      {args.pretrained_path or 'No (from scratch)'}")

    # Load all samples
    print("\nLoading sample lists...")
    all_samples = []
    for split_name in ["train.csv", "validation.csv", "test.csv"]:
        fpath = os.path.join(cf.splits_path, split_name)
        if os.path.exists(fpath):
            with open(fpath) as f:
                samples = [line.rstrip() for line in f if line.strip()]
            all_samples.extend(samples)
            print(f"  {split_name}: {len(samples)} samples")

    if not args.use_all_splits:
        # Use only train + validation for per-site experiments
        # (reserve test set for final evaluation)
        all_samples_filtered = []
        for split_name in ["train.csv", "validation.csv"]:
            fpath = os.path.join(cf.splits_path, split_name)
            if os.path.exists(fpath):
                with open(fpath) as f:
                    all_samples_filtered.extend([l.rstrip() for l in f if l.strip()])
        all_samples = all_samples_filtered
        print(f"\n  Using train+validation only: {len(all_samples)} total samples")
    else:
        print(f"\n  Using ALL splits: {len(all_samples)} total samples")

    # Group by site
    sites = group_by_site(all_samples)
    print(f"  Total unique sites: {len(sites)}")

    # Filter by minimum samples
    eligible_sites = {
        sid: samples
        for sid, samples in sites.items()
        if len(samples) >= args.min_samples
    }
    skipped = len(sites) - len(eligible_sites)
    print(f"  Eligible sites (n>={args.min_samples}): {len(eligible_sites)}")
    print(f"  Skipped sites: {skipped}")

    # Get a sample tensor for model initialization
    dummy_loader = make_loader(
        cf, list(list(eligible_sites.values())[0])[:2], transformer, batch_size=1
    )
    sample_tensor = next(iter(dummy_loader))[0][0]

    # Optionally load pretrained model once
    global_model = None
    if args.pretrained_path:
        print(f"\nLoading pretrained model: {args.pretrained_path}")
        global_model = load_pretrained_model(cf, args.pretrained_path, sample_tensor)

    print("STARTING PER-SITE TRAINING")

    all_results = []
    all_predictions = []
    sorted_sites = sorted(eligible_sites.keys())


    # Overall metrics from ALL predictions combined
    global_pred_classes = []
    global_true_classes = []
    class_values = torch.tensor([float(c) for c in class_names], dtype=torch.float32)

    for i, site_id in enumerate(sorted_sites, 1):
        site_samples = eligible_sites[site_id]
        n_total = len(site_samples)

        # Split this site's data
        train_samples, test_samples = split_site_data(
            site_samples, train_ratio=args.train_ratio, seed=args.seed
        )
        n_train = len(train_samples)
        n_test = len(test_samples)

        if n_test < 2:
            print(f"[{i}/{len(sorted_sites)}] {site_id}: skipped (test set too small)")
            continue

        print(
            f"\n[{i}/{len(sorted_sites)}] {site_id}: "
            f"total={n_total}, train={n_train}, test={n_test}"
        )

        # Create loaders
        train_loader = make_loader(
            cf, train_samples, train_transformer, batch_size=min(16, n_train), shuffle=True
        )
        test_loader = make_loader(
            cf, test_samples, transformer, batch_size=min(16, n_test), shuffle=False
        )

        if train_loader is None or test_loader is None:
            print(f"  Skipped: empty loader")
            continue

        # Create model
        if global_model is not None:
            # Fine-tune from pretrained
            model = deepcopy(global_model)
        else:
            # Fresh model from scratch
            model = create_fresh_model(cf, sample_tensor, train_loader=train_loader, seed=args.seed + i)


        # Train
        best_met, best_ep, history, site_outputs, site_targets = train_one_site(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            loss_fn=cf.loss_func,
            class_names=class_names,
            device=device,
            epochs=args.epochs,
            lr=lr,
            weight_decay=weight_decay,
            patience=args.patience,
            use_amp=cf.use_amp,
        )

        if best_met is None:
            print(f"  Training failed")
            continue

        # Accumulate raw predictions for pooled overall metrics
        sm = torch.softmax(site_outputs, dim=1)
        global_pred_classes.append(torch.argmax(sm, dim=1))           
        global_true_classes.append(torch.argmax(site_targets, dim=1))

        # Save per-sample predictions for analysis
        pred_cls = torch.argmax(sm, dim=1)
        true_cls = torch.argmax(site_targets, dim=1)
        for path, tv, pv in zip(
            test_samples,
            class_values[true_cls].tolist(),
            class_values[pred_cls].tolist(),
        ):
            all_predictions.append({
                "site_id":    site_id,
                "n_train":    n_train,
                "sample_path": path,
                "true_vis":   tv,
                "pred_vis":   pv,
            })

        # Store results
        result = {
            "site_id": site_id,
            "n_total": n_total,
            "n_train": n_train,
            "n_test": n_test,
            "best_epoch": best_ep,
            "epochs_trained": len(history),
            **best_met,
        }
        all_results.append(result)

        # Print result
        r2_str = f"{best_met['R2']:.4f}" if not np.isnan(best_met["R2"]) else "nan   "
        print(
            f"  Best@ep{best_ep}: "
            f"Acc={best_met['accuracy']:.4f}, "
            f"+-2={best_met['accuracy_within_2']:.4f}, "
            f"R2={r2_str}, "
            f"MAE={best_met['MAE_visibility']:.4f}"
        )

        # Log to TensorBoard
        writer.add_scalar(f"persite_accuracy/{site_id}", best_met["accuracy"], 1)
        writer.add_scalar(f"persite_R2/{site_id}", best_met["R2"], 1)
        writer.add_scalar(f"persite_MAE/{site_id}", best_met["MAE_visibility"], 1)
        writer.flush()

    print("PER-SITE TRAINING RESULTS")

    df = pd.DataFrame(all_results)

    if len(df) == 0:
        print("No sites were successfully trained.")
        writer.close()
        return

    # Sort by site number
    def site_sort_key(sid):
        try:
            return int(sid.replace("SITE", ""))
        except ValueError:
            return 99999

    df["_sort"] = df["site_id"].apply(site_sort_key)
    df = df.sort_values("_sort").drop(columns=["_sort"])

    # Print formatted table
    header = f"{'Site':<12} {'N':<6} {'Accuracy':<10} {'+-2 Acc':<10} {'R2':<10} {'MAE(mi)':<10} {'RMSE(mi)':<10}"
    separator = "-" * 110

    print(separator)
    print(header)
    print(separator)

    for _, row in df.iterrows():
        r2_str = f"{row['R2']:.4f}" if not np.isnan(row["R2"]) else "nan   "
        line = (
            f"{row['site_id']:<12} "
            f"{int(row['n_test']):<6} "
            f"{row['accuracy']:<10.4f} "
            f"{row['accuracy_within_2']:<10.4f} "
            f"{r2_str:<10} "
            f"{row['MAE_visibility']:<10.4f}"
            f"{row['RMSE_visibility']:<10.4f}"
        )
        print(line)

   
    all_pred = torch.cat(global_pred_classes, dim=0)
    all_true = torch.cat(global_true_classes, dim=0)
    total_n = all_pred.size(0)

    overall_acc = (all_pred == all_true).float().mean().item()

    overall_acc2 = (torch.abs(all_pred - all_true) <= 2).float().mean().item()

    pred_vis = class_values[all_pred]
    true_vis = class_values[all_true]

    ss_res = torch.sum((true_vis - pred_vis) ** 2)
    ss_tot = torch.sum((true_vis - true_vis.mean()) ** 2)
    if ss_tot.item() == 0.0:
        overall_r2 = 1.0 if ss_res.item() == 0.0 else float("nan")
    else:
        overall_r2 = 1.0 - (ss_res / ss_tot).item()

    overall_mae = torch.mean(torch.abs(pred_vis - true_vis)).item()
    overall_rmse = torch.sqrt(torch.mean((pred_vis - true_vis) ** 2)).item()

    print(separator)
    r2_overall_str = f"{overall_r2:.4f}" if not np.isnan(overall_r2) else "nan   "
    print(
        f"{'OVERALL':<12} "
        f"{int(total_n):<6} "
        f"{overall_acc:<10.4f} "
        f"{overall_acc2:<10.4f} "
        f"{r2_overall_str:<10} "
        f"{overall_mae:<10.4f}"
        f"{overall_rmse:<10.4f}"
    )
    print(separator)

    # Summary statistics
    print(f"\nSummary Statistics ({len(df)} sites):")
    print(f"  Mean Accuracy:      {df['accuracy'].mean():.4f} +/- {df['accuracy'].std():.4f}")
    print(f"  Mean +-2 Accuracy:  {df['accuracy_within_2'].mean():.4f} +/- {df['accuracy_within_2'].std():.4f}")
    r2_series = df["R2"].dropna()
    r2_series = r2_series[~r2_series.apply(np.isnan)]
    if len(r2_series) > 0:
        print(f"  Mean R2:            {r2_series.mean():.4f} +/- {r2_series.std():.4f}")
        print(f"  Median R2:          {r2_series.median():.4f}")
    print(f"  Mean MAE:           {df['MAE_visibility'].mean():.4f} +/- {df['MAE_visibility'].std():.4f}")
    print(f"  Mean epochs trained: {df['epochs_trained'].mean():.1f}")

    # Performance tiers
    print(f"\nPerformance Distribution (by R2):")
    bins = [-np.inf, 0.0, 0.3, 0.5, 0.7, np.inf]
    labels_bins = ["Poor (<0)", "Fair (0-0.3)", "Good (0.3-0.5)", "Very Good (0.5-0.7)", "Excellent (>0.7)"]
    if len(r2_series) > 0:
        categories = pd.cut(r2_series, bins=bins, labels=labels_bins)
        print(categories.value_counts().sort_index().to_string())

    overall_row = pd.DataFrame([{
        "site_id": "OVERALL",
        "n_total": total_n,
        "n_train": df["n_train"].sum(),
        "n_test": total_n,
        "best_epoch": 0,
        "epochs_trained": 0,
        "accuracy": overall_acc,
        "accuracy_within_2": overall_acc2,
        "R2": overall_r2,
        "MAE_visibility": overall_mae,
        "RMSE_visibility": overall_rmse,
        "train_loss": 0.0,
        "test_loss": 0.0,
        "epoch": 0,
    }])
    df_with_overall = pd.concat([df, overall_row], ignore_index=True)
    csv_path = os.path.join(args.output_dir, "persite_results.csv")
    df_with_overall.to_csv(csv_path, index=False)
    print(f"\nSaved results CSV -> {csv_path}")

    pred_csv_path = os.path.join(args.output_dir, "predictions.csv")
    pd.DataFrame(all_predictions).to_csv(pred_csv_path, index=False)
    print(f"Saved predictions CSV -> {pred_csv_path}")

    txt_path = os.path.join(args.output_dir, "Site_Specific_Results.txt")
    with open(txt_path, "w") as f:
        f.write(separator + "\n")
        f.write(header + "\n")
        f.write(separator + "\n")
        for _, row in df.iterrows():
            r2_s = f"{row['R2']:.4f}" if not np.isnan(row["R2"]) else "nan   "
            f.write(
                f"{row['site_id']:<12} "
                f"{int(row['n_test']):<6} "
                f"{row['accuracy']:<10.4f} "
                f"{row['accuracy_within_2']:<10.4f} "
                f"{r2_s:<10} "
                f"{row['MAE_visibility']:<10.4f}"
                f"{row['RMSE_visibility']:<10.4f}\n"
            )
        f.write(separator + "\n")
        f.write(
            f"{'OVERALL':<12} "
            f"{int(total_n):<6} "
            f"{overall_acc:<10.4f} "
            f"{overall_acc2:<10.4f} "
            f"{r2_overall_str:<10} "
            f"{overall_mae:<10.4f}"
            f"{overall_rmse:<10.4f}\n"
        )
        f.write(separator + "\n")
    print(f"Saved formatted table -> {txt_path}")

    # Log overall to TensorBoard
    writer.add_scalar("overall/accuracy", overall_acc, 1)
    writer.add_scalar("overall/accuracy_within_2", overall_acc2, 1)
    writer.add_scalar("overall/R2", overall_r2, 1)
    writer.add_scalar("overall/MAE", overall_mae, 1)
    writer.add_scalar("overall/RMSE", overall_rmse, 1)
    writer.close()

    print(f"\nTensorBoard logs -> {args.output_dir}")
    print("\nDONE.")


if __name__ == "__main__":
    main()

