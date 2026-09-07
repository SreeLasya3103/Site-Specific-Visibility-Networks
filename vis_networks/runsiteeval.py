import argparse
import os
import sys
import importlib.util
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import torchvision.transforms as tf

from sitespecificeval import evaluate_by_site, save_site_results, log_site_results_to_tensorboard


def load_config():
    spec = importlib.util.spec_from_file_location(
        "config", os.path.join(os.getcwd(), "config.py")
    )
    cfg = importlib.util.module_from_spec(spec)
    cfg.__package__ = "vis_networks"
    sys.modules["config"] = cfg
    spec.loader.exec_module(cfg)
    return SimpleNamespace(**cfg.CONFIG)


def main():
    parser = argparse.ArgumentParser(description="Site-specific evaluation of global model")
    parser.add_argument("--checkpoint", required=True, help="Path to best_epochN.pt")
    parser.add_argument("--split", default="validation", choices=["validation", "test"],
                        help="Which split to evaluate on")
    parser.add_argument("--min_samples", type=int, default=4,
                        help="Min samples per site to include in evaluation")
    parser.add_argument("--output_dir", default="site_eval_results",
                        help="Output directory for results")
    args = parser.parse_args()

    cf = load_config()
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup transforms (no augmentation for evaluation)
    transformer = lambda x: x
    if hasattr(cf.model_module, 'CUSTOM_TRANSFORM') and cf.model_module.CUSTOM_TRANSFORM is not None:
        transformer = cf.model_module.CUSTOM_TRANSFORM(**cf.transform_params)

    # Load split
    split_file = os.path.join(cf.splits_path, f"{args.split}.csv")
    with open(split_file) as f:
        samples = [line.rstrip() for line in f if line.strip()]
    print(f"Loaded {len(samples)} samples from {args.split} split")

    # Create dataset and loader
    DsetClass = cf.dset_module.DsetClass
    dim = cf.dset_params.get('dim', (280, 280))
    n_channels = cf.dset_params.get('n_channels', 3)
    extra = {k: v for k, v in cf.dset_params.items() if k not in ('dim', 'n_channels')}

    dataset = DsetClass(cf.dset_path, samples, dim, n_channels, transformer, **extra)
    num_workers = 0 if os.name == 'nt' else cf.n_workers
    loader = DataLoader(dataset, batch_size=cf.batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    # Load model
    ModelClass = cf.model_module.Model
    num_classes = len(DsetClass.CLASS_NAMES)
    sample = next(iter(loader))[0][0]

    model = ModelClass(num_classes, sample.size(), **cf.model_params)
    state_dict = torch.load(args.checkpoint, weights_only=True, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model(sample.unsqueeze(0))  # initialize lazy layers

    device = 'cuda' if cf.cuda and torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    # Output functions
    output_fn = cf.output_func if cf.output_func else (lambda x: x)
    label_fn = cf.label_func if cf.label_func else (lambda x: x)

    scaler = torch.amp.GradScaler(device, enabled=cf.use_amp)
    class_names = DsetClass.CLASS_NAMES

    print(f"\nEvaluating on {args.split} split with min_samples={args.min_samples}...")
    site_results, overall_results, skipped = evaluate_by_site(
        loader, model, cf.loss_func, device, scaler,
        output_fn, label_fn, class_names,
        min_samples=args.min_samples
    )

    # Save results
    writer = SummaryWriter(args.output_dir)
    df = save_site_results(site_results, overall_results, args.output_dir, epoch=0, set_name=args.split)
    log_site_results_to_tensorboard(site_results, overall_results, writer, epoch=0,
                                     set_name=args.split, skipped_sites=skipped)
    writer.close()

    # Print formatted summary
    print(f"\n{'='*100}")
    print(f"{'Site':<12} {'N':<6} {'Accuracy':<10} {'±2 Acc':<10} {'R²':<10} {'MAE(mi)':<10} {'RMSE(mi)':<10}")
    print(f"{'='*100}")

    for site_id in sorted(site_results.keys()):
        r = site_results[site_id]
        m = r['metrics']
        r2_val = m['R2'][0]
        r2_str = f"{r2_val:.4f}" if r2_val == r2_val else "nan"  # nan check
        print(f"{site_id:<12} {r['n_samples']:<6} {m['accuracy'][0]:<10.4f} "
              f"{m['accuracy_within_2'][0]:<10.4f} {r2_str:<10} "
              f"{m['MAE_visibility'][0]:<10.4f}{m['RMSE_visibility'][0]:<10.4f}")

    om = overall_results['metrics']
    print(f"{'='*100}")
    print(f"{'OVERALL':<12} {overall_results['n_samples']:<6} {om['accuracy'][0]:<10.4f} "
          f"{om['accuracy_within_2'][0]:<10.4f} {om['R2'][0]:<10.4f} "
          f"{om['MAE_visibility'][0]:<10.4f}{om['RMSE_visibility'][0]:<10.4f}")
    print(f"{'='*100}")

    # Summary stats
    import numpy as np
    r2s = [site_results[s]['metrics']['R2'][0] for s in site_results
           if site_results[s]['metrics']['R2'][0] == site_results[s]['metrics']['R2'][0]]
    print(f"\nSites evaluated: {len(site_results)}")
    print(f"Sites skipped (n < {args.min_samples}): {len(skipped)}")
    print(f"Sites with R² > 0.89: {sum(1 for r in r2s if r > 0.89)}")
    print(f"Sites with R² > 0.5:  {sum(1 for r in r2s if r > 0.5)}")
    print(f"Sites with R² > 0:    {sum(1 for r in r2s if r > 0)}")
    print(f"Sites with R² < 0:    {sum(1 for r in r2s if r < 0)}")
    print(f"\nResults saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
