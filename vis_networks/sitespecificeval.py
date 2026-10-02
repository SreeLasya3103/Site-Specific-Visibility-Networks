import torch
import os
from collections import defaultdict
import pandas as pd
from run_set import run_set


def extract_site_id(path: str) -> str:
    basename = os.path.basename(path)
    site_id = basename.split('_')[0]
    return site_id


def compute_accuracy_within_k(pred_classes: torch.Tensor,
                               true_classes: torch.Tensor,
                               k: int = 2) -> float:
    diff = torch.abs(pred_classes - true_classes)
    within_k = (diff <= k).float()
    return within_k.mean().item()


def compute_r2_from_classification(outputs: torch.Tensor,
                                   targets: torch.Tensor,
                                   class_names) -> tuple:
    outputs = outputs.cpu()
    targets = targets.cpu()

    # Parse class names to visibility values
    class_values = torch.tensor(
        [float(name) for name in class_names], dtype=torch.float32
    )

    # Get predicted and true class indices
    sm_outputs   = torch.softmax(outputs, dim=1)
    pred_classes = torch.argmax(sm_outputs, dim=1)
    true_classes = torch.argmax(targets, dim=1)

    # Map to visibility values (miles)
    pred_visibility = class_values[pred_classes]
    true_visibility = class_values[true_classes]

    ss_res = torch.sum((true_visibility - pred_visibility) ** 2)
    ss_tot = torch.sum((true_visibility - true_visibility.mean()) ** 2)

    if ss_tot.item() == 0.0:
        mae = torch.mean(torch.abs(pred_visibility - true_visibility)).item()
        r2  = 1.0 if mae == 0.0 else float('nan')
    else:
        r2 = 1.0 - (ss_res / ss_tot).item()

    return r2, pred_visibility, true_visibility, pred_classes, true_classes


def evaluate_by_site(loader,
                     model,
                     loss_fn,
                     device,
                     scaler,
                     output_fn,
                     labels_fn,
                     class_names,
                     min_samples: int = 10):
    
    print("Running inference on entire dataset...")
    all_outputs, all_labels, all_paths, overall_loss = run_set(
        loader, model, None, loss_fn, device, scaler, output_fn, labels_fn
    )

    all_outputs = all_outputs.cpu()
    all_labels  = all_labels.cpu()

    # Group by site
    site_data = defaultdict(lambda: {'outputs': [], 'targets': [], 'paths': []})

    for i, p in enumerate(all_paths):
        sid = extract_site_id(p)
        site_data[sid]['outputs'].append(all_outputs[i])
        site_data[sid]['targets'].append(all_labels[i])
        site_data[sid]['paths'].append(p)

    site_results  = {}
    skipped_sites = []

    print(f"\nFound {len(site_data)} unique sites in this split.")

    # Per-site evaluation
    evaluated = 0
    for site_id, data in site_data.items():
        n_samples = len(data['paths'])

        # Skip sites with too few samples
        if n_samples < min_samples:
            skipped_sites.append((site_id, n_samples))
            continue

        outputs = torch.stack(data['outputs'])
        targets = torch.stack(data['targets'])
        paths   = data['paths']

        # Class predictions
        sm_outputs   = torch.softmax(outputs, dim=1)
        pred_classes = torch.argmax(sm_outputs, dim=1)
        true_classes = torch.argmax(targets, dim=1)

        # Exact-match accuracy
        acc = (pred_classes == true_classes).float().mean().item()

        site_metrics = {}
        site_metrics['accuracy'] = (acc, 'scalar')

        # ±2 accuracy
        acc_within_2 = compute_accuracy_within_k(pred_classes, true_classes, k=2)
        site_metrics['accuracy_within_2'] = (acc_within_2, 'scalar')

        # R², MAE, RMSE in visibility space
        try:
            r2, pred_vis, true_vis, _, _ = compute_r2_from_classification(
                outputs, targets, class_names
            )
            mae  = torch.mean(torch.abs(pred_vis - true_vis)).item()
            rmse = torch.sqrt(torch.mean((pred_vis - true_vis) ** 2)).item()
        except Exception as e:
            print(f"  Warning: Metric calc failed for {site_id}: {e}")
            r2   = float('nan')
            mae  = float('nan')
            rmse = float('nan')

        site_metrics['R2']               = (r2,   'scalar')
        site_metrics['MAE_visibility']   = (mae,  'scalar')
        site_metrics['RMSE_visibility']  = (rmse, 'scalar')

        site_results[site_id] = {
            'metrics':   site_metrics,
            'n_samples': n_samples,
            'outputs':   outputs,
            'targets':   targets,
            'paths':     paths,
        }

        r2_str = f'{r2:.4f}' if not (r2 != r2) else 'nan   '   # nan check
        print(f"  {site_id}: n={n_samples:4d}, Acc={acc:.4f}, "
              f"±2Acc={acc_within_2:.4f}, R²={r2_str}, MAE={mae:.4f}")
        evaluated += 1

    print(f"\nEvaluated {evaluated} sites  |  Skipped {len(skipped_sites)} sites "
          f"(n < {min_samples})")

    # Overall metrics (all samples, no filtering)
    sm_all           = torch.softmax(all_outputs, dim=1)
    all_pred_classes = torch.argmax(sm_all, dim=1)
    all_true_classes = torch.argmax(all_labels, dim=1)

    overall_acc        = (all_pred_classes == all_true_classes).float().mean().item()
    overall_acc_within_2 = compute_accuracy_within_k(
        all_pred_classes, all_true_classes, k=2
    )

    overall_r2, all_pred_vis, all_true_vis, _, _ = compute_r2_from_classification(
        all_outputs, all_labels, class_names
    )
    overall_mae  = torch.mean(torch.abs(all_pred_vis - all_true_vis)).item()
    overall_rmse = torch.sqrt(torch.mean((all_pred_vis - all_true_vis) ** 2)).item()

    overall_metrics = {
        'accuracy':         (overall_acc,          'scalar'),
        'accuracy_within_2':(overall_acc_within_2, 'scalar'),
        'R2':               (overall_r2,            'scalar'),
        'MAE_visibility':   (overall_mae,           'scalar'),
        'RMSE_visibility':  (overall_rmse,          'scalar'),
        'loss':             (overall_loss,          'scalar'),
    }

    overall_results = {
        'metrics':   overall_metrics,
        'n_samples': len(all_paths),
        'outputs':   all_outputs,
        'targets':   all_labels,
        'paths':     all_paths,
    }

    return site_results, overall_results, skipped_sites


def save_site_results(site_results,
                      overall_results,
                      log_dir,
                      epoch: int,
                      set_name: str = 'validation'):
    summary_data = []

    for site_id, result in sorted(site_results.items()):
        row = {'site_id': site_id, 'n_samples': result['n_samples']}
        for metric_name, (value, _) in result['metrics'].items():
            row[metric_name] = value
        summary_data.append(row)

    # Overall row (append last)
    overall_row = {'site_id': 'OVERALL', 'n_samples': overall_results['n_samples']}
    for metric_name, (value, _) in overall_results['metrics'].items():
        if metric_name != 'loss':
            overall_row[metric_name] = value
    summary_data.append(overall_row)

    df = pd.DataFrame(summary_data)
    csv_path = os.path.join(log_dir, f'site_specific_{set_name}_epoch{epoch}.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved site-specific results → {csv_path}")
    return df


def log_site_results_to_tensorboard(site_results,
                                    overall_results,
                                    writer,
                                    epoch: int,
                                    set_name: str = 'validation',
                                    skipped_sites=None):
    
    # Overall metrics
    for metric_name, (value, m_type) in overall_results['metrics'].items():
        if m_type == 'scalar':
            writer.add_scalar(f'{metric_name}/{set_name}_overall', value, epoch)

    # Per-site metrics (cap at 50 to avoid overwhelming TensorBoard)
    site_list = sorted(site_results.keys())
    if len(site_list) > 50:
        print(f"  Note: Logging first 50/{len(site_list)} sites to TensorBoard")
        site_list = site_list[:50]

    for site_id in site_list:
        result = site_results[site_id]
        for metric_name, (value, m_type) in result['metrics'].items():
            if m_type == 'scalar' and not pd.isna(value):
                writer.add_scalar(
                    f'{metric_name}/{set_name}_{site_id}', value, epoch
                )

    # Aggregate statistics across all evaluated sites
    all_accuracies = [
        site_results[sid]['metrics']['accuracy'][0]
        for sid in sorted(site_results.keys())
    ]
    all_acc2 = [
        site_results[sid]['metrics']['accuracy_within_2'][0]
        for sid in sorted(site_results.keys())
    ]
    all_r2s = [
        site_results[sid]['metrics']['R2'][0]
        for sid in sorted(site_results.keys())
        if not pd.isna(site_results[sid]['metrics']['R2'][0])
    ]
    all_maes = [
        site_results[sid]['metrics']['MAE_visibility'][0]
        for sid in sorted(site_results.keys())
        if not pd.isna(site_results[sid]['metrics']['MAE_visibility'][0])
    ]

    def _log_stats(values, prefix):
        if not values:
            return
        writer.add_scalar(f'{prefix}/{set_name}_mean', sum(values) / len(values), epoch)
        writer.add_scalar(f'{prefix}/{set_name}_min',  min(values), epoch)
        writer.add_scalar(f'{prefix}/{set_name}_max',  max(values), epoch)

    _log_stats(all_accuracies, 'accuracy_by_site_stats')
    _log_stats(all_acc2,       'accuracy_within_2_by_site_stats')
    _log_stats(all_r2s,        'R2_by_site_stats')
    _log_stats(all_maes,       'MAE_by_site_stats')

    # Log number of skipped sites if provided
    if skipped_sites is not None:
        writer.add_scalar(f'site_stats/{set_name}_skipped', len(skipped_sites), epoch)
        writer.add_scalar(f'site_stats/{set_name}_evaluated', len(site_results), epoch)

    writer.flush()
