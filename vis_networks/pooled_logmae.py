""" python pooled_logmae.py --checkpoint runs/RMEPGlobal/best_epoch37.pt --split test --output_dir RMEPLogMAETest """
import csv
import os

import torch

import runsiteeval
from sitespecificeval import compute_r2_from_classification

_save_site_results = runsiteeval.save_site_results


def save_with_predictions(site_results, overall_results, log_dir, epoch, set_name='validation'):
    df = _save_site_results(site_results, overall_results, log_dir, epoch=epoch, set_name=set_name)
    class_names = runsiteeval.load_config().dset_module.DsetClass.CLASS_NAMES
    _, pred_vis, true_vis, _, _ = compute_r2_from_classification(
        overall_results['outputs'], overall_results['targets'], class_names)
    with open(os.path.join(log_dir, f'predictions_{set_name}.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['path', 'true_vis', 'pred_vis'])
        for p, t, q in zip(overall_results['paths'], true_vis.tolist(), pred_vis.tolist()):
            w.writerow([p, t, q])
    log_mae = torch.mean(torch.abs(torch.log10(true_vis) - torch.log10(pred_vis))).item()
    print(f"\n  OVERALL log-MAE on {set_name} ({len(true_vis)} images): {log_mae:.4f}")
    return df


runsiteeval.save_site_results = save_with_predictions

if __name__ == "__main__":
    runsiteeval.main()
