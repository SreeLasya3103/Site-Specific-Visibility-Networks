import torcheval.metrics.functional as temf
import torchmetrics.functional as tmf
import torch
from ..metrics import MetricInfo, Ref

OUTPUTS = Ref()
TARGETS = Ref()
CLASS_NAMES = Ref()
NUM_CLASSES = Ref()

def get_metrics(outputs, targets, metrics_list=None):
    OUTPUTS.value = outputs
    TARGETS.value = targets

    recorded_metrics = dict()

    if metrics_list is None or len(metrics_list) == 0:
        for metric, info in METRICS.items():
            recorded_metrics[metric] = info.get_result()
    else:
        for metric, extra_params in metrics_list.items():
            info = METRICS[metric]
            recorded_metrics[metric] = info.get_result(extra_params)

    return recorded_metrics

METRICS = {
    'R2': MetricInfo(temf.r2_score, {'input':OUTPUTS, 'target':TARGETS}, 'scalar'),
    'MAE': MetricInfo(tmf.mean_absolute_error, {'preds':OUTPUTS, 'target':TARGETS}, 'scalar'),
    'MSE': MetricInfo(temf.mean_squared_error, {'input':OUTPUTS, 'target':TARGETS}, 'scalar'),
    'RMSE': MetricInfo(lambda **x:torch.sqrt(temf.mean_squared_error(**x)), {'input':OUTPUTS, 'target':TARGETS}, 'scalar'),
    'MAPE': MetricInfo(tmf.mean_absolute_percentage_error, {'preds':OUTPUTS, 'target':TARGETS}, 'scalar'),
}