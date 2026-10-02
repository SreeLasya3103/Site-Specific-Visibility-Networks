import torcheval.metrics.functional as temf
import torchmetrics.functional as tmf
import torch
from . import figures
from ..metrics import MetricInfo, Ref

OUTPUTS = Ref()
I_TARGETS = Ref()
CLASS_NAMES = Ref()
NUM_CLASSES = Ref()

def get_metrics(outputs, targets, class_names, metrics_list=None):
    OUTPUTS.value = outputs
    I_TARGETS.value = torch.argmax(targets, 1)
    CLASS_NAMES.value = class_names
    NUM_CLASSES.value = len(class_names)

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
    'accuracy': MetricInfo(temf.multiclass_accuracy, {'input':OUTPUTS, 'target':I_TARGETS}, 'scalar'),
    'precision': MetricInfo(temf.multiclass_precision, {'input':OUTPUTS, 'target':I_TARGETS, 'num_classes':NUM_CLASSES, 'average':'macro'}, 'scalar'),
    'recall': MetricInfo(temf.multiclass_recall, {'input':OUTPUTS, 'target':I_TARGETS, 'num_classes':NUM_CLASSES, 'average':'macro'}, 'scalar'),
    'F1': MetricInfo(temf.multiclass_f1_score, {'input':OUTPUTS, 'target':I_TARGETS, 'num_classes':NUM_CLASSES, 'average':'macro'}, 'scalar'),
    'AUC': MetricInfo(temf.multiclass_auroc, {'input':OUTPUTS, 'target':I_TARGETS, 'num_classes':NUM_CLASSES}, 'scalar'),
    'expectedCE': MetricInfo(tmf.calibration_error, {'preds':OUTPUTS, 'target':I_TARGETS, 'task':'multiclass', 'norm':'l1', 'num_classes':NUM_CLASSES}, 'scalar'),
    'maxCE': MetricInfo(tmf.calibration_error, {'preds':OUTPUTS, 'target':I_TARGETS, 'task':'multiclass', 'norm':'max', 'num_classes':NUM_CLASSES}, 'scalar'),
    'RMSCE': MetricInfo(tmf.calibration_error, {'preds':OUTPUTS, 'target':I_TARGETS, 'task':'multiclass', 'norm':'l2', 'num_classes':NUM_CLASSES}, 'scalar'),
    'QWK': MetricInfo(tmf.cohen_kappa, {'preds':OUTPUTS, 'target':I_TARGETS, 'task':'multiclass', 'num_classes':NUM_CLASSES, 'weights':'quadratic'}, 'scalar'),
    'confusionMat': MetricInfo(figures.confusionMat, {'outputs':OUTPUTS, 'i_targets':I_TARGETS, 'num_classes':NUM_CLASSES, 'class_names':CLASS_NAMES}, 'figure'),
    'reliabilityFig': MetricInfo(figures.reliabilityFigure, {'outputs':OUTPUTS, 'i_targets':I_TARGETS, 'num_bins':15, 'graph_type':'line'}, 'figure')
}