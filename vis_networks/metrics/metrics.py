import torch
from copy import deepcopy
import os.path

class Ref:
    def __init__(self):
        self.value = object()

class MetricInfo():
    def __init__(self, function, params, metric_type):
        self.func = function
        self.params = params
        self.m_type = metric_type

    def deref_params(self, params):
        for k,v in params.items():
            if isinstance(v, Ref):
                params[k] = v.value

    def get_result(self, extra_params={}):
        params = deepcopy(self.params)
        self.deref_params(params)
        combined_params = {**params, **extra_params}
        output = self.func(**combined_params)
        if isinstance(output, torch.Tensor):
            output = output.item()
        
        return (output, self.m_type)

def record_to_writer(metrics, writer, set, epoch):
    try:
        for metric, r in metrics.items():
            value = r[0]
            m_type = r[1]
            if m_type == 'figure':
                writer.add_figure(f'{metric}/{set}', value, epoch)
            elif m_type == 'scalar':
                writer.add_scalar(f'{metric}/{set}', value, epoch)

        if epoch == 1 and set == 'validation':
            multiline_layout = dict()
            for metric, r in metrics.items():
                if r[1] == 'scalar':
                    multiline_layout[metric] = {'Train v. Validation': ['Multiline', [f'{metric}/train', f'{metric}/validation']]}
            writer.add_custom_scalars(multiline_layout)

        writer.flush()
    except Exception as e:
        import warnings
        warnings.warn(f'TensorBoard logging skipped for {set} epoch {epoch}: {e}. '
                      f'Training continues; scalar metrics are still saved to {set}_metrics.csv.')

def record_to_csv(log_dir, metrics, set, epoch):
    file_path = os.path.join(log_dir, f'{set}_metrics.csv')
    csv_file = open(file_path, 'a')

    if epoch == 1:
        csv_file.write('epoch')
        for metric, r in metrics.items():
            m_type = r[1]
            if m_type == 'scalar':
                csv_file.write(f',{metric}')
        csv_file.write('\n')

    csv_file.write(f'{epoch}')
    for metric, r in metrics.items():
        value = r[0]
        m_type = r[1]
        if m_type == 'scalar':
            csv_file.write(f',{value}')
    csv_file.write('\n')
    csv_file.close()