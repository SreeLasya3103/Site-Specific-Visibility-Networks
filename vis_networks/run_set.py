import torch
from progress.bar import ChargingBar
from progress.spinner import Spinner

def run_set(loader, model, optimizer, loss_fn, device, scaler, output_fn, labels_fn):
    all_outputs = None
    all_labels = None
    all_paths = []
    running_loss = 0.0

    bar = ChargingBar()
    bar.max = len(loader)
    bar.width = 0
    spinner = Spinner()

    for step, (data, labels, paths) in enumerate(loader):
        data = data.to(device)
        labels = labels.to(device)

        labels = labels_fn(labels)
        with torch.autocast(device, torch.bfloat16, enabled=scaler.is_enabled()):
            output = model(data)
            output = output_fn(output)
            loss = loss_fn(output, labels)

        if loader.drop_last:
            running_loss += loss.item() / len(loader)
        else:
            running_loss += loss.item()*data.size(0) / len(loader.dataset)
        
        if optimizer:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        out_for_metrics = output[0].detach().clone() if isinstance(output, tuple) else output.detach().clone()
        if step == 0:
            all_outputs = out_for_metrics
            all_labels = labels.detach().clone()
            all_paths += paths
        else:
            all_outputs = torch.cat((all_outputs, out_for_metrics), 0)
            all_labels = torch.cat((all_labels, labels.detach().clone()), 0)
            all_paths += paths

        
        bar.next()
        spinner.next()
    
    return (all_outputs, all_labels, all_paths, running_loss)