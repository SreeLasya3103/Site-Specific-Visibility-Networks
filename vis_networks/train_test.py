import torch
import os
from os import path
import metrics
from run_set import run_set
import sys

def get_cls_outputs_csv(outputs, targets, paths):
    outputs_str = 'predicted,truth,output,top,path\n'
    sm_output = torch.nn.functional.softmax(outputs, 1)
    for i in range(sm_output.size(0)):
        predicted = torch.argmax(sm_output[i]).item()
        label = torch.argmax(targets[i]).item()
        sm_output_str = [f'{x:.4f}' for x in sm_output[i].tolist()]
        sm_output_str = ', '.join(sm_output_str)
        line = f'{predicted},{label},"[{sm_output_str}]",{sm_output[i][predicted].item()},{paths[i]}\n'
        outputs_str += line
    
    return outputs_str

def get_reg_outputs_csv(outputs, targets, paths):
    outputs_str = 'predicted,truth,path\n'
    for i in range(outputs.size(0)):
        line = f'{outputs[i].item()},{targets[i].item()},{paths[i]}\n'
        outputs_str += line
    
    return outputs_str

def train(loaders, model, optimizer, loss_fn, epochs, metrics_list, device, use_amp, output_fn, labels_fn, writer, scheduler=None):
    model.to(device)
    scaler = torch.amp.grad_scaler.GradScaler(device, enabled=use_amp)

    class_names = None
    num_classes = 1
    if(hasattr(loaders[0].dataset, 'CLASS_NAMES')):
        class_names = loaders[0].dataset.CLASS_NAMES
        num_classes = len(class_names)

    best_loss = float('inf')
    best_epoch = -1

    for epoch in range(1, epochs+1):
        print('\nEpoch ' + str(epoch))
        
        print('Training...')
        loader = loaders[0]
        model.train()
        epoch_results = run_set(loader, model, optimizer, loss_fn, device, scaler, output_fn, labels_fn)

        all_outputs = epoch_results[0].cpu()
        all_labels = epoch_results[1].cpu()
        running_loss = epoch_results[3]

        print('\nTraining loss: ' + str(running_loss))
        if num_classes > 1:
            train_met = metrics.multiclass.get_metrics(all_outputs, all_labels, class_names, metrics_list)
            if 'accuracy' in train_met.keys():
                print('Training accuracy: ' + str(train_met['accuracy'][0]))
        else:
            train_met = metrics.regression.get_metrics(all_outputs, all_labels, metrics_list)
            if 'MAE' in train_met.keys():
                print('Training MAE: ' + str(train_met['MAE'][0]))
        
        train_met['loss'] = (running_loss, 'scalar')
        metrics.record_to_writer(train_met, writer, 'train', epoch)
        metrics.record_to_csv(writer.get_logdir(), train_met, 'train', epoch)

        print('Validating...')
        loader = loaders[1]
        with torch.inference_mode():
            model.eval()
            epoch_results = run_set(loader, model, None, loss_fn, device, scaler, output_fn, labels_fn)
            sample_input = next(iter(loader))[0][0].unsqueeze(0)
            sample_input = sample_input.to(device)
            sample_output = model(sample_input)
            sample_out_tensor = sample_output[0] if isinstance(sample_output, tuple) else sample_output
            model.register_buffer('example_input', sample_input)
            model.register_buffer('example_output', sample_out_tensor)


        all_outputs = epoch_results[0].cpu()
        all_labels = epoch_results[1].cpu()
        all_paths = epoch_results[2]
        running_loss = epoch_results[3]
        
        print('\nValidation loss: ' + str(running_loss))
        if num_classes > 1:
            val_met = metrics.multiclass.get_metrics(all_outputs, all_labels, class_names, metrics_list)
            if 'accuracy' in val_met.keys():
                print('Validation accuracy: ' + str(val_met['accuracy'][0]))
        else:
            val_met = metrics.regression.get_metrics(all_outputs, all_labels, metrics_list)
            if 'MAE' in val_met.keys():
                print('Validation MAE: ' + str(val_met['MAE'][0]))
        
        
        val_met['loss'] = (running_loss, 'scalar')
        metrics.record_to_writer(val_met, writer, 'validation', epoch)
        metrics.record_to_csv(writer.get_logdir(), val_met, 'validation', epoch)

        old_last = path.join(writer.get_logdir(), f'epoch{epoch-1}.pt')
        if path.exists(old_last):
            os.remove(old_last)
        new_last = path.join(writer.get_logdir(), f'epoch{epoch}.pt')
        torch.save(model.state_dict(), new_last)

        if running_loss < best_loss:
            best_loss = running_loss
            try:
                os.remove(path.join(writer.get_logdir(), f'best_epoch{best_epoch}.pt'))
            except:
                pass

            best_epoch = epoch
            torch.save(model.state_dict(), path.join(writer.get_logdir(), f'best_epoch{best_epoch}.pt'))

            if num_classes > 1:
                outputs_csv = get_cls_outputs_csv(all_outputs, all_labels, all_paths)
            else:
                outputs_csv = get_reg_outputs_csv(all_outputs, all_labels, all_paths)

            with open(path.join(writer.get_logdir(), 'outputs.csv'), 'w') as out_file:
                out_file.write(outputs_csv)

            if scheduler is not None:
                scheduler.step(running_loss)

        writer.flush()

    writer.close()

def test(loader, model, loss_fn, metrics_list, device, use_amp, output_fn, labels_fn, writer):
    model.to(device)
    scaler = torch.amp.grad_scaler.GradScaler(device, enabled=use_amp)

    class_names = None
    num_classes = 1
    if(hasattr(loader.dataset, 'CLASS_NAMES')):
        class_names = loader.dataset.CLASS_NAMES
        num_classes = len(class_names)

    print('Testing...')
    with torch.inference_mode():
        model.eval()
        results = run_set(loader, model, None, loss_fn, device, scaler, output_fn, labels_fn)

    all_outputs = results[0].cpu()
    all_labels = results[1].cpu()
    all_paths = results[2]
    running_loss = results[3]
    
    print('\nTesting loss: ' + str(running_loss))
    if num_classes > 1:
        test_met = metrics.multiclass.get_metrics(all_outputs, all_labels, class_names, metrics_list)
        if 'accuracy' in test_met.keys():
                print('Test accuracy: ' + str(test_met['accuracy'][0]))
    else:
        test_met = metrics.regression.get_metrics(all_outputs, all_labels, metrics_list)
        if 'MAE' in test_met.keys():
            print('Test MAE: ' + str(test_met['MAE'][0]))
    
    test_met['loss'] = (running_loss, 'scalar')
    metrics.record_to_writer(test_met, writer, 'test', 1)
    metrics.record_to_csv(writer.get_logdir(), test_met, 'test', 1)

    if num_classes > 1:
        outputs_csv = get_cls_outputs_csv(all_outputs, all_labels, all_paths)
    else:
        outputs_csv = get_reg_outputs_csv(all_outputs, all_labels, all_paths)

    with open(path.join(writer.get_logdir(), 'outputs.csv'), 'w') as out_file:
        out_file.write(outputs_csv)

    writer.flush()

    writer.close()