import torch
from torch.utils.data import DataLoader
import os
import importlib.util
import shutil
import sys
from glob import glob
from torch.utils.tensorboard.writer import SummaryWriter
import train_test
import torchvision.transforms as tf
import progress
from types import SimpleNamespace
import warnings

def load_sample_lists(config):
    cf = config
    DsetClass = cf.dset_module.DsetClass

    if cf.splits_path:
        train_list = os.path.join(cf.splits_path, 'train.csv')
        val_list = os.path.join(cf.splits_path, 'validation.csv')
        test_list = os.path.join(cf.splits_path, 'test.csv')

        try:
            with open(train_list) as f:
                train_samples = [line.rstrip() for line in f]
            with open(val_list) as f:
                val_samples = [line.rstrip() for line in f]
            with open(test_list) as f:
                test_samples = [line.rstrip() for line in f]
        except:
            raise SystemExit("Couldn't load all split files")
    #Otherwise look for folders with set names and try to load them
    else:
        train_dir = os.path.join(cf.dset_path, 'train')
        val_dir = os.path.join(cf.dset_path, 'validation')
        test_dir = os.path.join(cf.dset_path, 'test')

        try:
            for ext in DsetClass.FILE_TYPES:
                train_samples += glob(rf'**/*.{ext}', root_dir=train_dir, recursive=True)
                val_samples += glob(rf'**/*.{ext}', root_dir=val_dir, recursive=True)
                test_samples += glob(rf'**/*.{ext}', root_dir=test_dir, recursive=True)
        except:
            raise SystemExit("Couldn't load all split folders")
    
    return train_samples, val_samples, test_samples

def create_dsets(config, sample_lists):
    cf = config
    DsetClass = cf.dset_module.DsetClass

    dim = cf.dset_params.pop('dim')
    n_channels = cf.dset_params.pop('n_channels')

    augmenter = lambda x: x
    if not augmenter is None:
        augmenter = tf.Compose(cf.augment_list)

    transformer = lambda x: x
    if hasattr(cf.model_module, 'CUSTOM_TRANSFORM') and not cf.model_module.CUSTOM_TRANSFORM is None:
        transformer = cf.model_module.CUSTOM_TRANSFORM(**cf.transform_params)

    train_transformer = lambda x: transformer(augmenter(x))
    
    #Create the sets and loaders from the file lists
    train_set = DsetClass(cf.dset_path, sample_lists[0], dim, n_channels, train_transformer, **cf.dset_params)
    no_aug_train_set = DsetClass(cf.dset_path, sample_lists[0], dim, n_channels, transformer, **cf.dset_params)
    val_set = DsetClass(cf.dset_path, sample_lists[1], dim, n_channels, transformer, **cf.dset_params)
    test_set = DsetClass(cf.dset_path, sample_lists[2], dim, n_channels, transformer, **cf.dset_params)

    return (train_set, no_aug_train_set, val_set, test_set)

def prepare_dataloaders(config, split_sets):
    cf = config
    num_workers = cf.n_workers
    if os.name == 'nt':
        num_workers = 0


    train_loader = DataLoader(split_sets[0], cf.batch_size, True, num_workers=num_workers, pin_memory=True, drop_last=True)
    no_aug_train_loader = DataLoader(split_sets[1], cf.batch_size, cf.batch_size, num_workers=num_workers)
    val_loader = DataLoader(split_sets[2], cf.batch_size, True, num_workers=num_workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(split_sets[3], cf.batch_size, True, num_workers=num_workers, pin_memory=True, drop_last=False)

    return train_loader, no_aug_train_loader, val_loader, test_loader

def get_mean_std(dataloader):
    sample = dataloader.dataset.__getitem__(0)[0]
    dims = sample.size()

    print('Calculating mean...')
    bar = progress.bar.ChargingBar()
    bar.max = len(dataloader)
    bar.width = 0
    spinner = progress.spinner.Spinner()

    running_mean = torch.zeros(dims[:-2])

    for data, _, _ in dataloader:
        batch_mean = torch.mean(data, (0, -2, -1))
        running_mean += batch_mean*data.size(0)

        bar.next()
        spinner.next()

    n = len(dataloader.dataset)
    running_mean = running_mean / n

    print('\nCalculating standard deviation...')
    bar = progress.bar.ChargingBar()
    bar.max = len(dataloader)
    bar.width = 0
    spinner = progress.spinner.Spinner()

    running_variance = torch.zeros(dims[:-2])

    for data, _, _ in dataloader:
        dev = data - running_mean.view(1, *running_mean.size(), 1, 1)
        sqr_dev = torch.square(dev)
        running_variance += torch.sum(sqr_dev, (0,-2,-1)) / (n*dims[-2]*dims[-1] - 1)

        bar.next()
        spinner.next()

    std = torch.sqrt(running_variance)

    return (running_mean, std)

def prepare_model(config, loader):
    cf = config

    ModelClass = cf.model_module.Model
    DsetClass = cf.dset_module.DsetClass

    num_classes = 1
    if(hasattr(DsetClass, 'CLASS_NAMES')):
        num_classes = len(DsetClass.CLASS_NAMES)

    sample = next(iter(loader))[0][0]
    mean = None
    std = None

    #If the an existing model isn't specified, create a new one and set its mean and std accordingly
    if not cf.existing_model:
        if cf.normalize and not cf.test_only:
            mean, std = get_mean_std(loader)

        model = ModelClass(num_classes, sample.size(), mean, std, **cf.model_params)
        model(sample.unsqueeze(0))
    #If an existing model is specified, create a new one and load the existing one's weights
    else:
        model = ModelClass(num_classes, sample.size(), **cf.model_params)
        state_dict = torch.load(cf.existing_model, weights_only=True, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, False)
        model.register_buffer('example_input', state_dict['example_input'])
        model.register_buffer('example_output', state_dict['example_output'])

    return model
    
def main():
    #Import the config.py file where settings for training are
    spec = importlib.util.spec_from_file_location("config", os.path.join(os.getcwd(), 'config.py'))
    config = importlib.util.module_from_spec(spec)
    config.__package__ = 'vis_networks'
    sys.modules["config"] = config
    spec.loader.exec_module(config)
    config = config.CONFIG
    cf = SimpleNamespace(**config)

    #Initalize summary writer for tensorboard    
    if cf.test_only:
        writer = SummaryWriter('./test_results')
    else:
        writer = SummaryWriter()
    log_dir = writer.get_logdir()

    #Copy config to log dir
    config_path = os.path.join(os.getcwd(), 'config.py')
    shutil.copyfile(config_path, os.path.join(log_dir, 'config.py'))
    
    print('Preparing dataset...')
    sample_lists = load_sample_lists(cf)
    dsets = create_dsets(cf, sample_lists)
    train_loader, noaug_train_loader, val_loader, test_loader = prepare_dataloaders(cf, dsets)

    #Record the split lists to the log directory
    test_list = os.path.join(log_dir, 'test.csv')
    train_list = os.path.join(log_dir, 'train.csv')
    val_list = os.path.join(log_dir, 'validation.csv')
    with open(train_list, 'w') as out_file:
        for sample in sample_lists[0]: out_file.write(sample+'\n')
    with open(val_list, 'w') as out_file:
        for sample in sample_lists[1]: out_file.write(sample+'\n')
    with open(test_list, 'w') as out_file:
        for sample in sample_lists[2]: out_file.write(sample+'\n')

    print('Preparing model...')
    model = prepare_model(cf, noaug_train_loader)

    if cf.output_func is None:
        cf.output_func = lambda x: x
    if cf.label_func is None:
        cf.label_func = lambda x: x

    device = 'cpu'
    if cf.cuda:
        device = 'cuda'
    
    model.to(device)

    if not cf.test_only:
        model.train()
        optimizer = cf.OptimizerClass(model.parameters(), **cf.optimizer_params)
        scheduler = None
        if hasattr(cf, 'scheduler_class') and cf.scheduler_class is not None:
            scheduler = cf.scheduler_class(optimizer, **cf.scheduler_params)
        train_test.train((train_loader, val_loader), model, optimizer, cf.loss_func, cf.epochs, cf.metrics, device, cf.use_amp, cf.output_func, cf.label_func, writer, scheduler=scheduler)
    else:
        train_test.test(test_loader, model, cf.loss_func, cf.metrics, device, False, cf.output_func, cf.label_func, writer)

if __name__=='__main__':
    main()