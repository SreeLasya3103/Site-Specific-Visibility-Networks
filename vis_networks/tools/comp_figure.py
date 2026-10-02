import argparse
from torch.utils.data import DataLoader
import torch
from torchvision import io
import torchvision.transforms as tf
from ast import literal_eval
import importlib
import sys
import os
from glob import glob

#arguments: config_path figure_dims --augment=0 --subimage=index --class_index=index

parser = argparse.ArgumentParser()
parser.add_argument('config_path')
parser.add_argument('figure_dims')
parser.add_argument('--augment', default='None')
parser.add_argument('--subimage', default='None')
parser.add_argument('--class_index', default='None')
args = parser.parse_args()

config_path = args.config_path
figure_dims = literal_eval(args.figure_dims)
do_augment = literal_eval(args.augment)
subimage_index = literal_eval(args.subimage)
class_index = literal_eval(args.class_index)

spec = importlib.util.spec_from_file_location("config", config_path)
config = importlib.util.module_from_spec(spec)
config.__package__ = 'vis_networks'
sys.modules["config"] = config
spec.loader.exec_module(config)
CONFIG = config.CONFIG

model_module = CONFIG['model_module']
model_params = CONFIG['model_params']
transform_params = CONFIG['transform_params']

dset_module = CONFIG['dset_module']
dset_params = CONFIG['dset_params']
dset_path = CONFIG['dset_path']
splits_path = CONFIG['splits_path']
augment_list = CONFIG['augment_list']

dim = dset_params.pop('dim')
n_channels = dset_params.pop('n_channels')

n_workers = CONFIG['n_workers']

DsetClass = dset_module.DsetClass
n_classes = 1
if(hasattr(DsetClass, 'CLASS_NAMES')):
    n_classes = len(DsetClass.CLASS_NAMES)

augmenter = lambda x: x
if not augmenter is None:
    augmenter = tf.Compose(augment_list)

transformer = lambda x: x
if hasattr(model_module, 'CUSTOM_TRANSFORM') and not model_module.CUSTOM_TRANSFORM is None:
    transformer = model_module.CUSTOM_TRANSFORM(**transform_params)

train_transformer = lambda x: transformer(augmenter(x))

train_samples = []

if splits_path:
    train_list = os.path.join(splits_path, 'train.csv')

    try:
        with open(train_list) as f:
            train_samples = [line.rstrip() for line in f]
    except:
        raise SystemExit("Couldn't load all split files")
#Otherwise look for folders with set names and try to load them
else:
    train_dir = os.path.join(dset_path, 'train')

    try:
        for ext in DsetClass.FILE_TYPES:
            train_samples += glob(rf'**/*.{ext}', root_dir=train_dir, recursive=True)
    except:
        raise SystemExit("Couldn't load all split folders")
if do_augment:
    t = train_transformer
else:
    t = transformer
dataset = DsetClass(dset_path, train_samples, dim, n_channels, t, **dset_params)
dataloader = DataLoader(dataset, 1, True, num_workers=n_workers)

if hasattr(dataset, 'CLASS_NAMES'):
    dset_type = 'classification'
    num_classes = len(dataset.CLASS_NAMES)
else:
    dset_type = 'regression'

figure_list = []

loader_iter = iter(dataloader)

for y in range(figure_dims[0]):
    row_list = []
    for x in range(figure_dims[1]):
        while True:
            try:
                data, label, path = next(loader_iter)
            except StopIteration:
                raise SystemExit('Not enough images for figure')
            except Exception as e:
                print(e)
                
            if class_index is None or torch.argmax(label[0]) == class_index:
                break
        
        data = data[0]
        if len(data.size()) == 3:
            data = data.unsqueeze(0)
        images = data.split(1)
        if subimage_index is None:
            composite = torch.cat(images, -2)
            img = composite[0]
        else:
            img = images[subimage_index][0]

        row_list.append(img)
    
    row = torch.cat(row_list, 2)
    figure_list.append(row)

figure = torch.cat(figure_list, 1)
figure = torch.round(figure*255.0)
figure = torch.clamp(figure, 0.0, 255.0)
figure = figure.to(torch.uint8)

io.write_png(figure, 'figure.png')