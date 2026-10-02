import torch
from torch.utils.data import DataLoader
from glob import glob
import argparse
import numpy as np
import math
from .. import dsets
from random import shuffle

FILE_TYPES = ['jpg','jpeg','png']

#arguments: dset_module path --existing_list=/path/to/list.txt --splits=.7/.15/.15 --balance

parser = argparse.ArgumentParser()
parser.add_argument('dataset_module')
parser.add_argument('dataset_path')
parser.add_argument('--existing_list', default=None)
parser.add_argument('--splits', default=None)
parser.add_argument('--balance', action='store_true', default=False)
args = parser.parse_args()

dset_module = args.dataset_module
dset_path = args.dataset_path
sample_list_path = args.existing_list
splits = args.splits
balance = args.balance

dset_class = dsets
for obj in dset_module.split('.'):
    dset_class = getattr(dset_class, obj)
DsetClass = dset_class.DsetClass

#If a list is provided use that, otherwise find samples
if sample_list_path:
    try:
        with open(sample_list_path) as f:
            sample_list = [line.rstrip() for line in f]
    except:
        raise SystemExit("Couldn't load sample list")
else:
    print('Collecting samples...')
    sample_list = []
    for ext in FILE_TYPES: sample_list += glob(rf'**/*.{ext}', root_dir=dset_path, recursive=True)
    

dataset = DsetClass(dset_path, sample_list, (32,32))
dataloader = DataLoader(dataset, 1, True)

if hasattr(dataset, 'CLASS_NAMES'):
    dset_type = 'classification'
    class_names = dataset.CLASS_NAMES
    num_classes = len(class_names)
else:
    dset_type = 'regression'

if dset_type == 'classification':
    class_lists = []
    for _ in range(num_classes): class_lists.append([])
    
    print('Counting...')    
    dl_iter = iter(dataloader)
    while True:
        try:
            _, label, path = next(dl_iter)
            class_index = torch.argmax(label).item()
            class_lists[class_index].append(path[0])
        except StopIteration:
            break
        except Exception as e:
            print(e)


    #print out the class counts
    for i in range(num_classes):
        print(f'{class_names[i]}: {len(class_lists[i])}')

    if balance:
        print('Balancing...')
        min_count = math.inf
        for cl in class_lists:
            count = len(cl)
            if count < min_count:
                min_count = count

        if min_count == 0:
            raise SystemExit('Minimum class count is 0')

        for i in range(num_classes):
            class_lists[i] = class_lists[i][:min_count]
            print(f'{class_names[i]}: {len(class_lists[i])}')
        
    
    if splits:
        print('Splitting...')
        splits = splits.split('/')
        for i in range(3): splits[i] = float(splits[i])
        if not math.isclose(sum(splits), 1.0):
            raise SystemExit('Splits do not total to 1.0')
        
        train_samples = []
        val_samples = []
        test_samples = []

        for cl in class_lists:
            n_samples = len(cl)
            split_ints = [int(n_samples*splits[0]), int(n_samples*(splits[0]+splits[1]))]
            train, validation, test = np.split(cl, (split_ints[0], split_ints[1]))
            
            train_samples += list(train)
            val_samples += list(validation)
            test_samples += list(test)
        
        print('Saving...')
        with open('train.csv', 'w') as out_file:
            for sample in train_samples: out_file.write(sample+'\n')
        with open('validation.csv', 'w') as out_file:
            for sample in val_samples: out_file.write(sample+'\n')
        with open('test.csv', 'w') as out_file:
            for sample in test_samples: out_file.write(sample+'\n')

    else:
        print('Saving...')
        new_sample_list = []
        for cl in class_lists: new_sample_list += cl

        with open('sample_list.csv', 'w') as out_file:
            for sample in new_sample_list: out_file.write(sample+'\n')

if dset_type == 'regression':
    valid_list = []
    dl_iter = iter(dataloader)
    while True:
        try:
            _, label, path = next(dl_iter)
            valid_list.append((path, label.item()))
        except StopIteration:
            break
        except Exception as e:
            print(e)

    if balance:
        print('Balancing...')
        balanced_list = []
        max_ten_plus = 500
        ten_plus_count = 0

        for (s, l) in valid_list:
            if l >= 10.0:
                if ten_plus_count < max_ten_plus:
                    balanced_list.append(s)
                    ten_plus_count += 1
            else:
                balanced_list.append(s)

        valid_list = balanced_list
    
    if splits:
        print('Splitting...')
        splits = splits.split('/')
        for i in range(3): splits[i] = float(splits[i])
        if not math.isclose(sum(splits), 1.0):
            raise SystemExit('Splits do not total to 1.0')

        shuffle(valid_list)
        split_ints = [len(valid_list)*splits[0], 0]
        split_ints[1] = split_ints[0] + len(valid_list)*splits[1]
        train_samples = valid_list[:split_ints[0]]
        val_samples = valid_list[split_ints[0]:split_ints[1]]
        test_samples = valid_list[split_ints[1]:]

        print('Saving...')
        with open('train.csv', 'w') as out_file:
            for sample in train_samples: out_file.write(sample+'\n')
        with open('validation.csv', 'w') as out_file:
            for sample in val_samples: out_file.write(sample+'\n')
        with open('test.csv', 'w') as out_file:
            for sample in test_samples: out_file.write(sample+'\n')
    else:
        print('Saving...')
        with open('sample_list.csv', 'w') as out_file:
            for sample in valid_list: out_file.write(sample+'\n')


print('Done')