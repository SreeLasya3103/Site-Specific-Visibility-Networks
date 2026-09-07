import torch
from torch.utils.data import Dataset
from os import path
from torchvision import io
from ..img_util import get_resize_crop_fn
import csv

def get_sample_label_pairs(labels_path):
    with open(labels_path) as f:
        reader = csv.reader(f)
        data = list(reader)

    pairs = { row[8]+'.jpg':torch.tensor(float(row[7])) for row in data }
    return pairs

class DsetClass(Dataset):
    FILE_TYPES = ('jpg')
    DATA_TYPES = ('image',)
    LABEL_TYPES = ('scalar',)
    LABEL_NAMES = ('Visibility',)

    def __init__(self, dataset_dir, sample_list, dim, n_channels=3, transformer=None):
        self.dset_dir = dataset_dir
        self.sample_list = sample_list
        self.resize = get_resize_crop_fn(dim)
        self.transformer = transformer

        labels_path = path.join(dataset_dir, 'label.csv')
        sample_label_pairs = get_sample_label_pairs(labels_path)
        
        self.labels = []
        for sample in self.sample_list:
            self.labels.append(sample_label_pairs[sample])
        
    def __len__(self):
        return len(self.sample_list)
    
    def __getitem__(self, idx):
        rel_sample_path = self.sample_list[idx]
        sample_path = path.join(self.dset_dir, rel_sample_path)
        sample_data = io.decode_image(sample_path, 'RGB')/255.0

        sample_data = self.resize(sample_data)

        if self.transformer:
            sample_data = self.transformer(sample_data)

        label = self.labels[idx]

        return (sample_data, label, rel_sample_path)