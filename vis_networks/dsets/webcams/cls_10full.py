from torch.utils.data import Dataset
from os import path
from torchvision import io
from . import common as cmmn
from ..img_util import get_resize_crop_fn

class DsetClass(Dataset):
    FILE_TYPES = ('jpg','jpeg','png')
    DATA_TYPES = ('image',)
    LABEL_TYPES = ('onehot',)
    LABEL_NAMES = ('Visibility',)
    
    CLASS_NAMES = ['1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0', '10.0']
    
    CLASS_GROUPS = [{0.0, 1.0, 1.25, 1.5, 1.75}, {2.0, 2.25, 2.5}, {3.0}, {4.0}, {5.0}, {6.0}, {7.0}, {8.0}, {9.0}, {10.0}]

    def __init__(self, dataset_dir, sample_list, dim, n_channels=3, transformer=None, *, crop=True):
        self.dset_dir = dataset_dir
        self.sample_list = sample_list
        self.labels = cmmn.get_onehot_labels(sample_list, self.CLASS_GROUPS)
        self.resize = get_resize_crop_fn(dim)
        self.transformer = transformer
        self.crop = crop

        i = 0
        while i < len(self.labels):
            if self.labels[i] is None:
                del self.sample_list[i]
                del self.labels[i]
            else:
                i += 1
                
    def __len__(self):
        return len(self.sample_list)
    
    def __getitem__(self, idx):
        rel_sample_path = self.sample_list[idx]
        sample_path = path.join(self.dset_dir, rel_sample_path)
        sample_data = io.decode_image(sample_path, 'RGB')/255.0

        if self.crop:
            sample_data = cmmn.crop_margins(sample_data)
        
        sample_data = self.resize(sample_data)

        if self.transformer:
            sample_data = self.transformer(sample_data)

        label = self.labels[idx]

        return (sample_data, label, rel_sample_path)