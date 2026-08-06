from torch.utils.data import Dataset
from os import path
import os
import torch
from torchvision import io
from . import common as cmmn
from ..img_util import get_resize_crop_fn


class DsetClass(Dataset):
    FILE_TYPES = ('jpg', 'jpeg', 'png')
    DATA_TYPES = ('image',)
    LABEL_TYPES = ('onehot',)
    LABEL_NAMES = ('Visibility',)

    CLASS_NAMES = ['1.0', '2.0', '3.0', '4.0', '5.0',
                   '6.0', '7.0', '8.0', '9.0', '10.0']
    CLASS_GROUPS = [{0.0, 1.0, 1.25, 1.5, 1.75}, {2.0, 2.25, 2.5},
                    {3.0}, {4.0}, {5.0}, {6.0}, {7.0}, {8.0}, {9.0}, {10.0}]

    def __init__(self, dataset_dir, sample_list, dim, n_channels=3,
                 transformer=None, *, crop=True, ref_dir=None):
        self.dset_dir = dataset_dir
        self.sample_list = sample_list
        self.labels = cmmn.get_onehot_labels(sample_list, self.CLASS_GROUPS)
        self.resize = get_resize_crop_fn(dim)
        self.transformer = transformer
        self.crop = crop
        self.ref_dir = ref_dir
        self.dim = dim
        self.ref_cache = {}

        i = 0
        while i < len(self.labels):
            if self.labels[i] is None:
                del self.sample_list[i]
                del self.labels[i]
            else:
                i += 1

    def _get_site_id(self, sample_path):
        return os.path.basename(sample_path).split('_')[0]

    def _load_reference(self, site_id):
        if site_id in self.ref_cache:
            return self.ref_cache[site_id]

        if self.ref_dir is None:
            self.ref_cache[site_id] = None
            return None

        for ext in ['jpg', 'jpeg', 'png']:
            ref_path = os.path.join(self.ref_dir, f"{site_id}_reference.{ext}")
            if os.path.exists(ref_path):
                ref = io.decode_image(ref_path, 'RGB') / 255.0
                if self.crop:
                    ref = cmmn.crop_margins(ref)
                ref = self.resize(ref)
                self.ref_cache[site_id] = ref
                return ref

        self.ref_cache[site_id] = None
        return None

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        rel_sample_path = self.sample_list[idx]
        sample_path = path.join(self.dset_dir, rel_sample_path)
        sample_data = io.decode_image(sample_path, 'RGB') / 255.0

        if self.crop:
            sample_data = cmmn.crop_margins(sample_data)
        sample_data = self.resize(sample_data)

        site_id = self._get_site_id(rel_sample_path)
        ref_img = self._load_reference(site_id)

        if self.transformer:
            current_transformed = self.transformer(sample_data)
            if ref_img is not None:
                ref_transformed = self.transformer(ref_img)
            else:
                ref_transformed = current_transformed.clone()
        else:
            current_transformed = sample_data
            if ref_img is not None:
                ref_transformed = ref_img
            else:
                ref_transformed = current_transformed.clone()

        # Stack as [2, 3, 3, H, W] (with VisNet transform) or [2, 3, H, W] (without)
        paired = torch.stack([current_transformed, ref_transformed], dim=0)

        label = self.labels[idx]
        return (paired, label, rel_sample_path)
