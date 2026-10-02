from torch.utils.data import Dataset
from os import path
import os
import torch
from torchvision import io
from . import common as cmmn
from ..img_util import get_resize_crop_fn

_REF_TRANSFORMS = {}


def _visnet_transform(dim):
    #The plain, never-augmented VisNet transform, built once per image size.
    if dim not in _REF_TRANSFORMS:
        from models.visnet.transform import get_transformer   # lazy: avoids a dsets/models import cycle
        _REF_TRANSFORMS[dim] = get_transformer(dim)
    return _REF_TRANSFORMS[dim]


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
                 transformer=None, *, crop=True, ref_dir=None, ref_transform='visnet'):
        if ref_dir is None:
            raise ValueError("cls_10full_dual needs dset_params['ref_dir']: without "
                             "references there is nothing to compare against.")
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

        # The reference always gets the plain transform, never the augmenting
        # one the current image may be given during training.
        if ref_transform == 'visnet':
            self.ref_transformer = _visnet_transform(tuple(dim))
        elif ref_transform is None:
            self.ref_transformer = lambda x: x
        else:
            raise ValueError(f"Unknown ref_transform: {ref_transform!r}")

        # Fail now, not mid-epoch, if any camera view lacks a reference.
        missing = sorted({v for v in map(self._get_view_id, self.sample_list)
                          if self._reference_path(v) is None})
        if missing:
            raise FileNotFoundError(f"{len(missing)} camera view(s) have no reference in "
                                    f"{ref_dir}, e.g. {missing[:5]}. Run select_references.py.")

    def _get_view_id(self, sample_path):
        # SITE508_ORNT315_VIS1-25mi.png -> SITE508_ORNT315. Each orientation is a
        # different camera and a different scene, so the reference has to come
        # from the same view, not just the same site.
        return '_'.join(os.path.basename(sample_path).split('_')[:2])

    def _reference_path(self, view_id):
        for ext in ('jpg', 'jpeg', 'png'):
            ref_path = os.path.join(self.ref_dir, f"{view_id}_reference.{ext}")
            if os.path.exists(ref_path):
                return ref_path
        return None

    def _load_reference(self, view_id):
        """The view's reference: cropped, resized, and through the plain transform."""
        if view_id not in self.ref_cache:
            ref_path = self._reference_path(view_id)
            if ref_path is None:
                # No silent fallback: comparing an image with itself reads as
                # "identical to a clear day".
                raise FileNotFoundError(f"No reference for {view_id} in {self.ref_dir}")
            ref = io.decode_image(ref_path, 'RGB') / 255.0
            if self.crop:
                ref = cmmn.crop_margins(ref)
            self.ref_cache[view_id] = self.ref_transformer(self.resize(ref))
        return self.ref_cache[view_id]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        rel_sample_path = self.sample_list[idx]
        sample_path = path.join(self.dset_dir, rel_sample_path)
        sample_data = io.decode_image(sample_path, 'RGB') / 255.0

        if self.crop:
            sample_data = cmmn.crop_margins(sample_data)
        sample_data = self.resize(sample_data)

        # The current image gets the transformer it was given (the augmenting one
        # during training); the reference never does.
        current_transformed = self.transformer(sample_data) if self.transformer else sample_data
        ref_transformed = self._load_reference(self._get_view_id(rel_sample_path))

        # Stack as [2, 3, 3, H, W] (with VisNet transform) or [2, 3, H, W] (without)
        paired = torch.stack([current_transformed, ref_transformed], dim=0)

        label = self.labels[idx]
        return (paired, label, rel_sample_path)
