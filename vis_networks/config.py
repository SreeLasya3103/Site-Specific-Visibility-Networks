import os
import torch
import torchvision.transforms as tf
from models.visnet import visnet_similarity
from dsets.webcams import cls_10full_dual
from losses.similarity import SimilarityVisibilityLoss

CONFIG = {
    'model_module': visnet_similarity,
    'model_params': {
        #The pooled VisNet checkpoint behind the VisNet "Pooled" row of Table 1, the only best-epoch file in runs\VisNetGlobal.
        'pretrained_path': 'D:\\Research - Lasya\\Site-Specific-Visibility-Networks\\vis_networks\\runs\\VisNetGlobal\\best_epoch19.pt',
    },
    'transform_params': { 'dim': (280, 280) },
    'existing_model': None,
    'test_only': False,
    'use_amp': False,
    'dset_module': cls_10full_dual,
    'dset_params': {
        'dim': (280,280), 'n_channels': 3, 'crop': True,
        # One reference per camera view, written by the new select_references.py.
        'ref_dir': 'D:\\Research - Lasya\\NewWebcams\\references_by_view',
    },
    'dset_path': 'D:\\Research - Lasya\\NewWebcams',
    # The pooled split minus the two images whose camera views have no reference; neither is in a per-site run or in Table 1's evaluation set.
    'splits_path': 'D:\\Research - Lasya\\NewWebcams\\splits_similarity',
    'splits_path': 'D:\\Research - Lasya\\NewWebcams\\splits_similarity',
    # Photometric augmentation only, applied to the current image only (the dataset
    # never augments the reference). Flips and rotations are left out because they
    # would move the current image out of register with its reference.
    'augment_list': [
        tf.ColorJitter(brightness=0.2, contrast=0.2),
        tf.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
    ],
    # The frozen backbone must see inputs normalised exactly as during its own
    # training, and its checkpoint carries those mean/std buffers. Recomputing them
    # here would cost two full passes over the paired dataset and then be discarded.
    'normalize': False,
    'metrics': {},
    'cuda': True,
    'epochs': 80,
    'batch_size': 16,
    'loss_func': SimilarityVisibilityLoss(),
    'OptimizerClass': torch.optim.Adam,
    'optimizer_params': { 'lr': 1e-3 },
    'output_func': None,
    'label_func': None,
    'n_workers': 4,
}

# Stop immediately if either path is wrong, instead of after hours of training.
assert os.path.isfile(CONFIG['model_params']['pretrained_path']), \
    f"Backbone checkpoint not found: {CONFIG['model_params']['pretrained_path']}"
assert os.path.isdir(CONFIG['dset_params']['ref_dir']), \
    f"Reference folder not found: {CONFIG['dset_params']['ref_dir']}"
