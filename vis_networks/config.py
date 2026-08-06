import torch
import torch.nn as nn
from dsets import *
from models import *
from torch.optim import lr_scheduler as sched
from pytorch_forecasting.metrics.point import MAPE, SMAPE
from torchmetrics.regression import RelativeSquaredError as RSE
import torchvision.transforms as tf
from models.visnet import visnet_similarity
from dsets.webcams import cls_10full_dual
from losses.similarity import SimilarityVisibilityLoss

CONFIG = {
    'model_module': visnet_similarity,
    'model_params': {
        'pretrained_visnet_path': 'D:\\Research - Lasya\\Visibility-Networks-Site-Specific\\vis_networks\\runs\Visnet Global\\best_epoch30.pt',
    },
    'transform_params': { 'dim': (280, 280) },
    'existing_model': None,
    'test_only': False,
    'use_amp': False,
    'dset_module': cls_10full_dual,
    'dset_params': {
        'dim': (280,280), 'n_channels': 3, 'crop': True,
        'ref_dir': 'D:\\Research - Lasya\\NewWebcams\\references',
    },
    'dset_path': 'D:\\Research - Lasya\\NewWebcams',
    'splits_path': 'D:\\Research - Lasya\\NewWebcams',
    'augment_list': [
        tf.RandomHorizontalFlip(0.5),
        tf.RandomRotation(5),
        tf.ColorJitter(brightness=0.2, contrast=0.2),
        tf.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
    ],
    'normalize': True,
    'metrics': {},
    'cuda': True,
    'epochs': 80,
    'batch_size': 16,
    'loss_func': SimilarityVisibilityLoss(),
    'OptimizerClass': torch.optim.Adam,
    'optimizer_params': { 'lr': 1e-3 },
    #'scheduler_class': sched.ReduceLROnPlateau,
    #'scheduler_params': {'mode': 'min', 'factor': 0.5, 'patience': 10, 'min_lr': 1e-6},
    'output_func': None,
    'label_func': None,
    'n_workers': 4,
}
