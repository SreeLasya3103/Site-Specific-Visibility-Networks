import torch
from torch.utils.data import Dataset, DataLoader
import torchvision as tv
import torchvision.io as io
import os
from os import path
import random
from math import ceil
import torchvision.transforms.functional as tff
import torchvision.transforms as tf
import sys
import pathlib
import numpy as np
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.resolve()))
import torcheval.metrics.functional as temf
from ....models import *

from ...img_util import get_resize_crop_fn

DSET_DIR = '/home/feet/Documents/ResearchRedux/datasets/Webcams/Webcams/'
MODEL_PATH = '/home/feet/Documents/ResearchRedux/datasets/Webcams/goodbadlists/goodbad-bestloss.pt'
MANUAL_DIR = '/home/feet/Documents/ResearchRedux/datasets/Webcams/goodbadlists/'
ALL_SAMPLES = '/home/feet/Documents/ResearchRedux/datasets/Webcams/goodbadlists/all.csv'

SPLIT = .80
USE_CUDA = True
EPOCHS = 120
BATCH_SIZE = 32
LR = 0.000001
IMG_DIM = (310,468)
MAX_GOOD = 406
MAX_BAD = 406
NUM_WORKERS = 4

MAX_IMG_SIZE = (310, 468)

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def crop_margins(sample_data):
    orig_dim = sample_data.size()
    #Remove 12.81% top, 4 bottom, 4 left/right
    top = ceil(0.1281 * orig_dim[-2])
    bot = 4
    left = 4
    right = 4
    height = orig_dim[-2] - (top + bot)
    width = orig_dim[-1] - (left + right)

    return tff.resized_crop(sample_data, top, left, height, width, MAX_IMG_SIZE, tf.InterpolationMode.BICUBIC)

class GoodBadWebcams(Dataset):
    def __init__(self, dataset_dir, sample_list, labels, dim, transformer=lambda x:x, *, crop=True):
        self.dset_dir = dataset_dir
        self.sample_list = sample_list
        self.labels = labels
        self.resize = get_resize_crop_fn(dim)
        self.crop = crop
        self.transformer = transformer

    def __len__(self):
        return len(self.sample_list)
    
    def __getitem__(self, idx):
        rel_sample_path = self.sample_list[idx]
        sample_path = path.join(self.dset_dir, rel_sample_path)
        sample_data = io.decode_image(sample_path, 'RGB')/255.0

        if self.crop:
            sample_data = crop_margins(sample_data)
        
        sample_data = self.resize(sample_data)

        if self.transformer:
            sample_data = self.transformer(sample_data)

        label = self.labels[idx]

        return (sample_data, label, rel_sample_path)

def train():
    print('train_loss,train_acc,val_loss,val_acc,bad_acc,good_acc')

    model = tv.models.resnet34(num_classes=1)
    if USE_CUDA:
        model.cuda()
    
    good_list_path = os.path.join(MANUAL_DIR, 'good.csv')
    with open(good_list_path) as f:
        good_list = [line.rstrip() for line in f]
    bad_list_path = os.path.join(MANUAL_DIR, 'bad.csv')
    with open(bad_list_path) as f:
        bad_list = [line.rstrip() for line in f]
    
    random.shuffle(good_list)
    random.shuffle(bad_list)
    good_list = good_list[:MAX_GOOD]
    bad_list = bad_list[:MAX_BAD]

    split_int = int(len(good_list)*SPLIT)
    train_good = good_list[:split_int]
    val_good = good_list[split_int:]

    split_int = int(len(bad_list)*SPLIT)
    train_bad = bad_list[:split_int]
    val_bad = bad_list[split_int:]

    train_list = train_good + train_bad
    train_labels = [torch.tensor([1.0]) for _ in range(len(train_good))] + [torch.tensor([0.0]) for _ in range(len(train_bad))]

    val_list = val_good + val_bad
    val_labels = [torch.tensor([1.0]) for _ in range(len(val_good))] + [torch.tensor([0.0]) for _ in range(len(val_bad))]

    train_set = GoodBadWebcams(DSET_DIR, train_list, train_labels, IMG_DIM)
    val_set = GoodBadWebcams(DSET_DIR, val_list, val_labels, IMG_DIM)

    train_loader = DataLoader(train_set, BATCH_SIZE, True, num_workers=NUM_WORKERS, drop_last=True)
    val_loader = DataLoader(val_set, BATCH_SIZE, True, num_workers=NUM_WORKERS, drop_last=True)
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), LR)

    best_loss = 9999999.0

    for epoch in range(EPOCHS):
        # eprint('\nEpoch ' + str(epoch+1))

        # eprint('Training...')
        model.train()

        all_outputs = None
        all_labels = None
        all_paths = []
        running_loss = 0.0

        for step, (data, labels, paths) in enumerate(train_loader):
            if USE_CUDA:
                data = data.cuda()
                labels = labels.cuda()
                
            output = torch.sigmoid(model(data))
            loss = loss_fn(output, labels)
            loss.backward()
            running_loss += loss.item() / len(train_loader)
            
            optimizer.step()
            optimizer.zero_grad()

            if step == 0:
                all_outputs = output.detach().clone()
                all_labels = labels.detach().clone()
                all_paths += paths
            else:
                all_outputs = torch.cat((all_outputs, output.detach().clone()), 0)
                all_labels = torch.cat((all_labels, labels.detach().clone()), 0)
                all_paths += paths
        
        all_outputs = torch.squeeze(all_outputs).cpu()
        all_labels = torch.squeeze(all_labels).cpu()
        training_accuracy = temf.binary_accuracy(all_outputs, all_labels)
        training_loss = running_loss
        print(f'{training_loss},{training_accuracy},', end='')
        
        # eprint('\nValidating...')
        model.eval()

        all_outputs = None
        all_labels = None
        all_paths = []
        running_loss = 0.0

        for step, (data, labels, paths) in enumerate(val_loader):
            if USE_CUDA:
                data = data.cuda()
                labels = labels.cuda()
                
            output = torch.sigmoid(model(data))
            loss = loss_fn(output, labels)
            loss.backward()
            running_loss += loss.item() / len(val_loader)

            if step == 0:
                all_outputs = output.detach().clone()
                all_labels = labels.detach().clone()
                all_paths += paths
            else:
                all_outputs = torch.cat((all_outputs, output.detach().clone()), 0)
                all_labels = torch.cat((all_labels, labels.detach().clone()), 0)
                all_paths += paths
        
        all_outputs = torch.squeeze(all_outputs).cpu()
        all_labels = torch.squeeze(all_labels).cpu()
        val_accuracy = temf.binary_accuracy(all_outputs, all_labels)
        val_loss = running_loss
        print(f'{val_loss},{val_accuracy},', end='')

        good_outputs = all_outputs[all_labels > 0.5]
        good_labels = torch.ones(good_outputs.size(0))
        bad_outputs = all_outputs[all_labels < 0.5]
        bad_labels = torch.zeros(bad_outputs.size(0))
        good_acc = temf.binary_accuracy(good_outputs, good_labels)
        bad_acc = temf.binary_accuracy(bad_outputs, bad_labels)
        print(f'{bad_acc},{good_acc}')


        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'goodbad-bestloss.pt')

def clean_dset():
    with torch.inference_mode():
        model = tv.models.resnet34(num_classes=1)
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
        model.eval()
        if USE_CUDA:
            model.cuda()

        with open(ALL_SAMPLES) as f:
            all_samples_list = [line.rstrip() for line in f]

        dataset = GoodBadWebcams(DSET_DIR, all_samples_list, [0.0]*len(all_samples_list), IMG_DIM)
        loader = DataLoader(dataset, BATCH_SIZE, num_workers=NUM_WORKERS)

        all_outputs = None
        all_paths = []

        for step, (data, labels, paths) in enumerate(loader):
            if USE_CUDA:
                data = data.cuda()
                labels = labels.cuda()
                
            output = torch.sigmoid(model(data))

            if step == 0:
                all_outputs = output.detach().clone()
                all_paths += paths
            else:
                all_outputs = torch.cat((all_outputs, output.detach().clone()), 0)
                all_paths += paths
        
    all_outputs = torch.squeeze(all_outputs).cpu()
    good_predictions = np.array(all_paths)[all_outputs > 0.5]
    bad_predictions = np.array(all_paths)[all_outputs <= 0.5]

    with open('good_nnlabelled.csv', 'wt') as f:
        for path in good_predictions:
            f.write(f'{path}\n')
    with open('bad_nnlabelled.csv', 'wt') as f:
        for path in bad_predictions:
            f.write(f'{path}\n')


# def convert():
#     INPUT_SHAPE = (1,3,280,280)
#     OPTIMIZE = False
#     IMG_PATH = '/home/feet/Documents/SchoolResearch/VisibilityResearch/Datasets/datasets/NewWebcams/4-9-24-6-47-PM/SITE7_ORNT312_VIS9mi.png'

#     resize_func = util.image_cropping.get_resize_crop_fn(INPUT_SHAPE[-2:])

#     class WithSigmoid(nn.Module):
#         def __init__(self, model):
#             super(WithSigmoid, self).__init__()
#             self.model = model
        
#         def forward(self, x):
#             return nn.functional.sigmoid(self.model(x))
        
    
#     model = tv.models.resnet34(num_classes=1)

#     data = io.read_image(IMG_PATH, io.ImageReadMode.RGB)/255.0
#     data = resize_func(data)
#     data = torch.unsqueeze(data, 0)

#     preconv = model(data)
#     torch.onnx.export(model, data, "goodbad", input_names=['input'], output_names=['output'], opset_version=11)
#     onnx_model = onnx.load("goodbad")
#     tf_rep = prepare(onnx_model)
#     tf_model_dir = "./tf_model"
#     tf_rep.export_graph(tf_model_dir)

#     converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_dir)

#     converter.optimizations = []
#     if OPTIMIZE:
#         converter.optimizations = [tf.lite.Optimize.DEFAULT]


#     tflite_model = converter.convert()

#     with open("goodbad.tflite", "wb") as f:
#         f.write(tflite_model)

#     interpreter = tf.lite.Interpreter(model_path='goodbad.tflite')
#     interpreter.allocate_tensors()

#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()

#     data = np.array(data, dtype=np.float32)

#     input_data = np.array(data, dtype=np.float32)
#     interpreter.set_tensor(input_details[0]['index'], input_data)

#     interpreter.invoke()

#     postconv = interpreter.get_tensor(output_details[0]['index'])

#     os.remove("goodbad")
#     shutil.rmtree('tf_model')

#     print(preconv)
#     print(postconv)

clean_dset()