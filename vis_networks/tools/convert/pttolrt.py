import ai_edge_torch
import numpy
import torch
import torch.nn as nn
from torchvision import io
from ...models import *

INPUT_SHAPE = (1,3,280,280)
N_CLASSES = 10
MODEL_NAME = "RMEP"
MODEL_PATH = '/home/feet/Documents/Research/data/VisResearch/Visibility-Networks/runs/Nov04_16-37-58_lessland/best_epoch10.pt'
MODEL_MODULE = rmep.rmep

class WithSigmoid(nn.Module):
    def __init__(self, model):
        super(WithSigmoid, self).__init__()

        self.model = model
    
    def forward(self, x):
        return nn.functional.sigmoid(self.model(x))
    
class WithSoftmax(nn.Module):
    def __init__(self, model):
        super(WithSoftmax, self).__init__()

        self.model = model

    def forward(self, x):
        return nn.functional.softmax(self.model(x), 1)


state_dict = torch.load(MODEL_PATH, weights_only=True, map_location=torch.device('cpu'))
mean = torch.zeros((3))
std = torch.ones((3))

ex_in = state_dict['example_input']
ex_out = torch.nn.functional.softmax(state_dict['example_output'], 1)
model = MODEL_MODULE.Model(N_CLASSES, ex_in.size()[1:])
model.load_state_dict(state_dict, False)
model.register_buffer('example_input', ex_in)
model.register_buffer('example_output', ex_out)
model = WithSoftmax(model)
model.eval()

precon = model(ex_in)

tfl_model = ai_edge_torch.convert(model.eval(), (ex_in,))

postconv = tfl_model(ex_in)

print(precon)
print(postconv)
if (numpy.allclose(
    precon.detach().numpy(),
    postconv,
    atol=1e-5,
    rtol=1e-5,
)):
    print("Inference result with Pytorch and TfLite was within tolerance")
else:
    print("Something wrong with Pytorch --> TfLite")

print(ex_out)

tfl_model.export(f"./{MODEL_NAME}.tflite")