import argparse
import torch
from torchvision import io
import torchvision.transforms as tf
import torchvision.transforms.functional as tff
import importlib
import sys
import os
from types import SimpleNamespace

#arguments: config_path model_path < infile > outfile

parser = argparse.ArgumentParser()
parser.add_argument('config_path')
parser.add_argument('model_path')
args = parser.parse_args()

def Pt_Original():
    config_path = args.config_path
    spec = importlib.util.spec_from_file_location("config", os.path.join(os.getcwd(), config_path))
    config = importlib.util.module_from_spec(spec)
    config.__package__ = 'vis_networks'
    sys.modules["config"] = config
    spec.loader.exec_module(config)
    config = config.CONFIG
    cf = SimpleNamespace(**config)

    ModelClass = cf.model_module.Model
    
    state_dict = torch.load(args.model_path, weights_only=True, map_location=torch.device('cpu'))
    sample_in = state_dict['example_input']
    sample_out = state_dict['example_output']

    input_size = sample_in.size()
    output_size = sample_out.size(1)

    model = ModelClass(output_size, input_size[1:], **cf.model_params)
    model.load_state_dict(state_dict, False)
    model.register_buffer('example_input', state_dict['example_input'])
    model.register_buffer('example_output', state_dict['example_output'])
    model.eval()


    def resize_crop(image):
        #height over width
        target_ratio = input_size[-2] / input_size[-1]
        ratio = image.shape[-2] / image.shape[-1]
        
        #if the the image is too tall, crop the top and bottom
        #otherwise crop sides
        if ratio > target_ratio:
            crop = (round(target_ratio*image.shape[-1]), image.shape[-1])
        else:
            crop = (image.shape[-2], round(image.shape[-2]/target_ratio))

        image = tff.center_crop(image, crop)
        image = tff.resize(image, input_size[-2:], tf.InterpolationMode.BICUBIC)

        return image
    
    transformer = resize_crop
    if hasattr(cf.model_module, 'CUSTOM_TRANSFORM') and not cf.model_module.CUSTOM_TRANSFORM is None:
        custom_transformer = cf.model_module.CUSTOM_TRANSFORM(**cf.transform_params)
        transformer = lambda x: custom_transformer(resize_crop(x))

    with torch.inference_mode():
        while True:
            try:
                line = input()
            except:
                break

            if line == "":
                break
            
            sample_data = io.decode_image(line, 'RGB')/255.0
            sample_data = transformer(sample_data)

            sample_out = model(sample_data)

            if output_size > 1:
                sample_out = torch.softmax(sample_out, 1)
                pred_class = torch.argmax(sample_out).item()
                out_list = sample_out[0].tolist()
                sm_output_str = [f'{x:.4f}' for x in out_list]
                sm_output_str = ','.join(sm_output_str)
                print(f'"{sm_output_str}",{pred_class}')
            else:
                print(f'{sample_out.item():.4f}')

Pt_Original()