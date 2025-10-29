import os
import sys
import argparse
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from pytorch_nndct.apis import torch_quantizer, dump_xmodel
from torchvision import datasets, transforms, models

from common import *   # if you still use any helper functions like `test()` from common.py

DIVIDER = '-----------------------------------------'

def quantize(build_dir, quant_mode, batchsize):
    dset_dir = "data"
    float_model = os.path.join(build_dir, 'float_model')
    quant_model = os.path.join(build_dir, 'quant_model')

    # device selection (for info + loading)
    if (torch.cuda.device_count() > 0):
        print('You have', torch.cuda.device_count(), 'CUDA devices available')
        for i in range(torch.cuda.device_count()):
            print(' Device', str(i), ': ', torch.cuda.get_device_name(i))
        print('Selecting device 0..')
        device = torch.device('cuda:0')
    else:
        print('No CUDA devices available..selecting CPU')
        device = torch.device('cpu')

    # Determine number of classes from train folder (assumes one subfolder per class)
    train_folder = os.path.join(dset_dir, "train")
    if not os.path.isdir(train_folder):
        raise RuntimeError(f"Train folder not found: {train_folder}")
    class_names = [d for d in os.listdir(train_folder) if os.path.isdir(os.path.join(train_folder, d))]
    num_classes = len(class_names)
    if num_classes == 0:
        raise RuntimeError("No class subfolders found in data/train")

    print(f"Found {num_classes} classes: {class_names}")

    # Construct ResNet18 model (matching what you trained)
    # Use weights=None to avoid trying to download pretrained weights during quantize script
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # Load trained float weights
    float_model_path = os.path.join(float_model, 'f_model.pth')
    if not os.path.isfile(float_model_path):
        raise RuntimeError(f"Float model not found at {float_model_path}. Train the model first.")
    state = torch.load(float_model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print("Loaded float model from:", float_model_path)

    # override batchsize if in test mode (keeps consistent with original behavior)
    if quant_mode == 'test':
        batchsize = 1

    # random input used to create quantizer (3-channel, 224x224)
    rand_in = torch.randn([batchsize, 3, 224, 224])

    # Create the quantizer
    quantizer = torch_quantizer(quant_mode, model, (rand_in), output_dir=quant_model)
    quantized_model = quantizer.quant_model
    print("Quantizer created. Output dir:", quant_model)

    # Test / evaluation dataset: use ImageFolder val set and same transforms as training
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_folder = os.path.join(dset_dir, "val")
    if not os.path.isdir(val_folder):
        raise RuntimeError(f"Validation folder not found: {val_folder}")

    val_dataset = datasets.ImageFolder(val_folder, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batchsize,
                                             shuffle=False)

    # Evaluate the quantized model (uses your common.test)
    # Note: quantized_model should be on CPU by default depending on quantizer, but map to device if needed
    try:
        # move quantized model to device if supported (quantizer sometimes expects CPU)
        quantized_model = quantized_model.to(device)
    except Exception:
        pass

    print("Running evaluation on quantized model...")
    test(quantized_model, device, val_loader)

    # export config or xmodel as before
    if quant_mode == 'calib':
        quantizer.export_quant_config()
        print("Exported quant config.")
    if quant_mode == 'test':
        # export xmodel for deployment
        quantizer.export_xmodel(deploy_check=False, output_dir=quant_model)
        print("Exported xmodel to:", quant_model)

    return


def run_main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-d',  '--build_dir',  type=str, default='build',    help='Path to build folder. Default is build')
    ap.add_argument('-q',  '--quant_mode', type=str, default='calib',    choices=['calib','test'], help='Quantization mode (calib or test). Default is calib')
    ap.add_argument('-b',  '--batchsize',  type=int, default=100,        help='Testing batchsize - must be an integer. Default is 100')
    args = ap.parse_args()

    print('\n'+DIVIDER)
    print('PyTorch version : ', torch.__version__)
    print(sys.version)
    print(DIVIDER)
    print(' Command line options:')
    print ('--build_dir    : ', args.build_dir)
    print ('--quant_mode   : ', args.quant_mode)
    print ('--batchsize    : ', args.batchsize)
    print(DIVIDER)

    quantize(args.build_dir, args.quant_mode, args.batchsize)

    return


if __name__ == '__main__':
    run_main()
