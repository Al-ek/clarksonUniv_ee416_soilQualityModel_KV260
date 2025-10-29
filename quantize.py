import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from train import soilQualityClassifier, soilQualityDataset
from pytorch_nndct.apis import torch_quantizer
from torchvision import transforms
from torch.utils.data import DataLoader

DATA_DIR = 'data'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BUILD_DIR = 'build'
FLOAT_MODEL_DIR = BUILD_DIR + '/floatModel'
QUANT_MODEL_DIR = 'build/quantModel'
FLOAT_MODEL_FILE = FLOAT_MODEL_DIR + '/soilQualiy_floatModel.pth'   

def quantize(quant_mode): 

    batchsize = 32
    model = soilQualityClassifier(num_classes=2)
    # Load trained float weights
    FLOAT_MODEL_FILE
    if not os.path.isfile(FLOAT_MODEL_FILE):
        raise RuntimeError(f"Float model not found at {FLOAT_MODEL_FILE}. Train the model first.")
    state = torch.load(FLOAT_MODEL_FILE, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    print("Loaded float model from:", FLOAT_MODEL_FILE)

    # override BATCHSIZE if in test mode (keeps consistent with original behavior)
    if quant_mode == 'test':
        batchsize = 1

    # random input used to create quantizer (3-channel, 224x224)
    rand_in = torch.randn([batchsize, 3, 224, 224])

    # Create the quantizer
    quantizer = torch_quantizer(quant_mode, model, (rand_in), output_dir=QUANT_MODEL_DIR)
    quantized_model = quantizer.quant_model

    # Test / evaluation dataset: use ImageFolder val set and same transforms as training
    # Transform images to 128,128 and transform into a tensor
    transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    val_dataset = soilQualityDataset(DATA_DIR + '/val/', transform=transform)
    # Dataloader
    val_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False)

    # Evaluate the quantized model (uses your common.test)
    # Note: quantized_model should be on CPU by default depending on quantizer, but map to DEVICE if needed

    quantized_model = quantized_model.to(DEVICE)

    print("\nRunning evaluation on quantized model...")
    quantized_model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc='Validating model'):
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = quantized_model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    acc = 100. * correct / len(val_loader.dataset)
    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(correct, len(val_loader.dataset), acc))

    # export config or xmodel as before
    if quant_mode == 'calib':
        quantizer.export_quant_config()
        print("Exported quant config.")
    if quant_mode == 'test':
        # export xmodel for deployment
        quantizer.export_xmodel(deploy_check=False, output_dir=QUANT_MODEL_DIR)
        print("Exported xmodel to:", QUANT_MODEL_DIR)
    return

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-q',  '--quant_mode', type=str, default='calib',    choices=['calib','test'], help='Quantization mode (calib or test). Default is calib')
    args = ap.parse_args()
    quantize(args.quant_mode)
    return

if __name__ == "__main__":
    main()
