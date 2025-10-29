import torch
import torchvision
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from common import *

import os
import sys
import argparse
import shutil

DATA_DIR = "data"
DIVIDER = "<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def trainMethod(build_dir, batchsize, learnrate, epochs):

    float_model = build_dir + '/float_model'

    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_transform)

    
    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False, num_workers=2)
    

    classes = train_dataset.classes
    num_classes = len(classes)
    print(f"Detected {num_classes} soil classes: {classes}")

    # Load pre-trained ResNet18
    model = models.resnet18(pretrained=True)

    # Replace the final fully-connected layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Move to GPU (if available)
    model = model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=learnrate)

    # training with test after each epoch
    for epoch in range(1, epochs + 1):
        train(model, DEVICE, train_loader, optimizer, epoch)
        test(model, DEVICE, val_loader)


    # save the trained model
    shutil.rmtree(float_model, ignore_errors=True)    
    os.makedirs(float_model)   
    save_path = os.path.join(float_model, 'f_model.pth')
    torch.save(model.state_dict(), save_path) 
    print('Trained model written to',save_path)

    return


    

def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--build_dir',   type=str,  default='build',       help='Path to build folder. Default is build')
    ap.add_argument('-b', '--batchsize',   type=int,  default=100,           help='Training batchsize. Must be an integer. Default is 100')
    ap.add_argument('-e', '--epochs',      type=int,  default=3,             help='Number of training epochs. Must be an integer. Default is 3')
    ap.add_argument('-lr','--learnrate',   type=float,default=0.001,         help='Optimizer learning rate. Must be floating-point value. Default is 0.001')
    args = ap.parse_args()

    print('\n'+DIVIDER)
    print('PyTorch version : ',torch.__version__)
    print(sys.version)
    print(DIVIDER)
    print(' Command line options:')
    print ('--build_dir    : ',args.build_dir)
    print ('--batchsize    : ',args.batchsize)
    print ('--learnrate    : ',args.learnrate)
    print ('--epochs       : ',args.epochs)
    print(DIVIDER)

    # call train method

    trainMethod(args.build_dir, args.batchsize, args.learnrate, args.epochs) 

    return


if __name__ == "__main__":
    main()