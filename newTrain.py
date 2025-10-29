import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import timm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import os
import sys
import argparse
import shutil

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import models

#
DATA_DIR = '~/clarkson/ee416/soilData/data'
BATCHSIZE = 32
SHUFFLE = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 5
BUILD_DIR = 'build'

# Dataset
class soilQualityDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def classes(self):
        return self.data.classes
    

# Transform images to 128,128 and transform into a tensor
transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])


train_dataset = soilQualityDataset(DATA_DIR + '/train/', transform=transform)
val_dataset = soilQualityDataset(DATA_DIR + '/val/', transform=transform)

# Dataloader
train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=SHUFFLE)
val_loader = DataLoader(val_dataset, batch_size=BATCHSIZE, shuffle=False)

# Create Pytorch Model
class soilQualityClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(soilQualityClassifier, self).__init__()
        self.base_model = timm.create_model('resnet18', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        resnet_out_size = 512

        self.classifier = nn.Linear(resnet_out_size, num_classes)

    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)
        return output
    
# Training loop
model = soilQualityClassifier(num_classes=2)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(EPOCHS):

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc='Train loop'):

        images, labels = images.to(DEVICE), labels.to(DEVICE) 
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)
    train_acc = correct / total
    train_accuracies.append(train_acc)

    # Validation phase
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in tqdm(val_loader, desc='Validation loop'):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / len(val_loader.dataset)
    val_losses.append(val_loss)
    val_acc = correct / total
    val_accuracies.append(val_acc)

    print(f"[Epoch {epoch+1}/{EPOCHS}] "
          f"\nTrain loss: {train_loss}, Training accuracy: {train_acc*100:.2f}% "
          f"\nValidation loss {val_loss}, Validation accuracy: {val_acc*100:.2f}% ") 

shutil.rmtree(BUILD_DIR + '/floatModel', ignore_errors=True)    
os.makedirs(BUILD_DIR + '/floatModel')   
save_path = os.path.join(BUILD_DIR + '/floatModel', 'soilQualiy_floatModel.pth')
torch.save(model.state_dict(), save_path) 
print('Trained model written to',save_path)


