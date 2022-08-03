from logging import root
from typing import Callable, Union, Tuple, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms as transforms
from skimage import io, transform
import numpy as np
from pathlib import Path
import pandas as pd
#from PIL import Image

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import mlflow


class LabeledImageLoader(Dataset):
    labels : Tuple[str] = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    labels_map : Dict[str, int] = {l: i for i, l in enumerate(labels)}

    def __init__(self, root_dir : Path, transform : Callable = None):
        self.meta_data = pd.read_csv(root_dir / 'labels.csv')

        self.image_paths = [root_dir / f"{row}.png" for row in self.meta_data['id']]
        self.decoded_labels = self.meta_data['label'].map(self.labels_map).astype(int).tolist()

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx : Union[int, torch.tensor]):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        else:
            idx = [idx,]

        images = [io.imread(self.image_paths[i]) for i in idx]
        #images = [Image.open(self.image_paths[i]) for i in idx]
        labels = [self.decoded_labels[i] for i in idx]
        
        if self.transform:
            images = [self.transform(img) for img in images]

        return images, tuple(labels)
    
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def get_training_dataloader(root_dir : Path, batch_size : int = 32) -> DataLoader:
    trans = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])    
    data = LabeledImageLoader(root_dir / "train", transform=trans)
    return torch.utils.data.DataLoader(data, batch_size=batch_size)

def get_testing_dataloader(root_dir : Path, batch_size : int = 32) -> DataLoader:
    trans = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])    
    data = LabeledImageLoader(root_dir / "test", transform=trans)
    return torch.utils.data.DataLoader(data, batch_size=batch_size)


def create_optimizer(model : nn.Module) -> optim.Optimizer: 
    return optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def create_loss_fun():
    return nn.CrossEntropyLoss()


def train_model(model : nn.Module, epochs : int , data : DataLoader, print_steps = 5):
    print(f"Training model ...")
    model.train()
    S = len(data) // print_steps
    for epoch in range(epochs):  # loop over the dataset multiple times
        optimizer = create_optimizer(model)
        loss_func = create_loss_fun()
        running_loss = 0.0
        for i, (images, labels) in enumerate(data, 0):
            # TODO: Check why this is wrapped in a list!
            images = images[0]
            labels = labels[0]
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(images)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % S == S-1:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / i:.3f}')
        
        mlflow.log_metrics({"avg_loss": running_loss/len(data)}, step=epoch)
    print(f"Finished training after {epochs} epochs!")

def test_model(model : nn.Module, data : DataLoader) :
    print(f"Testing model ...")
    confusion_matrix = np.zeros((len(LabeledImageLoader.labels), len(LabeledImageLoader.labels)))
    
    model.eval()
    # again no gradients needed
    with torch.no_grad():
        for images, labels in data:
            # TODO: Check why this is wrapped in a list!
            images = images[0]
            labels = labels[0]
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                confusion_matrix[label, prediction] += 1

    print(f"Finished testing model ...") 
    return confusion_matrix