import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from utils import test, train
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class Net(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=0)
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=0)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(6400, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, 7)

        self.dropout = nn.Dropout(0.25)

    def forward(self, x, return_activations=False, return_logits=False):
        logits = []
        activations = []

        x = self.conv1_1(x)
        logits.append(x)
        x = F.relu(x)
        activations.append(x)

        x = self.conv1_2(x)
        logits.append(x)
        x = F.relu(x)
        activations.append(x)

        x = self.pool(x)
        logits.append(x)
        x = self.dropout(x)

        x = self.conv2_1(x)
        logits.append(x)
        x = F.relu(x)
        activations.append(x)

        x = self.conv2_2(x)
        logits.append(x)
        x = F.relu(x)
        activations.append(x)

        x = self.pool(x)
        logits.append(x)
        x = self.dropout(x)
        
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        logits.append(x)
        x = F.relu(x)
        activations.append(x)

        x = self.fc2(x)
        logits.append(x)
        x = F.relu(x)
        activations.append(x)

        x = self.out(x)
        logits.append(x)

        if return_activations:
            return x, activations
        elif return_logits:
            return x, logits

        return x


class FER2013(Dataset):

    def __init__(self, csv_file, split= "Training", transform = None):            
        dataset = pd.read_csv(csv_file)
        self.transform = transform
        self.split = split
        if self.split == "Training":
            self.data = dataset[dataset["Usage"] == "Training"]
            assert len(self.data) == 28709
        elif self.split == "PublicTest":
            self.data = dataset[dataset["Usage"] == "PublicTest"]
            assert len(self.data) == 3589
        else:
            self.data = dataset[dataset["Usage"] == "PrivateTest"]
            assert len(self.data) == 3589
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = list(map(int, self.data["pixels"].iloc[idx].split(" ")))
        image = np.array(image)
        image = image.reshape(48, 48).astype(np.uint8)
        
        #image = image[:, :, np.newaxis]
        #image = np.concatenate((image, image, image), axis= 2)
        image = Image.fromarray(image)
        
        if self.transform is not None:
            image = self.transform(image)
        
        target = self.data["emotion"].iloc[idx]
        return image, target


if __name__ == "__main__":
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 100
    cuda = False

    #load the data
    train_loader = torch.utils.data.DataLoader(
        FER2013('./data/fer2013/fer2013.csv', split="Training",
                    transform = transforms.Compose([
                                    transforms.RandomCrop(48, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        FER2013('./data/fer2013/fer2013.csv', split="PublicTest", transform=transforms.Compose([
                        transforms.Resize((48, 48)),
                        transforms.ToTensor(),
                        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])),
        batch_size=batch_size, shuffle=True)

    model = Net()
    if cuda:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)
    parameters = {
            "cuda": cuda,
            "loss": nn.CrossEntropyLoss(),
            "log_interval": 150
        }
    for epoch in range(1, num_epochs + 1):
        train(epoch, model, train_loader, optimizer, parameters)
        acc, _ = test(model, test_loader, parameters, scheduler)
        
        if acc >= 90.:
            break

    torch.save(model.state_dict(), "./Model_fer2013_1.pth")