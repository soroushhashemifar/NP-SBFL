import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from config import args
from utils import (test, train)


class Net(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64*2*2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = self.pool(x)

        x = F.relu(self.conv2_1(x))
        x = self.pool(x)

        x = F.relu(self.conv3_1(x))
        x = self.pool(x)

        x = F.relu(self.conv4_1(x))
        x = self.pool(x)
        
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


if __name__ == "__main__":
    #load the data
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=True, download=True,
                    transform = transforms.Compose([
                                    transforms.RandomHorizontalFlip(), # randomly flip and rotate
                                    transforms.RandomRotation(10),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])),
        batch_size=args['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])),
        batch_size=args['test_batch_size'], shuffle=True)

    model = Net()
    if args['cuda']:
        model = model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
    for epoch in range(1, args['epochs'] + 1):
        train(epoch, model, train_loader, optimizer, args)
        test(model, test_loader, args, scheduler)

    torch.save(model.state_dict(), "./mymodel_cifar10.pth")