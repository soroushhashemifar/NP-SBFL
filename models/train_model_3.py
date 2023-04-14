import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from utils import test, train


class Net(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128*8*8, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 10)

        self.dropout = nn.Dropout(0.25)

    def forward(self, x, return_activations=False, return_logits=False):
        logits = []
        activations = []

        x = self.conv1_1(x)
        logits.append(x)
        x = F.relu(x)
        activations.append(x)

        x = self.conv2_1(x)
        logits.append(x)
        x = F.relu(x)
        activations.append(x)
        x = self.pool(x)
        logits.append(x)
        x = self.dropout(x)

        x = self.conv3_1(x)
        logits.append(x)
        x = F.relu(x)
        activations.append(x)

        x = self.conv4_1(x)
        logits.append(x)
        x = F.relu(x)
        activations.append(x)
        x = self.pool(x)
        logits.append(x)
        x = self.dropout(x)
        
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        logits.append(x)
        x = F.relu(x)
        activations.append(x)
        x = self.fc2(x)
        logits.append(x)
        x = F.relu(x)
        activations.append(x)
        x = self.dropout(x)
        x = self.fc3(x)
        logits.append(x)

        if return_activations:
            return x, activations
        elif return_logits:
            return x, logits

        return x


if __name__ == "__main__":
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 100
    cuda = False

    #load the data
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=True, download=True,
                    transform = transforms.Compose([
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
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
            "log_interval": 10
        }
    for epoch in range(1, num_epochs + 1):
        train(epoch, model, train_loader, optimizer, parameters)
        acc, _ = test(model, test_loader, parameters, scheduler)
        
        if acc > 75:
            break

    torch.save(model.state_dict(), "./Model_3.pth")