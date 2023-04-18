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
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 30)
        self.fc2 = nn.Linear(30, 30)
        self.fc3 = nn.Linear(30, 30)
        self.fc4 = nn.Linear(30, 30)
        self.fc5 = nn.Linear(30, 30)
        self.out = nn.Linear(30, 10)

    def forward(self, x, return_activations=False, return_logits=False):
        logits = []
        activations = []

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        logits.append(x)
        x = F.relu(x)
        activations.append(x)

        x = self.fc2(x) 
        logits.append(x)
        x = F.relu(x)
        activations.append(x)

        x = self.fc3(x)
        logits.append(x)
        x = F.relu(x)
        activations.append(x)

        x = self.fc4(x)
        logits.append(x)
        x = F.relu(x)
        activations.append(x)

        x = self.fc5(x)
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


if __name__ == "__main__":
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 100
    cuda = False

    #load the data
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        # transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        # transforms.Normalize((0.1307,), (0.3081,))
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
        
        if acc >= 96.6:
            break

    torch.save(model.state_dict(), "./Model_mnist_1.pth")