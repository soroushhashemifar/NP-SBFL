import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from utils import (test, train)


class Net(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x)) 
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


if __name__ == "__main__":
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 10
    cuda = False

    #load the data
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=batch_size, shuffle=True)

    model = Net()
    if cuda:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
    parameters = {
            "cuda": cuda,
            "loss": "nll_loss",
            "log_interval": 10
        }
    for epoch in range(1, num_epochs + 1):
        train(epoch, model, train_loader, optimizer, parameters)
        test(model, test_loader, parameters, scheduler)

    torch.save(model.state_dict(), "./mymodel_1.pth")