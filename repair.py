import copy
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from models.train_model_4 import Net
from synthesize_v2 import SynthesizedDataset
from utils import test, train


def repair_model(model_name, SFL_strategy, suspiciousness_threshold, train_loader, test_loader, model, learning_rate, num_epochs):
    with open(f"./pickles/synthesized_dataset_{model_name}_{SFL_strategy}_k{suspiciousness_threshold}.pickle", 'rb') as handle:
        synthesized_dataset = pickle.load(handle)

    old_train_data = []
    for data, labels in train_loader:
        old_train_data.append((None, data[0].detach().numpy(), labels[0].item()))
        if len(old_train_data) == len(synthesized_dataset):
            break

    total_dataset = synthesized_dataset + old_train_data
    synth_dataset = SynthesizedDataset(total_dataset)
    synth_loader = torch.utils.data.DataLoader(
        synth_dataset,
        batch_size=256, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)
    parameters = {
            "cuda": False,
            "loss": nn.CrossEntropyLoss(),
            "log_interval": 10
        }

    best_model = None
    best_accuracy = 0
    best_loss = torch.inf
    for epoch in range(1, num_epochs + 1):
        train(epoch, model, synth_loader, optimizer, parameters)
        acc, test_loss = test(model, test_loader, parameters, scheduler)
        if acc > best_accuracy:
            best_model = copy.deepcopy(model)
            best_accuracy = acc
            best_loss = test_loss

    print("Best model acc:", best_accuracy.item(), "loss:", best_loss.item())

    torch.save({
        "model": best_model.state_dict(),
        "accuracy": best_accuracy.item(),
        "loss": best_loss.item(),
        }, f"./models/repaired_{model_name}.pth")

if __name__ == "__main__":
    num_epochs = 10
    learning_rate = 0.001
    model_name = "Model_cifar_1"
    SFL_strategy = "tarantula"
    suspiciousness_threshold = 10

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('models/data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                    ])),
        batch_size=1, shuffle=False)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('models/data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                    ])),
        batch_size=256, shuffle=False)

    model = Net()
    model.load_state_dict(torch.load(f"models/{model_name}.pth", map_location="cpu"))
    model = model.to("cpu")

    repair_model(model_name, SFL_strategy, suspiciousness_threshold, train_loader, test_loader, model, learning_rate, num_epochs)