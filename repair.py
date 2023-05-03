import copy
import pickle

import torch
import torch.nn as nn
import torch.optim as optim

from synthesize import SynthesizedDataset
from utils import test, train


def repair_model(model_name, SFL_strategy, suspiciousness_threshold, train_loader, test_loader, model, learning_rate, num_epochs):
    with open(f"./pickles/synthesized_dataset_{model_name}_{SFL_strategy}_k{suspiciousness_threshold}.pickle", 'rb') as handle:
        synthesized_dataset = pickle.load(handle)

    old_train_data = []
    for data, labels in train_loader:
        old_train_data.append((None, data[0].detach().numpy(), labels[0].item()))
        if len(old_train_data) == 10000:
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
