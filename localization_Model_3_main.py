import torch
from torchvision import datasets, transforms
import numpy as np

import lrp_src
from deepcp_method import DeepCP
from models.train_model_3 import Net
from utils import myLRPModel


class Model3(DeepCP):
    
    def get_relevancy_and_activations(self, data):
        relevancy, activations = self.lrp_model.forward(data)

        relevancy = [r.view(r.shape[0], r.shape[1], -1).sum(2) for r in relevancy]
        relevancy[8] = relevancy[8].view(-1, 64, 4).sum(2) # input of fc1, output of flatten!
        activations = [torch.norm(a, p='fro', dim=(2, 3)) if len(a.shape) == 4 else a for a in activations]

        return relevancy, activations


if __name__ == "__main__":
    ALPHA = 0.7

    # lrp_src.lrp_layers.top_k_percent = 0.9

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('models/data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])),
        batch_size=1, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('models/data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])),
        batch_size=1, shuffle=True)

    model = Net()
    model.load_state_dict(torch.load("./models/Model_3.pth", map_location="cpu"))
    model = model.to("cpu")
    model.eval()

    layers_structure = [
        model.conv1_1, torch.nn.ReLU(), model.pool, 
        model.conv2_1, torch.nn.ReLU(), model.pool, 
        model.conv3_1, torch.nn.ReLU(), model.pool,
        model.conv4_1, torch.nn.ReLU(), model.pool, 
        torch.nn.Flatten(1), 
        model.fc1, torch.nn.ReLU(), 
        model.fc2, torch.nn.ReLU(), 
        model.fc3 
    ]

    model3 = Model3(
        model_name="Model_3",
        model=model,
        layers_structure=layers_structure, 
        input_size=(3, 32, 32), 
        train_loader=train_loader, 
        test_loader=test_loader,
        device="cpu",
        batch_size=128,
        alpha=ALPHA, beta=0.99, min_match=0.8, 
        PCA_n_components=[8, 32], 
        Birch_thresholds=[0.1, 0.5, 0.9],
        Birch_n_clusters=[6, 10, 15],
    )

    model3.run()
