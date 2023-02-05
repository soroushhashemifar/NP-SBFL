import torch
from torchvision import datasets, transforms

import lrp_src
from deepcp_method import DeepCP
from models.train_model_2 import Net
from utils import myLRPModel


class Model2(DeepCP):
    
    def get_relevancy_and_activations(self, data):
        relevancy, activations = self.lrp_model.forward(data)

        return relevancy, activations


if __name__ == "__main__":
    ALPHA = 0.99

    lrp_src.lrp_layers.top_k_percent = ALPHA

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('models/data', train=True, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=1, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('models/data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=1, shuffle=True)

    model = Net()
    model.load_state_dict(torch.load("models/mymodel_2.pth", map_location="cpu"))
    model = model.to("cpu")
    model.eval()

    layers_structure = [
        torch.nn.Flatten(1), 
        model.fc1, torch.nn.ReLU(),
        model.fc2, torch.nn.ReLU(),
        model.fc3, torch.nn.ReLU(),
        model.fc4, torch.nn.ReLU(),
        model.fc5
    ]

    model2 = Model2(
        model_name="Model_2",
        layers_structure=layers_structure, 
        input_size=(1, 28, 28), 
        train_loader=train_loader, 
        test_loader=test_loader,
        device="cpu",
        batch_size=128,
        alpha=ALPHA, beta=0.6, min_match=0.8, 
        PCA_n_components=[16, 32, 64, 128], 
        Birch_thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        Birch_n_clusters=[2, 5, 7, 10],
    )

    model2.run()
