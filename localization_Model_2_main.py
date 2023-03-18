import torch
from torchvision import datasets, transforms

import lrp_src
from deepcp_method2 import DeepCP
from models.train_model_2 import Net
from utils import myLRPModel


class Model2(DeepCP):
    
    def get_relevancy_and_activations(self, data):
        relevancy, activations = self.lrp_model.forward(data)

        return relevancy, activations


if __name__ == "__main__":
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('models/data', train=True, transform=transforms.Compose([
                        transforms.ToTensor(),
                        # transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=1, shuffle=False)

    model = Net()
    model.load_state_dict(torch.load("models/Model_2.pth", map_location="cpu"))
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
        model=model,
        layers_structure=layers_structure, 
        input_size=(1, 28, 28), 
        train_loader=train_loader, 
        device="cpu",
        alpha=1.5,
    )

    model2.run()
