import torch
import torch.nn as nn
from torchvision import datasets, transforms

from deepcp_method import DeepCP
from models.train_model_3 import Net
from synthesize import Synthesize, evaluation
from verification import Verification


class Model3(DeepCP):
    
    def get_relevancy_and_activations(self, data):
        relevancies, activations = self.lrp_model.forward(data)

        predicted_class = torch.argmax(torch.softmax(activations[-1][1], 1)).item()

        relevancy = [(r[0], r[1].flatten(1)) for r in relevancies]
        activations = [(a[0], a[1].flatten(1)) for a in activations[:-1]]

        relevancy = list(filter(lambda item: item[0] in ["RelevancePropagationConv2d", "RelevancePropagationMaxPool2d", "RelevancePropagationLinear"], relevancy))
        relevancy = list(map(lambda item: item[1], relevancy))
        activations = list(filter(lambda item: item[0] in ["RelevancePropagationConv2d", "RelevancePropagationMaxPool2d", "RelevancePropagationLinear"], activations))
        activations = list(map(lambda item: torch.relu(item[1]) if item[0] in ["RelevancePropagationConv2d", "RelevancePropagationLinear"] else item[1], activations))

        g_fx = torch.sum(relevancies[0][1]).item()

        return relevancy, activations, g_fx, predicted_class


if __name__ == "__main__":
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
        batch_size=128, shuffle=False)

    model = Net()
    model.load_state_dict(torch.load("models/Model_3.pth", map_location="cpu"))
    model = model.to("cpu")
    model.eval()

    layers_structure = [
        model.conv1_1, torch.nn.ReLU(),
        model.conv2_1, torch.nn.ReLU(), 
        model.pool, 
        model.conv3_1, torch.nn.ReLU(), 
        model.conv4_1, torch.nn.ReLU(), 
        model.pool, 
        torch.nn.Flatten(1), 
        model.fc1, torch.nn.ReLU(), 
        model.fc2, torch.nn.ReLU(), 
        model.fc3 
    ]

    print("Localizing faults in model 3")
    deepcp3 = Model3(
        model_name="Model_3",
        model=model,
        layers_structure=layers_structure, 
        input_size=(3, 32, 32), 
        train_loader=train_loader, 
        device="cpu",
        alpha=0.99, 
        beta=0.6, activation_threshold=0.
    )
    deepcp3.run()

    print("Synthesizing dataset for model 3")
    model_3_synthsizer = Synthesize("Model_3", model, test_loader, pickles_path="pickles/", step_size=20, distance=0.1)
    
    parameters = {
            "cuda": False,
            "loss": nn.CrossEntropyLoss(),
        }
    model_3_synthsizer.run("tarantula", suspiciousness_threshold=10)
    evaluation("Model_3", "tarantula", model, test_loader, parameters)
    model_3_synthsizer.run("ochiai", suspiciousness_threshold=10)
    evaluation("Model_3", "ochiai", model, test_loader, parameters)
    model_3_synthsizer.run("barinel", suspiciousness_threshold=10)
    evaluation("Model_3", "barinel", model, test_loader, parameters)

    print("Verifying model 3")
    verification = Verification(deepcp3, "pickles/")

    verification.verify("tarantula", suspiciousness_threshold=10)
    verification.verify("ochiai", suspiciousness_threshold=10)
    verification.verify("barinel", suspiciousness_threshold=10)
