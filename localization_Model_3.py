import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.nn.functional as F

from deepcp_method import DeepCP
from models.train_model_3 import Net
from synthesize import Synthesize, evaluation
from verification import Verification


class Model3(DeepCP):
    
    def get_relevancy_and_activations(self, data):
        relevancies, activations = self.lrp_model.forward(data)

        activations = activations[:-1]

        relevancy = list(filter(lambda item: item[0] in ["RelevancePropagationLinear"], relevancies))
        relevancy = list(map(lambda item: item[1], relevancy))
        activations = list(filter(lambda item: item[0] in ["RelevancePropagationLinear"], activations))
        activations = list(map(lambda item: F.relu(item[1]) if item[0] in ["RelevancePropagationLinear"] else item[1], activations))

        g_fx = torch.sum(relevancies[0][1]).item()
        predicted_class = torch.max(relevancies[-1][1], dim=1).indices.item()

        return relevancy, activations, g_fx, predicted_class


if __name__ == "__main__":
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('models/data', train=True, transform=transforms.Compose([
                        transforms.ToTensor(),
                    ])),
        batch_size=1, shuffle=False)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('models/data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                    ])),
        batch_size=128, shuffle=False)

    model = Net()
    model.load_state_dict(torch.load("models/Model_mnist_3.pth", map_location="cpu"))
    model = model.to("cpu")
    model.eval()

    layers_structure = [
        torch.nn.Flatten(1), 
        model.fc1, torch.nn.ReLU(),
        model.fc2, torch.nn.ReLU(),
        model.fc3, torch.nn.ReLU(),
        model.fc4, torch.nn.ReLU(),
        model.fc5, torch.nn.ReLU(),
        model.fc6, torch.nn.ReLU(),
        model.fc7, torch.nn.ReLU(),
        model.fc8, torch.nn.ReLU(),
        model.out
    ]

    print("Localizing faults in model 3")
    deepcp3 = Model3(
        model_name="Model_mnist_3",
        model=model,
        layers_structure=layers_structure, 
        input_size=(1, 28, 28), 
        train_loader=train_loader, 
        device="cpu", 
        alpha=0.9, 
        beta=0.6, activation_threshold=0.
    )
    # deepcp3.run()

    print("Synthesizing dataset for model 3")
    model_3_synthsizer = Synthesize(deepcp3.model_name, model, test_loader, pickles_path=deepcp3.path_to_save_pickles, step_size=1, distance=0.1)

    suspiciousness_threshold = 5

    parameters = {
            "cuda": False,
            "loss": nn.CrossEntropyLoss(),
        }
    model_3_synthsizer.run("tarantula", suspiciousness_threshold=suspiciousness_threshold)
    evaluation(deepcp3.model_name, "tarantula", model, test_loader, parameters, suspiciousness_threshold=suspiciousness_threshold)
    model_3_synthsizer.run("ochiai", suspiciousness_threshold=suspiciousness_threshold)
    evaluation(deepcp3.model_name, "ochiai", model, test_loader, parameters, suspiciousness_threshold=suspiciousness_threshold)
    model_3_synthsizer.run("barinel", suspiciousness_threshold=suspiciousness_threshold)
    evaluation(deepcp3.model_name, "barinel", model, test_loader, parameters, suspiciousness_threshold=suspiciousness_threshold)

    print("Verifying model 3")
    verification = Verification(deepcp3)

    verification.verify("tarantula", suspiciousness_threshold=suspiciousness_threshold)
    verification.verify("ochiai", suspiciousness_threshold=suspiciousness_threshold)
    verification.verify("barinel", suspiciousness_threshold=suspiciousness_threshold)
