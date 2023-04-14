import torch
import torch.nn as nn
from torchvision import datasets, transforms

from deepcp_method import DeepCP
from models.train_model_2 import Net
from synthesize import Synthesize, evaluation
from verification import Verification


class Model2(DeepCP):
    
    def get_relevancy_and_activations(self, data):
        relevancies, activations = self.lrp_model.forward(data)

        activations = activations[:-1]

        relevancy = list(filter(lambda item: item[0] in ["RelevancePropagationLinear"], relevancies))
        relevancy = list(map(lambda item: item[1], relevancy))
        activations = list(filter(lambda item: item[0] in ["RelevancePropagationLinear"], activations))
        activations = list(map(lambda item: torch.relu(item[1]) if item[0] in ["RelevancePropagationLinear"] else item[1], activations))

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

    print("Localizing faults in model 2")
    deepcp2 = Model2(
        model_name="Model_2",
        model=model,
        layers_structure=layers_structure, 
        input_size=(1, 28, 28), 
        train_loader=train_loader, 
        device="cpu", 
        alpha=0.9, 
        beta=0.6, activation_threshold=0.
    )
    deepcp2.run()

    print("Synthesizing dataset for model 2")
    model_2_synthsizer = Synthesize("Model_2", model, test_loader, pickles_path="pickles/", step_size=10, distance=0.9)

    parameters = {
            "cuda": False,
            "loss": nn.CrossEntropyLoss(),
        }
    model_2_synthsizer.run("tarantula", suspiciousness_threshold=0.99)
    evaluation("Model_2", "tarantula", model, test_loader, parameters)
    model_2_synthsizer.run("ochiai", suspiciousness_threshold=0.99)
    evaluation("Model_2", "ochiai", model, test_loader, parameters)
    model_2_synthsizer.run("barinel", suspiciousness_threshold=0.99)
    evaluation("Model_2", "barinel", model, test_loader, parameters)

    print("Verifying model 2")
    verification = Verification(deepcp2, "pickles/")

    verification.verify("tarantula", suspiciousness_threshold=0.99)
    verification.verify("ochiai", suspiciousness_threshold=0.99)
    verification.verify("barinel", suspiciousness_threshold=0.99)

"""
Synthesizing dataset for model 2
Synthesizing tarantula
[[(86, 0.5462808083566602)], [(37, 0.5552567755385684)], [(27, 0.561562517702166)], [(9, 0.5242393870830873)]]
100%|███████████████████████████████████████████| 79/79 [00:07<00:00, 11.28it/s]
Repairing tarantula
Evaluation on synthesized dataset:
Test set: Average loss: 0.1288, Accuracy: 788/9336 (8%)
Synthesizing ochiai
[[(11, 0.25687637730266294)], [(50, 0.26248831663866756)], [(24, 0.26222402370580283)], [(2, 0.258362686365035)]]
100%|███████████████████████████████████████████| 79/79 [00:07<00:00, 11.21it/s]
Repairing ochiai
Evaluation on synthesized dataset:
Test set: Average loss: 0.0731, Accuracy: 3298/9336 (35%)
Synthesizing barinel
[[(86, 0.07625881780588661)], [(37, 0.0788540396763826)], [(27, 0.08073163040050457)], [(9, 0.070245787538177)]]
100%|███████████████████████████████████████████| 79/79 [00:07<00:00, 11.28it/s]
Repairing barinel
Evaluation on synthesized dataset:
Test set: Average loss: 0.1288, Accuracy: 788/9336 (8%)
Verifying model 2
100%|██████████████████████████████████████| 9336/9336 [00:13<00:00, 683.26it/s]
fails activating faulty paths 6618 8548 0.7742161909218531
100%|██████████████████████████████████████| 9336/9336 [00:13<00:00, 707.44it/s]
fails activating faulty paths 6030 6038 0.998675057966214
100%|██████████████████████████████████████| 9336/9336 [00:13<00:00, 708.27it/s]
fails activating faulty paths 6618 8548 0.7742161909218531
"""