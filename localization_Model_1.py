import torch
import torch.nn as nn
from torchvision import datasets, transforms

from deepcp_method import DeepCP
from models.train_model_1 import Net
from synthesize import Synthesize, evaluation
from verification import Verification


class Model1(DeepCP):
    
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
    model.load_state_dict(torch.load("models/Model_1.pth", map_location="cpu"))
    model = model.to("cpu")
    model.eval()

    layers_structure = [
        torch.nn.Flatten(1), 
        model.fc1, torch.nn.ReLU(),
        model.fc2, torch.nn.ReLU(),
        model.fc3
    ]

    print("Localizing faults in model 1")
    deepcp1 = Model1(
        model_name="Model_1",
        model=model, 
        layers_structure=layers_structure, 
        input_size=(1, 28, 28), 
        train_loader=train_loader, 
        device="cpu",
        alpha=0.9, 
        beta=0.6, activation_threshold=0.
    )
    deepcp1.run()

    print("Synthesizing dataset for model 1")
    model_1_synthsizer = Synthesize("Model_1", model, test_loader, pickles_path="pickles/", step_size=10, distance=0.9)

    parameters = {
            "cuda": False,
            "loss": nn.CrossEntropyLoss(),
        }
    model_1_synthsizer.run("tarantula", suspiciousness_threshold=0.99)
    evaluation("Model_1", "tarantula", model, test_loader, parameters)
    model_1_synthsizer.run("ochiai", suspiciousness_threshold=0.99)
    evaluation("Model_1", "ochiai", model, test_loader, parameters)
    model_1_synthsizer.run("barinel", suspiciousness_threshold=0.99)
    evaluation("Model_1", "barinel", model, test_loader, parameters)

    print("Verifying model 1")
    verification = Verification(deepcp1, "pickles/")

    verification.verify("tarantula", suspiciousness_threshold=0.99)
    verification.verify("ochiai", suspiciousness_threshold=0.99)
    verification.verify("barinel", suspiciousness_threshold=0.99)

"""
Synthesizing dataset for model 1
Synthesizing tarantula
[[(11, 0.5383477344910744)], [(8, 0.5178378862609527)]]
100%|███████████████████████████████████████████| 79/79 [00:05<00:00, 13.31it/s]
Repairing tarantula
Evaluation on synthesized dataset:
Test set: Average loss: 0.1400, Accuracy: 995/9326 (11%)
Synthesizing ochiai
[[(45, 0.26349639851717904)], [(14, 0.2654588423301316)]]
100%|███████████████████████████████████████████| 79/79 [00:05<00:00, 13.34it/s]
Repairing ochiai
Evaluation on synthesized dataset:
Test set: Average loss: 0.0795, Accuracy: 1584/9326 (17%)
Synthesizing barinel
[[(11, 0.07761295200354568)], [(8, 0.07192146953056666)]]
100%|███████████████████████████████████████████| 79/79 [00:05<00:00, 13.24it/s]
Repairing barinel
Evaluation on synthesized dataset:
Test set: Average loss: 0.1400, Accuracy: 995/9326 (11%)
Verifying model 1
100%|█████████████████████████████████████| 9326/9326 [00:08<00:00, 1115.96it/s]
fails activating faulty paths 8331 8331 1.0
100%|█████████████████████████████████████| 9326/9326 [00:08<00:00, 1113.46it/s]
fails activating faulty paths 7742 7742 1.0
100%|█████████████████████████████████████| 9326/9326 [00:08<00:00, 1114.17it/s]
fails activating faulty paths 8331 8331 1.0
"""