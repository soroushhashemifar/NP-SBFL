import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from deepcp_base import DeepCP
from models.train_model_1 import Net
from synthesize import SynthesizeV1, SynthesizeV2, evaluation
from utils import report_common_neurons_SFLs
from verification import SynthesizedsetVerification


class Model1(DeepCP):
    
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
        batch_size=1, shuffle=False)

    model = Net()
    model.load_state_dict(torch.load("models/Model_mnist_1.pth", map_location="cpu"))
    model = model.to("cpu")
    model.eval()

    layers_structure = [
        torch.nn.Flatten(1), 
        model.fc1, torch.nn.ReLU(),
        model.fc2, torch.nn.ReLU(),
        model.fc3, torch.nn.ReLU(),
        model.fc4, torch.nn.ReLU(),
        model.fc5, torch.nn.ReLU(),
        model.out
    ]

    deepcp1 = Model1(
        model_name="Model_mnist_1",
        model=model, 
        layers_structure=layers_structure, 
        input_size=(1, 28, 28), 
        train_loader=train_loader, 
        device="cpu",
        alpha=0.7, 
        beta=0.6, activation_threshold=0.
    )

    # model_1_synthsizer = SynthesizeV1(deepcp1.model_name, model, test_loader, pickles_path=deepcp1.path_to_save_pickles, output_path=os.path.join(deepcp1.path_to_save_pickles, "synth_v1"), step_size=1, distance=0.5)
    model_1_synthsizer = SynthesizeV2(deepcp1.model_name, model, test_loader, pickles_path=deepcp1.path_to_save_pickles, output_path=os.path.join(deepcp1.path_to_save_pickles, "synth_v2"), num_iterations=5, learning_rate=0.006)
    verification = SynthesizedsetVerification(deepcp1)

    suspiciousness_threshold = 1

    # print("Localizing faults in model 1")
    # deepcp1.run()

    report_common_neurons_SFLs(model_1_synthsizer, suspiciousness_threshold, 5)

    print("Synthesizing dataset for model 1")
    parameters = {
            "cuda": False,
            "loss": nn.CrossEntropyLoss(),
        }
    # model_1_synthsizer.run("tarantula", suspiciousness_threshold=suspiciousness_threshold)
    # evaluation(deepcp1.model_name, "tarantula", model, model_1_synthsizer.output_path, parameters, suspiciousness_threshold=suspiciousness_threshold)
    verification.verify("tarantula", suspiciousness_threshold=suspiciousness_threshold, synth_dataset_path=model_1_synthsizer.output_path)

    # model_1_synthsizer.run("ochiai", suspiciousness_threshold=suspiciousness_threshold)
    # evaluation(deepcp1.model_name, "ochiai", model, model_1_synthsizer.output_path, parameters, suspiciousness_threshold=suspiciousness_threshold)
    verification.verify("ochiai", suspiciousness_threshold=suspiciousness_threshold, synth_dataset_path=model_1_synthsizer.output_path)

    # model_1_synthsizer.run("barinel", suspiciousness_threshold=suspiciousness_threshold)
    # evaluation(deepcp1.model_name, "barinel", model, model_1_synthsizer.output_path, parameters, suspiciousness_threshold=suspiciousness_threshold)
    verification.verify("barinel", suspiciousness_threshold=suspiciousness_threshold, synth_dataset_path=model_1_synthsizer.output_path)