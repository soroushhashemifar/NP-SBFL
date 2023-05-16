import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from deepcp_base import DeepCP
from models.train_model_6 import Net
from synthesize import SynthesizeV1, SynthesizeV2, evaluation
from utils import report_common_neurons_SFLs
from verification import SynthesizedsetVerification


class Model6(DeepCP):
    
    def get_relevancy_and_activations(self, data):
        relevancies, activations = self.lrp_model.forward(data)

        predicted_class = torch.argmax(torch.softmax(activations[-1][1], 1)).item()

        relevancy = [(r[0], r[1].flatten(1)) for r in relevancies]
        activations = [(a[0], a[1].flatten(1)) for a in activations[:-1]]

        relevancy = list(filter(lambda item: item[0] in ["RelevancePropagationConv2d", "RelevancePropagationMaxPool2d", "RelevancePropagationLinear"], relevancy))
        relevancy = list(map(lambda item: item[1], relevancy))
        activations = list(filter(lambda item: item[0] in ["RelevancePropagationConv2d", "RelevancePropagationMaxPool2d", "RelevancePropagationLinear"], activations))
        activations = list(map(lambda item: F.relu(item[1]) if item[0] in ["RelevancePropagationConv2d", "RelevancePropagationLinear"] else item[1], activations))

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
        batch_size=1, shuffle=False)

    model = Net()
    model.load_state_dict(torch.load("models/Model_cifar_3.pth", map_location="cpu"))
    model = model.to("cpu")
    model.eval()

    layers_structure = [
        model.conv1_1, torch.nn.ReLU(),
        model.conv1_2, torch.nn.ReLU(), 
        model.pool, 
        model.conv2_1, torch.nn.ReLU(), 
        model.conv2_2, torch.nn.ReLU(), 
        model.pool, 
        torch.nn.Flatten(1), 
        model.fc1, torch.nn.ReLU(), 
        model.out 
    ]

    deepcp6 = Model6(
        model_name="Model_cifar_3",
        model=model,
        layers_structure=layers_structure, 
        input_size=(3, 32, 32), 
        train_loader=train_loader, 
        device="cpu",
        alpha=0.7, 
        beta=0.6, activation_threshold=0.
    )

    # model_6_synthsizer = SynthesizeV1(deepcp6.model_name, model, test_loader, pickles_path=deepcp6.path_to_save_pickles, output_path=os.path.join(deepcp6.path_to_save_pickles, "synth_v1"), step_size=10, distance=0.1)
    model_6_synthsizer = SynthesizeV2(deepcp6.model_name, model, test_loader, pickles_path=deepcp6.path_to_save_pickles, output_path=os.path.join(deepcp6.path_to_save_pickles, "synth_v2"), num_iterations=5, learning_rate=0.06)
    verification = SynthesizedsetVerification(deepcp6)

    suspiciousness_threshold = 1

    # print("Localizing faults in model 6")
    # deepcp6.run()

    report_common_neurons_SFLs(model_6_synthsizer, suspiciousness_threshold, 7)

    print("Synthesizing dataset for model 6")
    parameters = {
            "cuda": False,
            "loss": nn.CrossEntropyLoss(),
        }
    model_6_synthsizer.run("tarantula", suspiciousness_threshold=suspiciousness_threshold)
    evaluation(deepcp6.model_name, "tarantula", model, model_6_synthsizer.output_path, parameters, suspiciousness_threshold=suspiciousness_threshold)
    verification.verify("tarantula", suspiciousness_threshold=suspiciousness_threshold, synth_dataset_path=model_6_synthsizer.output_path)

    model_6_synthsizer.run("ochiai", suspiciousness_threshold=suspiciousness_threshold)
    evaluation(deepcp6.model_name, "ochiai", model, model_6_synthsizer.output_path, parameters, suspiciousness_threshold=suspiciousness_threshold)
    verification.verify("ochiai", suspiciousness_threshold=suspiciousness_threshold, synth_dataset_path=model_6_synthsizer.output_path)

    model_6_synthsizer.run("barinel", suspiciousness_threshold=suspiciousness_threshold)
    evaluation(deepcp6.model_name, "barinel", model, model_6_synthsizer.output_path, parameters, suspiciousness_threshold=suspiciousness_threshold)
    verification.verify("barinel", suspiciousness_threshold=suspiciousness_threshold, synth_dataset_path=model_6_synthsizer.output_path)
