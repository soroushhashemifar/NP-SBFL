import sys
sys.path.insert(0, "..")
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np

from deepfault_method import DeepFault
from localization_Model_1 import Model1
from models.train_model_1 import Net
from synthesize import SynthesizeV1, evaluation
from verification import Verification


if __name__ == "__main__":
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../models/data', train=True, transform=transforms.Compose([
                        transforms.ToTensor(),
                    ])),
        batch_size=1, shuffle=False)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../models/data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                    ])),
        batch_size=1, shuffle=False)

    model = Net()
    model.load_state_dict(torch.load("../models/Model_mnist_1.pth", map_location="cpu"))
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
        alpha=0.9, 
        beta=0.6, activation_threshold=0.
    )

    deepfault1 = DeepFault(
        model_name="Model_mnist_1",
        model=model, 
        layers_structure=layers_structure, 
        input_size=(1, 28, 28), 
        train_loader=train_loader, 
        device="cpu",
        get_relevancy_and_activations_fn = deepcp1.get_relevancy_and_activations
    )

    model_1_synthsizer = SynthesizeV1(deepcp1.model_name, model, test_loader, pickles_path=deepcp1.path_to_save_pickles, step_size=1, distance=0.1)
    verification = Verification(deepcp1)

    suspiciousness_threshold = 1

    print("Localizing faults in model 1")
    deepfault1.run()

    print("Synthesizing dataset for model 1")
    parameters = {
            "cuda": False,
            "loss": nn.CrossEntropyLoss(),
        }
    model_1_synthsizer.run("tarantula", suspiciousness_threshold=suspiciousness_threshold)
    evaluation(deepcp1.model_name, "tarantula", model, test_loader, parameters, suspiciousness_threshold=suspiciousness_threshold)
    model_1_synthsizer.run("ochiai", suspiciousness_threshold=suspiciousness_threshold)
    evaluation(deepcp1.model_name, "ochiai", model, test_loader, parameters, suspiciousness_threshold=suspiciousness_threshold)
    model_1_synthsizer.run("barinel", suspiciousness_threshold=suspiciousness_threshold)
    evaluation(deepcp1.model_name, "barinel", model, test_loader, parameters, suspiciousness_threshold=suspiciousness_threshold)

    print("Verifying model 1")
    verification.verify("tarantula", suspiciousness_threshold=suspiciousness_threshold)
    verification.verify("ochiai", suspiciousness_threshold=suspiciousness_threshold)
    verification.verify("barinel", suspiciousness_threshold=suspiciousness_threshold)