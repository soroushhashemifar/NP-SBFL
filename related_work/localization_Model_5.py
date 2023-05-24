import sys
sys.path.insert(0, "..")
import os

import torch
import torch.nn as nn
from models.train_model_5 import Net
from synthesize import SynthesizeV1, evaluation
from torchvision import datasets, transforms

from deepfault_base import DeepFault, Synthesize_DF, Verification_DF
from localization_Model_5 import Model5


if __name__ == "__main__":
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../models/data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                    ])),
        batch_size=1, shuffle=False)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../models/data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                    ])),
        batch_size=128, shuffle=False)

    model = Net()
    model.load_state_dict(torch.load("../models/Model_cifar_2.pth", map_location="cpu"))
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
        model.fc2, torch.nn.ReLU(), 
        model.out 
    ]

    deepcp5 = Model5(
        model_name="Model_cifar_2",
        model=model,
        layers_structure=layers_structure, 
        input_size=(3, 32, 32), 
        train_loader=train_loader, 
        device="cpu",
        activation_threshold=0.
    )

    deepfault5 = DeepFault(
        model_name=deepcp5.model_name,
        model=model, 
        path_to_save_pickles="pickles_deepfault",
        layers_structure=layers_structure, 
        input_size=deepcp5.input_size, 
        train_loader=train_loader, 
        device="cpu",
        get_relevancy_and_activations_fn=deepcp5.get_relevancy_and_activations
    )

    suspiciousness_threshold = 50
    num_susp_neurons = suspiciousness_threshold * 8

    model_5_synthsizer = Synthesize_DF(deepfault5.model_name, model, test_loader, pickles_path=deepfault5.path_to_save_pickles, output_path=os.path.join(deepfault5.path_to_save_pickles, "synth_v1"), step_size=10, distance=0.1)
    verification = Verification_DF(deepfault5)

    # print("Localizing faults in model 5")
    # deepfault5.run()

    print("Synthesizing dataset for model 5")
    parameters = {
            "cuda": False,
            "loss": nn.CrossEntropyLoss(),
        }
    model_5_synthsizer.run("tarantula", suspiciousness_threshold=num_susp_neurons)
    evaluation(deepfault5.model_name, "tarantula", model, model_5_synthsizer.output_path, parameters, suspiciousness_threshold=num_susp_neurons)
    verification.verify("tarantula", suspiciousness_threshold=num_susp_neurons, synth_dataset_path=model_5_synthsizer.output_path)

    model_5_synthsizer.run("ochiai", suspiciousness_threshold=num_susp_neurons)
    evaluation(deepfault5.model_name, "ochiai", model, model_5_synthsizer.output_path, parameters, suspiciousness_threshold=num_susp_neurons)
    verification.verify("ochiai", suspiciousness_threshold=num_susp_neurons, synth_dataset_path=model_5_synthsizer.output_path)

    model_5_synthsizer.run("barinel", suspiciousness_threshold=num_susp_neurons)
    evaluation(deepfault5.model_name, "barinel", model, model_5_synthsizer.output_path, parameters, suspiciousness_threshold=num_susp_neurons)
    verification.verify("barinel", suspiciousness_threshold=num_susp_neurons, synth_dataset_path=model_5_synthsizer.output_path)
