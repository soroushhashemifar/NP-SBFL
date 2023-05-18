import sys
sys.path.insert(0, "..")
import os

import torch
import torch.nn as nn
from models.train_model_1 import Net
from synthesize import evaluation
from torchvision import datasets, transforms

from deepfault_base import DeepFault, Synthesize_DF, Verification_DF
from localization_Model_1 import Model1


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
        batch_size=128, shuffle=False)

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
        activation_threshold=0.
    )

    deepfault1 = DeepFault(
        model_name=deepcp1.model_name,
        model=model, 
        path_to_save_pickles="pickles_deepfault",
        layers_structure=layers_structure, 
        input_size=deepcp1.input_size, 
        train_loader=train_loader, 
        device="cpu",
        get_relevancy_and_activations_fn=deepcp1.get_relevancy_and_activations
    )

    suspiciousness_threshold = 10
    num_susp_neurons = suspiciousness_threshold * 5

    model_1_synthsizer = Synthesize_DF(deepfault1.model_name, model, test_loader, pickles_path=deepfault1.path_to_save_pickles, output_path=os.path.join(deepfault1.path_to_save_pickles, "synth_v1"), step_size=1, distance=0.1)
    verification = Verification_DF(deepfault1)

    # print("Localizing faults in model 1")
    # deepfault1.run()

    print("Synthesizing dataset for model 1")
    parameters = {
            "cuda": False,
            "loss": nn.CrossEntropyLoss(),
        }
    model_1_synthsizer.run("tarantula", suspiciousness_threshold=num_susp_neurons)
    evaluation(deepfault1.model_name, "tarantula", model, model_1_synthsizer.output_path, parameters, suspiciousness_threshold=num_susp_neurons)
    verification.verify("tarantula", suspiciousness_threshold=num_susp_neurons, synth_dataset_path=model_1_synthsizer.output_path)

    model_1_synthsizer.run("ochiai", suspiciousness_threshold=num_susp_neurons)
    evaluation(deepfault1.model_name, "ochiai", model, model_1_synthsizer.output_path, parameters, suspiciousness_threshold=num_susp_neurons)
    verification.verify("ochiai", suspiciousness_threshold=num_susp_neurons, synth_dataset_path=model_1_synthsizer.output_path)

    model_1_synthsizer.run("barinel", suspiciousness_threshold=num_susp_neurons)
    evaluation(deepfault1.model_name, "barinel", model, model_1_synthsizer.output_path, parameters, suspiciousness_threshold=num_susp_neurons)
    verification.verify("barinel", suspiciousness_threshold=num_susp_neurons, synth_dataset_path=model_1_synthsizer.output_path)
