import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np

from deepcp_method import DeepCP
from models.train_model_4 import Net
from synthesize import SynthesizeV1, SynthesizeV2, evaluation
from verification import Verification


class Model4(DeepCP):
    
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
    model.load_state_dict(torch.load("models/Model_cifar_1.pth", map_location="cpu"))
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
        model.fc3, torch.nn.ReLU(), 
        model.fc4, torch.nn.ReLU(), 
        model.out 
    ]

    deepcp4 = Model4(
        model_name="Model_cifar_1",
        model=model,
        layers_structure=layers_structure, 
        input_size=(3, 32, 32), 
        train_loader=train_loader, 
        device="cpu",
        alpha=0.7, 
        beta=0.6, activation_threshold=0.
    )

    # model_4_synthsizer = SynthesizeV1(deepcp4.model_name, model, test_loader, pickles_path=deepcp4.path_to_save_pickles, step_size=10, distance=0.1)
    model_4_synthsizer = SynthesizeV2(deepcp4.model_name, model, test_loader, pickles_path=deepcp4.path_to_save_pickles, num_iterations=5, learning_rate=0.08)
    verification = Verification(deepcp4)

    suspiciousness_threshold = 10

    # print("Localizing faults in model 4")
    # deepcp4.run()

    inters = []
    taran = model_4_synthsizer.get_suspicious_neurons("tarantula", suspiciousness_threshold)
    ochiai = model_4_synthsizer.get_suspicious_neurons("ochiai", suspiciousness_threshold)
    for layer in range(10):
        taran_ = taran[layer]
        taran_ = list(map(lambda item: item[0], taran_))
        ochiai_ = ochiai[layer]
        ochiai_ = list(map(lambda item: item[0], ochiai_))
        inter = len(set(taran_).intersection(ochiai_)) / suspiciousness_threshold
        inters.append(inter)

    print("#common neurons in each layer: (tarantula/ochiai)", np.median(inters), np.mean(inters))

    inters = []
    taran = model_4_synthsizer.get_suspicious_neurons("tarantula", suspiciousness_threshold)
    bari = model_4_synthsizer.get_suspicious_neurons("barinel", suspiciousness_threshold)
    for layer in range(10):
        taran_ = taran[layer]
        taran_ = list(map(lambda item: item[0], taran_))
        bari_ = bari[layer]
        bari_ = list(map(lambda item: item[0], bari_))
        inter = len(set(taran_).intersection(bari_)) / suspiciousness_threshold
        inters.append(inter)

    print("#common neurons in each layer: (tarantula/barinel)", np.median(inters), np.mean(inters))

    inters = []
    ochiai = model_4_synthsizer.get_suspicious_neurons("ochiai", suspiciousness_threshold)
    bari = model_4_synthsizer.get_suspicious_neurons("barinel", suspiciousness_threshold)
    for layer in range(10):
        ochiai_ = ochiai[layer]
        ochiai_ = list(map(lambda item: item[0], ochiai_))
        bari_ = bari[layer]
        bari_ = list(map(lambda item: item[0], bari_))
        inter = len(set(ochiai_).intersection(bari_)) / suspiciousness_threshold
        inters.append(inter)

    print("#common neurons in each layer: (ochiai/barinel)", np.median(inters), np.mean(inters))

    print("Synthesizing dataset for model 4")
    parameters = {
            "cuda": False,
            "loss": nn.CrossEntropyLoss(),
        }

    model_4_synthsizer.run("tarantula", suspiciousness_threshold=suspiciousness_threshold)
    evaluation(deepcp4.model_name, "tarantula", model, test_loader, parameters, suspiciousness_threshold=suspiciousness_threshold)
    verification.verify("tarantula", suspiciousness_threshold=suspiciousness_threshold)
    
    model_4_synthsizer.run("ochiai", suspiciousness_threshold=suspiciousness_threshold)
    evaluation(deepcp4.model_name, "ochiai", model, test_loader, parameters, suspiciousness_threshold=suspiciousness_threshold)
    verification.verify("ochiai", suspiciousness_threshold=suspiciousness_threshold)

    model_4_synthsizer.run("barinel", suspiciousness_threshold=suspiciousness_threshold)
    evaluation(deepcp4.model_name, "barinel", model, test_loader, parameters, suspiciousness_threshold=suspiciousness_threshold)
    verification.verify("barinel", suspiciousness_threshold=suspiciousness_threshold)