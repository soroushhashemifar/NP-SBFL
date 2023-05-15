import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np

from deepcp_method import DeepCP
from models.train_model_1 import Net
from synthesize import SynthesizeV1, SynthesizeV2, evaluation
from verification import SynthesizedsetVerification, Verification


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

    # deepfault1 = DeepFault(
    #     model_name="Model_mnist_1",
    #     model=model, 
    #     layers_structure=layers_structure, 
    #     input_size=(1, 28, 28), 
    #     train_loader=train_loader, 
    #     device="cpu",
    #     get_relevancy_and_activations_fn = deepcp1.get_relevancy_and_activations
    # )

    # model_1_synthsizer = SynthesizeV1(deepcp1.model_name, model, test_loader, pickles_path=deepcp1.path_to_save_pickles, step_size=1, distance=0.5)
    model_1_synthsizer = SynthesizeV2(deepcp1.model_name, model, test_loader, pickles_path=deepcp1.path_to_save_pickles, num_iterations=5, learning_rate=0.006)
    verification = SynthesizedsetVerification(deepcp1)

    suspiciousness_threshold = 10

    # print("Localizing faults in model 1")
    # deepcp1.run()

    inters = []
    taran = model_1_synthsizer.get_suspicious_neurons("tarantula", suspiciousness_threshold)
    ochiai = model_1_synthsizer.get_suspicious_neurons("ochiai", suspiciousness_threshold)
    for layer in range(5):
        taran_ = taran[layer]
        taran_ = list(map(lambda item: item[0], taran_))
        ochiai_ = ochiai[layer]
        ochiai_ = list(map(lambda item: item[0], ochiai_))
        inter = len(set(taran_).intersection(ochiai_)) / suspiciousness_threshold
        inters.append(inter)

    print("#common neurons in each layer: (tarantula/ochiai)", np.median(inters), np.mean(inters))

    inters = []
    taran = model_1_synthsizer.get_suspicious_neurons("tarantula", suspiciousness_threshold)
    bari = model_1_synthsizer.get_suspicious_neurons("barinel", suspiciousness_threshold)
    for layer in range(5):
        taran_ = taran[layer]
        taran_ = list(map(lambda item: item[0], taran_))
        bari_ = bari[layer]
        bari_ = list(map(lambda item: item[0], bari_))
        inter = len(set(taran_).intersection(bari_)) / suspiciousness_threshold
        inters.append(inter)

    print("#common neurons in each layer: (tarantula/barinel)", np.median(inters), np.mean(inters))

    inters = []
    ochiai = model_1_synthsizer.get_suspicious_neurons("ochiai", suspiciousness_threshold)
    bari = model_1_synthsizer.get_suspicious_neurons("barinel", suspiciousness_threshold)
    for layer in range(5):
        ochiai_ = ochiai[layer]
        ochiai_ = list(map(lambda item: item[0], ochiai_))
        bari_ = bari[layer]
        bari_ = list(map(lambda item: item[0], bari_))
        inter = len(set(ochiai_).intersection(bari_)) / suspiciousness_threshold
        inters.append(inter)

    print("#common neurons in each layer: (ochiai/barinel)", np.median(inters), np.mean(inters))

    print("Synthesizing dataset for model 1")
    parameters = {
            "cuda": False,
            "loss": nn.CrossEntropyLoss(),
        }
    model_1_synthsizer.run("tarantula", suspiciousness_threshold=suspiciousness_threshold)
    evaluation(deepcp1.model_name, "tarantula", model, test_loader, parameters, suspiciousness_threshold=suspiciousness_threshold)
    verification.verify("tarantula", suspiciousness_threshold=suspiciousness_threshold)

    model_1_synthsizer.run("ochiai", suspiciousness_threshold=suspiciousness_threshold)
    evaluation(deepcp1.model_name, "ochiai", model, test_loader, parameters, suspiciousness_threshold=suspiciousness_threshold)
    verification.verify("ochiai", suspiciousness_threshold=suspiciousness_threshold)

    model_1_synthsizer.run("barinel", suspiciousness_threshold=suspiciousness_threshold)
    evaluation(deepcp1.model_name, "barinel", model, test_loader, parameters, suspiciousness_threshold=suspiciousness_threshold)
    verification.verify("barinel", suspiciousness_threshold=suspiciousness_threshold)

    # suspicousness_neurons_tarantula, suspicousness_neurons_ochiai, suspicousness_neurons_barinel = deepfault1.run(num_susp_neurons=5*suspiciousness_threshold)
    # print(suspicousness_neurons_tarantula)