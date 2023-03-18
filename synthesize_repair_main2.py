import os
import pickle
import torch
import torchvision.transforms as transforms
from torchvision import datasets, transforms
import tqdm
from models.train_model_1 import Net as Net1
from localization_Model_1_main import Model1
from models.train_model_2 import Net as Net2
from localization_Model_2_main import Model2
from utils import test, train
import torch.optim as optim
from models.train_model_3 import Net as Net3
from localization_Model_3_main import Model3
import lrp_src
from torch.autograd import Variable
import numpy as np
import torch.nn as nn


class Synthesize:

    def __init__(self, model_name, model, test_loader, pickles_path, step_size=5, distance=0.1):
        self.step_size = step_size
        self.distance = distance
        self.model_name = model_name

        self.model = model
        self.test_loader = test_loader
        self.pickles_path = pickles_path

    def synthsize_image(self, data, perturbed_data, gradients):
        for i in range(data.shape[2]):
            for j in range(data.shape[3]):
                sum_grad = torch.zeros((data.shape[0], data.shape[1]))
                for k in range(len(gradients)):
                    sum_grad += gradients[k][:, :, i, j]

                avg_grad = sum_grad / len(gradients)
                avg_grad = avg_grad * self.step_size

                # Clipping gradients.
                avg_grad[avg_grad > self.distance] = self.distance
                avg_grad[avg_grad < -self.distance] = -self.distance

                perturbed_data[:, :, i, j] = data[:, :, i, j] + avg_grad

        return perturbed_data

    def applyDomainConstraints(self, perturbed_data):
        perturbed_data[perturbed_data > 1] = 1
        perturbed_data[perturbed_data < 0] = 0

        return perturbed_data

    def get_suspicious_neurons(self, SFL_strategy):
        with open(os.path.join(self.pickles_path, f"{self.model_name}_{SFL_strategy}_objects.pickle"), 'rb') as handle:
            self.objects_dict = pickle.load(handle)

        layerwise_suspicousness_scores = self.objects_dict["layerwise_suspicousness_scores"]
        suspicousness_neurons_per_layer = []
        for layer_scores in layerwise_suspicousness_scores:
            layer_scores_ = list(filter(lambda item: not np.isnan(item[1]) and item[1] > 0.99, layer_scores))
            if len(layer_scores_) == 0:
                layer_scores_ = [max(layer_scores, key=lambda item: not np.isnan(item[1]) and item[1])]

            suspicousness_neurons_per_layer.append(layer_scores_)

        return suspicousness_neurons_per_layer

    def synthesize_testset(self, suspicousness_neurons_per_layer):
        synthesized_dataset = []
        for data, target in tqdm.tqdm(self.test_loader):
            inputs = torch.autograd.Variable(data, requires_grad=True)
            outputs, features = self.model(inputs, return_logits=True)
            outputs = torch.softmax(outputs, 1)
            outputs = torch.argmax(outputs, 1)

            mask = outputs == target
            gradients = []
            for layer_idx in range(len(suspicousness_neurons_per_layer)):
                for sn_index, _ in suspicousness_neurons_per_layer[layer_idx]:
                    features_ = features[layer_idx][:, [sn_index]]
                    gradients_ = torch.autograd.grad(outputs=features_, inputs=inputs, grad_outputs=torch.ones(features_.size()).to("cpu"), retain_graph=True)[0]
                    gradients_ = gradients_[mask]
                    gradients.append(gradients_)

            data = data[mask]
            perturbed_data = data.clone()
            perturbed_data = self.synthsize_image(data, perturbed_data, gradients)
            perturbed_data = self.applyDomainConstraints(perturbed_data)

            data = data.permute(0, 2, 3, 1).numpy()
            perturbed_data = perturbed_data.permute(0, 2, 3, 1).numpy()
            target = target[mask].numpy()
            for datam, perturbed_datam, label in zip(data, perturbed_data, target):
                synthesized_dataset.append((datam, perturbed_datam, label))

        return synthesized_dataset

    def run(self):
        for SFL_strategy in [
            "tarantula", 
            "ochiai", 
            "barinel"
            ]:
            print(f"Synthesizing {SFL_strategy}")
            suspicousness_neurons_per_layer = self.get_suspicious_neurons(SFL_strategy)
            print(suspicousness_neurons_per_layer)
            synthesized_dataset = self.synthesize_testset(suspicousness_neurons_per_layer)
            with open(f"./pickles/synthesized_dataset_{self.model_name}_{SFL_strategy}.pickle", 'wb') as handle:
                pickle.dump(synthesized_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)


class SynthesizedDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, transforms=None):
        self.dataset = dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, perturbed_data, label = self.dataset[idx]
        perturbed_data = torch.tensor(perturbed_data).permute(2, 0, 1)

        return perturbed_data, label


def repair_method(model_name, SFL_strategy, model, test_loader, parameters):
    print(f"Repairing {SFL_strategy}")

    with open(f"./pickles/synthesized_dataset_{model_name}_{SFL_strategy}.pickle", 'rb') as handle:
        synthesized_dataset = pickle.load(handle)

    synth_dataset = SynthesizedDataset(synthesized_dataset)
    synth_loader = torch.utils.data.DataLoader(
        synth_dataset,
        batch_size=128, shuffle=False)

    print("Evaluation on synthesized dataset:")
    test(model, synth_loader, parameters, scheduler=None)

    # print("Evaluation on testset (before finetune):")
    # test(model, test_loader, parameters, scheduler=None)

    # optimizer = optim.SGD(model.parameters(), lr=parameters["learning_rate"], momentum=0.9)
    # for epoch in range(1, parameters["num_epochs"] + 1):
    #     train(epoch, model, synth_loader, optimizer, parameters)

    # print("Evaluation on testset (after finetune):")
    # test(model, test_loader, parameters, scheduler=None)

    return model

def repair_model_1():
    print("Repair model 1 started")

    ALPHA = 0.99
    lrp_src.lrp_layers.top_k_percent = ALPHA

    model = Net1()
    model.load_state_dict(torch.load("models/Model_1.pth", map_location="cpu"))
    model = model.to("cpu")
    model.eval()

    layers_structure = [
        torch.nn.Flatten(1), 
        model.fc1, torch.nn.ReLU(),
        model.fc2, torch.nn.ReLU(),
        model.fc3
    ]

    deepcp1 = Model1(
        model_name="Model_1",
        model=model, 
        layers_structure=layers_structure, 
        input_size=(1, 28, 28),
        device="cpu",
        alpha=ALPHA
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('models/data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=1, shuffle=False)

    metric_thresholds = [("tarantula", 0.99), ("ochiai", 0.26), ("barinel", 0.18)]

    data_sample_preprocess_fn = lambda item: item.view(-1, 784)

    model_1_synthsizer = Synthesize(deepcp1, model, test_loader, metric_thresholds, (0.1307,), (0.3081,), data_sample_preprocess_fn, step_size=20, distance=0.1)
    model_1_synthsizer.run()

    # parameters = {
    #         "cuda": False,
    #         "loss": "nll_loss",
    #         "log_interval": 1000,
    #         "learning_rate": 0.1,
    #         "num_epochs": 20,
    #     }

    # for SFL_strategy, _ in metric_thresholds:
    #     model = Net1()
    #     model.load_state_dict(torch.load("models/Model_1.pth", map_location="cpu"))
    #     model = model.to("cpu")
    #     repaired_model = repair_method(deepcp1, SFL_strategy, model, test_loader, parameters)
    #     torch.save(repaired_model.state_dict(), f"models/{deepcp1.model_name}_repaired_{SFL_strategy}.pth")

def repair_model_2():
    print("Repair model 2 started")

    model = Net2()
    model.load_state_dict(torch.load("models/Model_2.pth", map_location="cpu"))
    model = model.to("cpu")
    model.eval()

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('models/data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        # transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=128, shuffle=False)

    model_2_synthsizer = Synthesize("Model_2", model, test_loader, pickles_path="./pickles", step_size=10, distance=0.5)
    model_2_synthsizer.run()

    parameters = {
            "cuda": False,
            "loss": nn.CrossEntropyLoss(),
            "log_interval": 1000,
            "learning_rate": 0.1,
            "num_epochs": 20,
        }

    for SFL_strategy in ["tarantula", "ochiai", "barinel"]:
        model = Net2()
        model.load_state_dict(torch.load("models/Model_2.pth", map_location="cpu"))
        model = model.to("cpu")
        repaired_model = repair_method("Model_2", SFL_strategy, model, test_loader, parameters)
    #     torch.save(repaired_model.state_dict(), f"models/{deepcp2.model_name}_repaired_{SFL_strategy}.pth")

def repair_model_3():
    print("Repair model 3 started")

    ALPHA = 0.9
    lrp_src.lrp_layers.top_k_percent = ALPHA

    model = Net3()
    model.load_state_dict(torch.load("models/Model_3.pth", map_location="cpu"))
    model = model.to("cpu")
    model.eval()

    layers_structure = [
        model.conv1_1, torch.nn.ReLU(), model.pool, 
        model.conv2_1, torch.nn.ReLU(), model.pool, 
        model.conv3_1, torch.nn.ReLU(), model.pool, 
        model.conv4_1, torch.nn.ReLU(), model.pool, 
        torch.nn.Flatten(1), 
        model.fc1, torch.nn.ReLU(), 
        model.fc2, torch.nn.ReLU(), 
        model.fc3 
    ]

    deepcp3 = Model3(
        model_name="Model_3",
        model=model, 
        layers_structure=layers_structure, 
        input_size=(3, 32, 32),
        device="cpu",
        alpha=ALPHA
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('models/data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])),
        batch_size=128, shuffle=False)

    metric_thresholds = [("tarantula", 0.90802413), ("ochiai", 0.37238748), ("barinel", 0.3964497)]

    model_3_synthsizer = Synthesize(deepcp3, model, test_loader, metric_thresholds, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), step_size=1, distance=0.1)
    model_3_synthsizer.run()

    parameters = {
            "cuda": False,
            "loss": "cross_entropy",
            "log_interval": 1000,
            "learning_rate": 0.01,
            "num_epochs": 20,
        }

    for SFL_strategy, _ in metric_thresholds:
        model = Net3()
        model.load_state_dict(torch.load("models/Model_3.pth", map_location="cpu"))
        model = model.to("cpu")
        repaired_model = repair_method(deepcp3, SFL_strategy, model, test_loader, parameters)
        torch.save(repaired_model.state_dict(), f"models/{deepcp3.model_name}_repaired_{SFL_strategy}.pth")

if __name__ == "__main__":
    # repair_model_1()
    repair_model_2()
    # repair_model_3()