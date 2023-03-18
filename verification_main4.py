import pickle
import time
import random
import os

import torch
import tqdm
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np

import lrp_src
from models.train_model_1 import Net as Net1
from models.train_model_2 import Net as Net2
from models.train_model_3 import Net as Net3
from localization_Model_1_main import Model1
from localization_Model_2_main import Model2
from localization_Model_3_main import Model3


class Verification:

    def __init__(self, deepcp, pickles_path):
        self.deepcp = deepcp
        self.pickles_path = pickles_path

    def get_faulty_paths_vector(self, SFL_strategy):
        with open(os.path.join(self.pickles_path, f"{self.deepcp.model_name}_{SFL_strategy}_objects.pickle"), 'rb') as handle:
            self.objects_dict = pickle.load(handle)

        layerwise_suspicousness_scores = self.objects_dict["layerwise_suspicousness_scores"]
        suspicousness_scores_vector = np.zeros(sum(list(map(lambda item: len(item), layerwise_suspicousness_scores))))
        indices = np.cumsum([0] + self.deepcp.layer_shapes)
        for i, layer_scores in zip(range(indices.shape[0]-1), layerwise_suspicousness_scores):
            layer_scores_ = list(filter(lambda item: not np.isnan(item[1]) and item[1] > 0.99, layer_scores))
            if len(layer_scores_) == 0:
                layer_scores_ = [max(layer_scores, key=lambda item: not np.isnan(item[1]) and item[1])]

            print(layer_scores_)

            for neuron_index, _ in layer_scores_:
                suspicousness_scores_vector[indices[i]:indices[i+1]][neuron_index] = 1

        return suspicousness_scores_vector

    def calculate_tests_ratio(self, faulty_paths_vector):
        num_failed_tests = 0
        num_total_tests_failed = 0
        num_passed_tests = 0
        num_total_tests_passed = 0
        for data, target in tqdm.tqdm(self.test_loader):
            if self.deepcp.cuda:
                data, target = data.cuda(), target.cuda()

            data, target = Variable(data), Variable(target)

            cdp_representation, critical_neurons_layers_test, predicted_class, activation_mask = self.deepcp.generate_cdp_representation(data)
            if cdp_representation is None and critical_neurons_layers_test is None and predicted_class is None:
                continue

            if target != predicted_class:
                num_total_tests_failed += 1

                if all(activation_mask[faulty_paths_vector.astype(bool)]):
                    num_failed_tests += 1
            else:
                num_total_tests_passed += 1

                if not all(activation_mask[faulty_paths_vector.astype(bool)]):
                    num_passed_tests += 1

        return num_failed_tests, num_total_tests_failed, num_passed_tests, num_total_tests_passed

    def verify(self, SFL_strategy):
        with open(f"./pickles/synthesized_dataset_{self.deepcp.model_name}_{SFL_strategy}.pickle", 'rb') as handle:
            synthesized_dataset = pickle.load(handle)

        synth_dataset = SynthesizedDataset(synthesized_dataset)
        self.test_loader = torch.utils.data.DataLoader(
            synth_dataset,
            batch_size=1, shuffle=False)

        faulty_paths_vector = self.get_faulty_paths_vector(SFL_strategy)
        num_failed_tests, num_total_tests_failed, num_passed_tests, num_total_tests_passed = self.calculate_tests_ratio(faulty_paths_vector)

        print("fails activating faulty paths", num_failed_tests, num_total_tests_failed, num_failed_tests / num_total_tests_failed)
        print("passes not activating faulty paths", num_passed_tests, num_total_tests_passed, num_passed_tests / num_total_tests_passed)

    def run(self):
        self.verify("tarantula")
        self.verify("ochiai")
        self.verify("barinel")


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


def verify_main1():
    ALPHA = 0.99

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('models/data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=1, shuffle=True)

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

    time_budget = 3 # seconds
    decision_birch_pickle_path = 'pickles/Model_1_decision_birch.pickle'

    data_sample_preprocess_fn = lambda item: item.view(-1, 784)

    verification = Verification(deepcp1, test_loader, time_budget, data_sample_preprocess_fn, decision_birch_pickle_path)
    verification.run()

def verify_main2():
    model = Net2()
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

    deepcp2 = Model2(
        model_name="Model_2",
        model=model, 
        layers_structure=layers_structure, 
        input_size=(1, 28, 28),
        device="cpu",
        alpha=1.5
    )

    verification = Verification(deepcp2, "pickles/")
    verification.run()

def verify_main3():
    ALPHA = 0.4

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('models/data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])),
        batch_size=1, shuffle=True)

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

    data_sample_preprocess_fn = lambda item: item

    verification = Verification(deepcp3, test_loader, data_sample_preprocess_fn, "pickles/")
    verification.run()

if __name__ == "__main__":
    # verify_main1()
    verify_main2()
    # verify_main3()