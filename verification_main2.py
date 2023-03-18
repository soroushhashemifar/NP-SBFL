import pickle
import time
import random
import os

import torch
import tqdm
from torch.autograd import Variable
from torchvision import datasets, transforms

import lrp_src
from models.train_model_1 import Net as Net1
from models.train_model_2 import Net as Net2
from models.train_model_3 import Net as Net3
from localization_Model_1_main import Model1
from localization_Model_2_main import Model2
from localization_Model_3_main import Model3


class Verification:

    def __init__(self, deepcp, test_loader, time_budget, data_sample_preprocess_fn, decision_birch_pickle_path):
        self.deepcp = deepcp
        self.test_loader = test_loader
        self.time_budget = time_budget
        self.data_sample_preprocess_fn = data_sample_preprocess_fn

        with open(decision_birch_pickle_path, 'rb') as handle:
            self.decision_birch = pickle.load(handle)

    def calculate_FHR(self, faulty_cdps):
        num_corrects = 0
        num_total_samples = 0
        for data, target in tqdm.tqdm(self.test_loader):
            if self.deepcp.cuda:
                data, target = data.cuda(), target.cuda()

            data, target = Variable(data), Variable(target)

            cdp_representation, critical_neurons_layers_test, predicted_class, _ = self.deepcp.generate_cdp_representation(self.data_sample_preprocess_fn(data))
            if (cdp_representation is None and critical_neurons_layers_test is None and predicted_class is None) or predicted_class not in self.decision_birch.keys():
                continue

            predicted_cluster = self.decision_birch[predicted_class].predict(cdp_representation[None, ...])[0]
            if (predicted_class, predicted_cluster) in faulty_cdps:
                num_total_samples += 1
                if target != predicted_class:
                    num_corrects += 1

        return num_corrects, num_total_samples

    def calculate_HFR(self, faulty_cdps):
        num_failures = 0
        num_total_samples = 0
        for data, target in tqdm.tqdm(self.test_loader):
            if self.deepcp.cuda:
                data, target = data.cuda(), target.cuda()

            data, target = Variable(data), Variable(target)

            cdp_representation, critical_neurons_layers_test, predicted_class, _ = self.deepcp.generate_cdp_representation(self.data_sample_preprocess_fn(data))
            if (cdp_representation is None and critical_neurons_layers_test is None and predicted_class is None) or predicted_class not in self.decision_birch.keys():
                continue

            predicted_cluster = self.decision_birch[predicted_class].predict(cdp_representation[None, ...])[0]

            if (predicted_class, predicted_cluster) not in faulty_cdps:
                num_total_samples += 1
                if target == predicted_class:
                    num_failures += 1

        return num_failures, num_total_samples

    def verify(self, SFL_strategy):
        with open(os.path.join("results", f"{self.deepcp.model_name}_{SFL_strategy}.txt")) as file:
            content = file.readlines()
            content = list(map(lambda item: item.strip(), content))
            content = list(map(lambda item: item.split("\t"), content))
            scores = list(map(lambda item: (eval(item[0]), float(item[1]), eval(item[2])), content))

        thresholds = sorted(list(set(map(lambda item: item[1], scores))), reverse=True)
        for threshold in thresholds:
            print(f"THRESHOLD: {threshold}")
            scores_ = list(filter(lambda item: item[1] >= threshold, scores))
            faulty_cdps = list(map(lambda item: item[0], scores_))

            num_corrects, num_total_samples_fhr = self.calculate_FHR(faulty_cdps)
            fhr_rate = num_corrects / num_total_samples_fhr if num_total_samples_fhr != 0 else 0
            print(f"FHR: {num_corrects} {num_total_samples_fhr} {fhr_rate}")
            
            num_failures, num_total_samples_hfr = self.calculate_HFR(faulty_cdps)
            hfr_rate = num_failures / num_total_samples_hfr if num_total_samples_hfr != 0 else 0
            print(f"HFR: {num_failures} {num_total_samples_hfr} {hfr_rate}")

            with open(os.path.join("results", f"{self.deepcp.model_name}_{SFL_strategy}_verif.txt"), "a") as file:
                file.write(f"THRESHOLD: {threshold}\n")
                file.write(f"FHR: {num_corrects} {num_total_samples_fhr} {fhr_rate}\n")
                file.write(f"HFR: {num_failures} {num_total_samples_hfr} {hfr_rate}\n")

    def run(self):
        # self.verify("tarantula")
        self.verify("ochiai")
        # self.verify("barinel")


def verify_main1():
    ALPHA = 0.99
    lrp_src.lrp_layers.top_k_percent = ALPHA

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
    ALPHA = 0.99
    lrp_src.lrp_layers.top_k_percent = ALPHA

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('models/data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=1, shuffle=True)

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
        alpha=ALPHA
    )

    time_budget = 3 # seconds
    decision_birch_pickle_path = 'pickles/Model_2_decision_birch.pickle'

    data_sample_preprocess_fn = lambda item: item.view(-1, 784)

    verification = Verification(deepcp2, test_loader, time_budget, data_sample_preprocess_fn, decision_birch_pickle_path)
    verification.run()

def verify_main3():
    ALPHA = 0.99

    # lrp_src.lrp_layers.top_k_percent = 0.7

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

    time_budget = 1 # seconds
    decision_birch_pickle_path = 'pickles/Model_3_decision_birch.pickle'

    data_sample_preprocess_fn = lambda item: item

    verification = Verification(deepcp3, test_loader, time_budget, data_sample_preprocess_fn, decision_birch_pickle_path)
    verification.run()

if __name__ == "__main__":
    # verify_main1()
    # verify_main2()
    verify_main3()