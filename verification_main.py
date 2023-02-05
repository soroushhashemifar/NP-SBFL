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

        self.available_transformations = [
            transforms.RandomRotation(120),
            transforms.RandomCrop(self.deepcp.input_size[1:]),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomHorizontalFlip(0.5),
            transforms.Lambda(lambda x : x + torch.randn_like(x)),
            transforms.ColorJitter(brightness=0, contrast=0.4, saturation=0, hue=0),
        ]

        with open(decision_birch_pickle_path, 'rb') as handle:
            self.decision_birch = pickle.load(handle)

    def is_transformation_FHR_possible(self, faulty_cdps, data, target):
        predicted_class_ = -1
        predicted_cluster_ = -1
        start_time = time.time()
        OUT_OF_TIME = False
        while (predicted_class_, predicted_cluster_) in faulty_cdps or predicted_class_ != target:
            if time.time() - start_time >= self.time_budget:
                OUT_OF_TIME = True
                break

            transformator = transforms.Compose(random.sample(self.available_transformations, 3))
            data_ = transformator(data)

            cdp_representation_, critical_neurons_layers_test_, predicted_class_, _ = self.deepcp.generate_cdp_representation(self.data_sample_preprocess_fn(data_))
            if (cdp_representation_ is None and critical_neurons_layers_test_ is None and predicted_class_ is None) or predicted_class_ not in self.decision_birch.keys():
                continue

            predicted_cluster_ = self.decision_birch[predicted_class_].predict(cdp_representation_[None, ...])[0]
        
        return not OUT_OF_TIME

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
            if (predicted_class, predicted_cluster) in faulty_cdps and target != predicted_class:
                num_total_samples += 1                
                if self.is_transformation_FHR_possible(faulty_cdps, data, target):
                    num_corrects += 1

        return num_corrects, num_total_samples

    def is_transformation_HFR_possible(self, faulty_cdps, data, target):
        predicted_class_ = -1
        predicted_cluster_ = -1
        start_time = time.time()
        OUT_OF_TIME = False
        while (predicted_class_, predicted_cluster_) not in faulty_cdps or predicted_class_ == target:
            if time.time() - start_time >= self.time_budget:
                OUT_OF_TIME = True
                break

            transformator = transforms.Compose(random.sample(self.available_transformations, 3))
            data_ = transformator(data)

            cdp_representation_, critical_neurons_layers_test_, predicted_class_, _ = self.deepcp.generate_cdp_representation(self.data_sample_preprocess_fn(data_))
            if (cdp_representation_ is None and critical_neurons_layers_test_ is None and predicted_class_ is None) or predicted_class_ not in self.decision_birch.keys():
                continue

            predicted_cluster_ = self.decision_birch[predicted_class_].predict(cdp_representation_[None, ...])[0]
        
        return not OUT_OF_TIME

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

            if (predicted_class, predicted_cluster) not in faulty_cdps and target == predicted_class:
                num_total_samples += 1
                if self.is_transformation_HFR_possible(faulty_cdps, data, target):
                    num_failures += 1

        return num_failures, num_total_samples

    def verify(self, SFL_strategy):
        with open(os.path.join("results", f"{self.deepcp.model_name}_{SFL_strategy}.txt")) as file:
            content = file.readlines()
            content = list(map(lambda item: item.strip(), content))
            content = list(map(lambda item: item.split("\t"), content))
            scores = list(map(lambda item: (eval(item[0]), float(item[1]), eval(item[2])), content))

        thresholds = sorted(list(set(map(lambda item: item[1], scores))))
        for threshold in thresholds:
            print(f"THRESHOLD: {threshold}")
            scores = list(filter(lambda item: item[1] >= threshold, scores))
            faulty_cdps = list(map(lambda item: item[0], scores))

            num_corrects, num_total_samples_fhr = self.calculate_FHR(faulty_cdps)
            rate = num_corrects / num_total_samples_fhr if num_total_samples_fhr != 0 else 0
            print(f"FHR: {num_corrects} {num_total_samples_fhr} {rate}")
            
            num_failures, num_total_samples_hfr = self.calculate_HFR(faulty_cdps)
            rate = num_failures / num_total_samples_hfr if num_total_samples_hfr != 0 else 0
            print(f"HFR: {num_failures} {num_total_samples_hfr} {rate}")

            with open(os.path.join("results", f"{self.deepcp.model_name}_{SFL_strategy}_verif_ajshgashgowog.txt"), "a") as file:
                file.write(f"THRESHOLD: {threshold}\n")
                file.write(f"FHR: {num_corrects} {num_total_samples_fhr} {rate}\n")
                file.write(f"HFR: {num_failures} {num_total_samples_hfr} {rate}\n")

    def run(self):
        self.verify("tarantula")
        self.verify("ochiai")
        self.verify("barinel")


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
    model.load_state_dict(torch.load("models/mymodel_1.pth", map_location="cpu"))
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
    model.load_state_dict(torch.load("models/mymodel_2.pth", map_location="cpu"))
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
    lrp_src.lrp_layers.top_k_percent = ALPHA

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('models/data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])),
        batch_size=1, shuffle=True)

    model = Net3()
    model.load_state_dict(torch.load("models/mymodel_3.pth", map_location="cpu"))
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

    time_budget = 3 # seconds
    decision_birch_pickle_path = 'pickles/Model_3_decision_birch.pickle'

    data_sample_preprocess_fn = lambda item: item

    verification = Verification(deepcp3, test_loader, time_budget, data_sample_preprocess_fn, decision_birch_pickle_path)
    verification.run()

if __name__ == "__main__":
    verify_main1()
    verify_main2()
    verify_main3()