import enum
import os
import pickle
from unicodedata import name
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
from torch.nn.utils import prune


def get_suspicious_neurons(deepcp, decision_graph, SFL_strategy, metric_threshold):
    with open(os.path.join("results", f"{deepcp.model_name}_{SFL_strategy}.txt")) as file:
        content = file.readlines()
        content = list(map(lambda item: item.strip(), content))
        content = list(map(lambda item: item.split("\t"), content))
        scores = list(map(lambda item: (eval(item[0]), float(item[1]), eval(item[2])), content))

    faulty_scores = list(filter(lambda item: item[1] >= metric_threshold, scores))
    faulty_cdps = list(map(lambda item: item[0], faulty_scores))
    fpl_detected_neurons = [(layer_index, neuron_index) for faulty_cdp in faulty_cdps for layer_index in range(len(decision_graph[faulty_cdp])-1) for neuron_index in decision_graph[faulty_cdp][layer_index]]
    fpl_detected_neurons = set(fpl_detected_neurons)

    return fpl_detected_neurons

def detected_neurons_to_mask(detected_neurons_prev, detected_neurons, weight_shape):
    mask = torch.zeros(weight_shape)
    for _, neuron_index in detected_neurons:
        for _, prev_neuron_index in detected_neurons_prev:
            mask[neuron_index, prev_neuron_index] = 1

    return mask

def prune_layer(layer, detected_neurons_prev, detected_neurons):
    weight_shape = layer.weight.shape
    mask = detected_neurons_to_mask(detected_neurons_prev, detected_neurons, weight_shape)
    layer = prune.custom_from_mask(layer, name="weight", mask=mask)
    layer = prune.remove(layer, "weight")

    return layer

def repair_method(layers_structure, model, train_loader, test_loader, parameters, deepcp, decision_graph, SFL_strategy, metric_threshold):
    layers_to_prune = list(filter(lambda layer: layer.__class__.__name__ in ["Conv2d", "MaxPool2d", "Linear"], layers_structure))
    layers_to_prune = layers_to_prune[:-1]

    detected_neurons = get_suspicious_neurons(deepcp, decision_graph, SFL_strategy, metric_threshold)
    latest_trainable_layer_index = -1
    for layer_index, layer in enumerate(layers_to_prune): 
        if layer_index > 0 and layer.__class__.__name__ != "MaxPool2d":
            detected_neurons_prev = list(filter(lambda item: item[0] == latest_trainable_layer_index, detected_neurons))
            detected_neurons_ = list(filter(lambda item: item[0] == layer_index, detected_neurons))
            layer = prune_layer(layer, detected_neurons_prev, detected_neurons_)

        if layer.__class__.__name__ in ["Conv2d", "Linear"]:
            latest_trainable_layer_index = layer_index

    optimizer = optim.SGD(model.parameters(), lr=parameters["learning_rate"], momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)
    for epoch in range(1, parameters["num_epochs"] + 1):
        train(epoch, model, train_loader, optimizer, parameters)
        test(model, test_loader, parameters, scheduler=scheduler, log=True)

    return model

def repair_model_1():
    print("Repair model 1 started")

    ALPHA = 0.99
    lrp_src.lrp_layers.top_k_percent = ALPHA

    metric_thresholds = [("tarantula", 0.99), ("ochiai", 0.26), ("barinel", 0.18)]

    for SFL_strategy, metric_threshold in metric_thresholds:
        print(SFL_strategy)

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

        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('models/data', train=True, download=True, transform=transforms.Compose([
                            # transforms.RandomHorizontalFlip(),
                            # transforms.RandomVerticalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=512, shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('models/data', train=False, download=True, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=512, shuffle=False)

        with open(f'./pickles/{deepcp1.model_name}_decision_graph.pickle', 'rb') as handle:
            decision_graph = pickle.load(handle)

        parameters = {
                "cuda": False,
                "loss": "nll_loss",
                "log_interval": 1000,
                "learning_rate": 0.1,
                "num_epochs": 10,
            }

        print("Evaluation on testset (before finetune):")
        test(model, test_loader, parameters, scheduler=None)

        repaired_model = repair_method(layers_structure, model, train_loader, test_loader, parameters, deepcp1, decision_graph, SFL_strategy, metric_threshold)

        print("Evaluation on testset (after finetune):")
        test(repaired_model, test_loader, parameters, scheduler=None)

        # torch.save(repaired_model.state_dict(), f"models/{deepcp2.model_name}_repaired_{SFL_strategy}.pth")

def repair_model_2():
    print("Repair model 2 started")

    ALPHA = 0.99
    lrp_src.lrp_layers.top_k_percent = ALPHA

    metric_thresholds = [("tarantula", 0.87), ("ochiai", 0.05), ("barinel", 0.014)]

    for SFL_strategy, metric_threshold in metric_thresholds:
        print(SFL_strategy)

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

        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('models/data', train=True, download=True, transform=transforms.Compose([
                            # transforms.RandomHorizontalFlip(),
                            # transforms.RandomVerticalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=512, shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('models/data', train=False, download=True, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=512, shuffle=False)

        with open(f'./pickles/{deepcp2.model_name}_decision_graph.pickle', 'rb') as handle:
            decision_graph = pickle.load(handle)

        parameters = {
                "cuda": False,
                "loss": "nll_loss",
                "log_interval": 1000,
                "learning_rate": 0.1,
                "num_epochs": 10,
            }

        print("Evaluation on testset (before finetune):")
        test(model, test_loader, parameters, scheduler=None)

        repaired_model = repair_method(layers_structure, model, train_loader, test_loader, parameters, deepcp2, decision_graph, SFL_strategy, metric_threshold)

        print("Evaluation on testset (after finetune):")
        test(repaired_model, test_loader, parameters, scheduler=None)

        # torch.save(repaired_model.state_dict(), f"models/{deepcp2.model_name}_repaired_{SFL_strategy}.pth")

def repair_model_3():
    print("Repair model 3 started")

    ALPHA = 0.9
    lrp_src.lrp_layers.top_k_percent = ALPHA

    metric_thresholds = [("tarantula", 0.90802413), ("ochiai", 0.37238748), ("barinel", 0.3964497)]

    for SFL_strategy, metric_threshold in metric_thresholds:
        print(SFL_strategy)

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

        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('models/data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])),
            batch_size=512, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('models/data', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])),
            batch_size=512, shuffle=False)

        with open(f'./pickles/{deepcp3.model_name}_decision_graph.pickle', 'rb') as handle:
            decision_graph = pickle.load(handle)

        parameters = {
                "cuda": False,
                "loss": "cross_entropy",
                "log_interval": 1000,
                "learning_rate": 0.1,
                "num_epochs": 20,
            }

        print("Evaluation on testset (before finetune):")
        test(model, test_loader, parameters, scheduler=None)

        repaired_model = repair_method(layers_structure, model, train_loader, test_loader, parameters, deepcp3, decision_graph, SFL_strategy, metric_threshold)

        print("Evaluation on testset (after finetune):")
        test(repaired_model, test_loader, parameters, scheduler=None)

        # torch.save(repaired_model.state_dict(), f"models/{deepcp2.model_name}_repaired_{SFL_strategy}.pth")

if __name__ == "__main__":
    # repair_model_1()
    # repair_model_2()
    repair_model_3()
