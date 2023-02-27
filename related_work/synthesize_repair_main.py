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


class Synthesize:

    def __init__(self, deepcp, model, test_loader, metric_thresholds, pixel_mean, pixel_std, step_size=5, distance=0.1):
        self.step_size = step_size
        self.distance = distance

        self.deepcp = deepcp
        self.model = model
        self.test_loader = test_loader
        self.metric_thresholds = metric_thresholds

        self.pixel_mean = torch.tensor(list(pixel_mean)).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        self.pixel_std = torch.tensor(list(pixel_std)).unsqueeze(0).unsqueeze(2).unsqueeze(3)

        with open(f'./pickles/{deepcp.model_name}_decision_graph.pickle', 'rb') as handle:
            self.decision_graph = pickle.load(handle)

        with open(f"./pickles/{deepcp.model_name}_decision_birch.pickle", 'rb') as handle:
                self.decision_birch = pickle.load(handle)

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
        perturbed_data = (perturbed_data * self.pixel_std) + self.pixel_mean
        perturbed_data[perturbed_data > 1] = 1
        perturbed_data[perturbed_data < 0] = 0
        perturbed_data = (perturbed_data - self.pixel_mean) / self.pixel_std

        return perturbed_data

    def get_suspicious_neurons(self, SFL_strategy, metric_threshold):
        with open(os.path.join("results", f"{self.deepcp.model_name}_{SFL_strategy}.txt")) as file:
            content = file.readlines()
            content = list(map(lambda item: item.strip(), content))
            content = list(map(lambda item: item.split("\t"), content))
            scores = list(map(lambda item: (eval(item[0]), float(item[1]), eval(item[2])), content))

        faulty_scores = list(filter(lambda item: item[1] >= metric_threshold, scores))
        faulty_cdps = list(map(lambda item: item[0], faulty_scores))
        fpl_detected_neurons = [(layer_index, neuron_index) for faulty_cdp in faulty_cdps for layer_index in range(len(self.decision_graph[faulty_cdp])-1) for neuron_index in self.decision_graph[faulty_cdp][layer_index]]
        fpl_detected_neurons = set(fpl_detected_neurons)

        return fpl_detected_neurons

    def synthesize_testset(self, fpl_detected_neurons):
        features = []
        def get_features():
            def hook(model, input, output):
                features.append(output)

            return hook

        for layer in self.model.modules():
            if layer.__class__.__name__ in ["Conv2d", "MaxPool2d", "Linear"]:
                layer.register_forward_hook(get_features())

        # data_sample_preprocess_fn = lambda item: item

        synthesized_dataset = []
        for data, labels in tqdm.tqdm(self.test_loader):
            features = []

            inputs = torch.autograd.Variable(data, requires_grad=True)
            logits = self.model(inputs)
            logits = torch.softmax(logits, 1)
            outputs = torch.argmax(logits, 1)

            # predicted_clusters = []
            # for sample_index in range(data.shape[0]):
            #     class_specific_birch = decision_birch[outputs[sample_index].item()]
            #     cdp_representation, _, _, _ = deepcp3.generate_cdp_representation(data_sample_preprocess_fn(data[sample_index][None, ...]))
            #     predicted_cluster = class_specific_birch.predict(cdp_representation[None, ...])[0]
            #     predicted_clusters.append(predicted_cluster)

            # predicted_clusters = torch.tensor(predicted_clusters)
            # cdp_tuples = torch.concat([outputs[..., None], predicted_clusters[..., None]], dim=1)
            # cdp_mask = torch.tensor([tuple(cdp_tuple.numpy().tolist()) in faulty_cdps for cdp_tuple in cdp_tuples])
            # mask = torch.logical_and(outputs == labels, cdp_mask)

            mask = outputs == labels

            gradients = []
            for layer_idx, sn_index in fpl_detected_neurons:
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
            labels = labels[mask].numpy()
            for datam, perturbed_datam, label in zip(data, perturbed_data, labels):
                synthesized_dataset.append((datam, perturbed_datam, label))

        return synthesized_dataset

    def run(self):
        for SFL_strategy, metric_threshold in self.metric_thresholds:
            print(f"Synthesizing {SFL_strategy}")
            suspicious_neurons = self.get_suspicious_neurons(SFL_strategy, metric_threshold)
            synthesized_dataset = self.synthesize_testset(suspicious_neurons)
            with open(f"./pickles/synthesized_dataset_{self.deepcp.model_name}_{SFL_strategy}.pickle", 'wb') as handle:
                pickle.dump(synthesized_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)


class SynthesizedDataset(torch.utils.data.Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, perturbed_data, label = self.dataset[idx]
        perturbed_data = torch.tensor(perturbed_data).permute(2, 0, 1)

        return perturbed_data, label


def repair_method(deepcp, SFL_strategy, model, test_loader, parameters):
    print(f"Repairing {SFL_strategy}")

    with open(f"./pickles/synthesized_dataset_{deepcp.model_name}_{SFL_strategy}.pickle", 'rb') as handle:
        synthesized_dataset = pickle.load(handle)

    synth_dataset = SynthesizedDataset(synthesized_dataset)
    synth_loader = torch.utils.data.DataLoader(
        synth_dataset,
        batch_size=128, shuffle=True)

    print("Evaluation on synthesized dataset:")
    test(model, synth_loader, parameters, scheduler=None)

    print("Evaluation on testset (before finetune):")
    test(model, test_loader, parameters, scheduler=None)

    optimizer = optim.SGD(model.parameters(), lr=parameters["learning_rate"], momentum=0.9)
    for epoch in range(1, parameters["num_epochs"] + 1):
        train(epoch, model, synth_loader, optimizer, parameters)

    print("Evaluation on testset (after finetune):")
    test(model, test_loader, parameters, scheduler=None)

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
        batch_size=128, shuffle=False)

    metric_thresholds = [("tarantula", 0.99), ("ochiai", 0.26), ("barinel", 0.18)]

    model_1_synthsizer = Synthesize(deepcp1, model, test_loader, metric_thresholds, (0.1307,), (0.3081,), step_size=50, distance=0.3)
    model_1_synthsizer.run()

    parameters = {
            "cuda": False,
            "loss": "nll_loss",
            "log_interval": 1000,
            "learning_rate": 0.1,
            "num_epochs": 15,
        }

    for SFL_strategy, _ in metric_thresholds:
        model = Net1()
        model.load_state_dict(torch.load("models/Model_1.pth", map_location="cpu"))
        model = model.to("cpu")
        repaired_model = repair_method(deepcp1, SFL_strategy, model, test_loader, parameters)
        torch.save(repaired_model.state_dict(), f"models/{deepcp1.model_name}_repaired_{SFL_strategy}.pth")

def repair_model_2():
    print("Repair model 2 started")

    ALPHA = 0.99
    lrp_src.lrp_layers.top_k_percent = ALPHA

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

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('models/data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=128, shuffle=False)

    metric_thresholds = [("tarantula", 0.87), ("ochiai", 0.05), ("barinel", 0.014)]

    model_2_synthsizer = Synthesize(deepcp2, model, test_loader, metric_thresholds, (0.1307,), (0.3081,), step_size=20, distance=0.1)
    model_2_synthsizer.run()

    parameters = {
            "cuda": False,
            "loss": "nll_loss",
            "log_interval": 1000,
            "learning_rate": 0.1,
            "num_epochs": 15,
        }

    for SFL_strategy, _ in metric_thresholds:
        model = Net2()
        model.load_state_dict(torch.load("models/Model_2.pth", map_location="cpu"))
        model = model.to("cpu")
        repaired_model = repair_method(deepcp2, SFL_strategy, model, test_loader, parameters)
        torch.save(repaired_model.state_dict(), f"models/{deepcp2.model_name}_repaired_{SFL_strategy}.pth")

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

    metric_thresholds = [("tarantula", 0.91), ("ochiai", 0.38), ("barinel", 0.38)]

    model_3_synthsizer = Synthesize(deepcp3, model, test_loader, metric_thresholds, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), step_size=1, distance=0.3)
    model_3_synthsizer.run()

    parameters = {
            "cuda": False,
            "loss": "cross_entropy",
            "log_interval": 1000,
            "learning_rate": 0.1,
            "num_epochs": 15,
        }

    for SFL_strategy, _ in metric_thresholds:
        model = Net3()
        model.load_state_dict(torch.load("models/Model_3.pth", map_location="cpu"))
        model = model.to("cpu")
        repaired_model = repair_method(deepcp3, SFL_strategy, model, test_loader, parameters)
        torch.save(repaired_model.state_dict(), f"models/{deepcp3.model_name}_repaired_{SFL_strategy}.pth")

if __name__ == "__main__":
    # repair_model_1()
    # repair_model_2()
    repair_model_3()