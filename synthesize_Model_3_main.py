import pickle
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from models.train_model_3 import Net
import tqdm
import numpy as np
from utils import get_BARINEL_score, get_ochiai_score, get_tarantula_score, test, train
import torch.optim as optim


def synthesize_testset():
    with open('./pickles/Model_3_decision_graph.pickle', 'rb') as handle1, open('./pickles/Model_3_cdp_spectrums.pickle', 'rb') as handle2:
        decision_graph = pickle.load(handle1)
        cdp_spectrums = pickle.load(handle2)

    for metric, metric_threshold in [("tarantula", 0.91), ("ochiai", 0.37), ("barinel", 0.38)]:
        scores = []
        for key, path_spectrum in cdp_spectrums.items():
            if metric == "tarantula":
                score = get_tarantula_score(path_spectrum)
            elif metric == "ochiai":
                score = get_ochiai_score(path_spectrum)
            elif metric == "barinel":
                score = get_BARINEL_score(path_spectrum)

            scores.append((key, score))

        scores = list(filter(lambda item: item[1] > metric_threshold, scores))
        faulty_cdps = list(map(lambda item: item[0], scores))

        fpl_detected_neurons = [(layer_index, neuron_index) for faulty_cdp in faulty_cdps for layer_index in range(6) for neuron_index in decision_graph[faulty_cdp][layer_index]]
        fpl_detected_neurons = set(fpl_detected_neurons)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('models/data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])),
        batch_size=1, shuffle=False)

    model = Net()
    model.load_state_dict(torch.load("models/mymodel_3.pth", map_location="cpu"))
    model = model.to("cpu")

    features = []
    def get_features():
        def hook(model, input, output):
            features.append(output)

        return hook

    model.conv1_1.register_forward_hook(get_features())
    model.conv2_1.register_forward_hook(get_features())
    model.conv3_1.register_forward_hook(get_features())
    model.conv4_1.register_forward_hook(get_features())
    model.fc1.register_forward_hook(get_features())
    model.fc2.register_forward_hook(get_features())
    model.fc3.register_forward_hook(get_features())

    step_size = 1
    distance = 0.1

    synthesized_dataset = []
    for data, label in tqdm.tqdm(test_loader):
        features = []

        inputs = torch.autograd.Variable(data, requires_grad=True)
        logits = model(inputs)
        logits = torch.softmax(logits, 1)
        outputs = torch.argmax(logits, 1)

        if outputs.item() != label.item():
            continue

        gradients = []
        for layer_idx, sn_index in fpl_detected_neurons:
            features_ = features[layer_idx][:, [sn_index]]
            gradients_ = torch.autograd.grad(outputs=features_, inputs=inputs, grad_outputs=torch.ones(features_.size()).to("cpu"), retain_graph=True)[0]
            gradients.append(gradients_)

        perturbed_data = data.clone()
        for i in range(data.shape[2]):
            for j in range(data.shape[3]):
                sum_grad = torch.tensor([0., 0., 0.])
                for k in range(len(gradients)):
                    sum_grad += gradients[k][0, :, i, j]

                avg_grad = sum_grad / len(gradients)
                avg_grad = avg_grad * step_size

                # Clipping gradients.
                avg_grad[avg_grad > distance] = distance
                avg_grad[avg_grad < -distance] = -distance

                perturbed_data[0, :, i, j] = data[0, :, i, j] + avg_grad
                # perturbed_data[perturbed_data < 0.] = 0.
                # perturbed_data[perturbed_data > 1.] = 1.

        data = data[0].permute(1, 2, 0).numpy()
        perturbed_data = perturbed_data[0].permute(1, 2, 0).numpy()
        label = label[0].numpy()
        synthesized_dataset.append((data, perturbed_data, label))

    with open("./pickles/synthesized_dataset_Model_3.pickle", 'wb') as handle:
        pickle.dump(synthesized_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)


class SynthesizedDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, transform=None):
        self.dataset = dataset

        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, perturbed_data, label = self.dataset[idx]
        label = label[0]

        perturbed_data = torch.tensor(perturbed_data).permute(2, 0, 1)

        # print(perturbed_data.min(), perturbed_data.max())

        # if self.transform:
        #     perturbed_data = self.transform(perturbed_data)

        return perturbed_data, label


def repair_model():
    with open("./pickles/synthesized_dataset_Model_3.pickle", 'rb') as handle:
        synthesized_dataset = pickle.load(handle)

    transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])

    synth_dataset = SynthesizedDataset(synthesized_dataset, transform=transform)
    synth_loader = torch.utils.data.DataLoader(
        synth_dataset,
        batch_size=128, shuffle=True)
    parameters = {
            "cuda": False,
            "loss": "cross_entropy",
            "log_interval": 1000
        }

    model = Net()
    model.load_state_dict(torch.load("models/mymodel_3.pth", map_location="cpu"))
    model = model.to("cpu")

    print("Evaluation on synthesized dataset:")
    test(model, synth_loader, parameters, scheduler=None)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('models/data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])),
        batch_size=128, shuffle=False)

    print("Evaluation on testset (before finetune):")
    test(model, test_loader, parameters, scheduler=None)

    learning_rate = 0.01
    num_epochs = 10

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    for epoch in range(1, num_epochs + 1):
        train(epoch, model, synth_loader, optimizer, parameters)

    print("Evaluation on testset (after finetune):")
    test(model, test_loader, parameters, scheduler=None)

synthesize_testset()
repair_model()