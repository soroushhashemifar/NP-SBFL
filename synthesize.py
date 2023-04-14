import os
import pickle

import numpy as np
import torch
import tqdm

from utils import test


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

    def get_suspicious_neurons(self, SFL_strategy, suspiciousness_threshold):
        with open(os.path.join(self.pickles_path, f"{self.model_name}_{SFL_strategy}_objects.pickle"), 'rb') as handle:
            self.objects_dict = pickle.load(handle)

        layerwise_suspicousness_scores = self.objects_dict["layerwise_suspicousness_scores"]
        suspicousness_neurons_per_layer = []
        for layer_scores in layerwise_suspicousness_scores:
            # layer_scores_ = list(filter(lambda item: not np.isnan(item[1]) and item[1] > suspiciousness_threshold, layer_scores))
            layer_scores_ = list(filter(lambda item: not np.isnan(item[1]), layer_scores))
            layer_scores_ = sorted(layer_scores_, key=lambda item: item[1])
            layer_scores_ = layer_scores_[:suspiciousness_threshold]
            # print(len(layer_scores_))
            
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

            features = [feature.flatten(1) for feature in features]
            # print([a.shape for a in features])

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

    def run(self, SFL_strategy, suspiciousness_threshold):
        print(f"Synthesizing {SFL_strategy}")
        suspicousness_neurons_per_layer = self.get_suspicious_neurons(SFL_strategy, suspiciousness_threshold)
        # print(suspicousness_neurons_per_layer)
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


def evaluation(model_name, SFL_strategy, model, test_loader, parameters):
    print(f"Evaluation synthesized dataset for {SFL_strategy}")

    with open(f"./pickles/synthesized_dataset_{model_name}_{SFL_strategy}.pickle", 'rb') as handle:
        synthesized_dataset = pickle.load(handle)

    synth_dataset = SynthesizedDataset(synthesized_dataset)
    synth_loader = torch.utils.data.DataLoader(
        synth_dataset,
        batch_size=128, shuffle=False)

    print("Evaluation on synthesized dataset:")
    test(model, synth_loader, parameters, scheduler=None)
