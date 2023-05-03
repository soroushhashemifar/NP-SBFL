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

    def get_suspicious_neurons(self, SFL_strategy, suspiciousness_threshold):
        with open(os.path.join(self.pickles_path, f"{self.model_name}_{SFL_strategy}_objects.pickle"), 'rb') as handle:
            self.objects_dict = pickle.load(handle)

        layerwise_suspicousness_scores = self.objects_dict["layerwise_suspicousness_scores"]
        suspicousness_neurons_per_layer = []
        for layer_scores in layerwise_suspicousness_scores:
            layer_scores_ = list(filter(lambda item: not np.isnan(item[1]), layer_scores))
            layer_scores_ = sorted(layer_scores_, key=lambda item: item[1])
            layer_scores_ = layer_scores_[:suspiciousness_threshold]
            
            if len(layer_scores_) == 0:
                layer_scores_ = [max(layer_scores, key=lambda item: not np.isnan(item[1]) and item[1])]

            suspicousness_neurons_per_layer.append(layer_scores_)

        return suspicousness_neurons_per_layer

    def run(self, SFL_strategy, suspiciousness_threshold):
        print(f"Synthesizing {SFL_strategy}")
        suspicousness_neurons_per_layer = self.get_suspicious_neurons(SFL_strategy, suspiciousness_threshold)
        # print(suspicousness_neurons_per_layer)
        synthesized_dataset = self.synthesize_testset(suspicousness_neurons_per_layer)
        with open(f"./pickles/synthesized_dataset_{self.model_name}_{SFL_strategy}_k{suspiciousness_threshold}.pickle", 'wb') as handle:
            pickle.dump(synthesized_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)


class SynthesizeV1(Synthesize):

    """
    DeepFault synthesizer
    """

    def __init__(self, model_name, model, test_loader, pickles_path, step_size=5, distance=0.1):
        self.step_size = step_size
        self.distance = distance
        self.model_name = model_name

        self.model = model
        self.test_loader = test_loader
        self.pickles_path = pickles_path

    def __synthsize_image(self, data, perturbed_data, gradients):
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

    def __applyDomainConstraints(self, perturbed_data):
        perturbed_data[perturbed_data > 1] = 1
        perturbed_data[perturbed_data < 0] = 0

        return perturbed_data

    def synthesize_testset(self, suspicousness_neurons_per_layer):
        synthesized_dataset = []
        for data, target in tqdm.tqdm(self.test_loader):
            data_ = data.clone()
            target_ = target.clone()
            for iteration in range(3):
                inputs = torch.autograd.Variable(data_, requires_grad=True)
                outputs, features = self.model(inputs, return_logits=True)
                outputs = torch.softmax(outputs, 1)
                outputs = torch.argmax(outputs, 1)

                features = [feature.flatten(1) for feature in features]

                if iteration == 0:
                    mask = outputs == target_
                    data_original = data_[mask]
                else:
                    mask = torch.ones_like(target_, dtype=torch.bool)

                gradients = []
                for layer_idx in range(len(suspicousness_neurons_per_layer)):
                    for sn_index, _ in suspicousness_neurons_per_layer[layer_idx]:
                        features_ = features[layer_idx][:, [sn_index]]
                        gradients_ = torch.autograd.grad(outputs=features_, inputs=inputs, grad_outputs=torch.ones(features_.size()).to("cpu"), retain_graph=True)[0]
                        gradients_ = gradients_[mask]
                        gradients.append(gradients_)

                data_ = data_[mask]
                perturbed_data = data_.clone()
                perturbed_data = self.__synthsize_image(data_, perturbed_data, gradients)
                perturbed_data = self.__applyDomainConstraints(perturbed_data)

                target_ = target_[mask]
                data_ = perturbed_data

            for datam, perturbed_datam, label in zip(data_original, perturbed_data, target_):
                synthesized_dataset.append((datam.permute(1, 2, 0).numpy(), perturbed_datam.permute(1, 2, 0).numpy(), label.numpy()))

        return synthesized_dataset


class SynthesizeV2(Synthesize):

    def __init__(self, model_name, model, test_loader, pickles_path, num_iterations=10, learning_rate=0.01):
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.model_name = model_name

        self.model = model
        self.test_loader = test_loader
        self.pickles_path = pickles_path

        assert test_loader.batch_size == 1, f"This synthesis procedure only works for batch size = 1 (current batch size = {test_loader.batch_size})"

    def __loss_function(self, activations, layer_index, target_neurons, original_activations):
        output = activations[layer_index].reshape(1, -1)
        target_activation = output[:, target_neurons[layer_index]].sum(1)
        loss = -target_activation

        # # Keep activation of other neurons fixed
        # remaining_neurons = list(set(range(output.shape[1])) - set(target_neurons[layer_index]))
        # original_output = original_activations[layer_index].reshape(1, -1)
        # loss += torch.square(original_output[:, remaining_neurons] - output[:, remaining_neurons]).sum()
        
        prev_loss = torch.zeros_like(loss)
        for l_index in range(layer_index):
            output = activations[l_index].reshape(activations[l_index].shape[0], -1)
            prev_target_activation = output[:, target_neurons[l_index]].sum(1)
            prev_loss += -prev_target_activation + torch.abs(target_activation - prev_target_activation) 

        loss += prev_loss

        return loss

    def __generate_image(self, image, target_neurons, num_iterations=100, learning_rate=0.01):
        image.requires_grad = True

        with torch.no_grad():
            _, original_activations = self.model(image, return_logits=True)

        for _ in range(num_iterations):
            # Set up the optimizer
            optimizer = torch.optim.Adam([image], lr=learning_rate)

            # Iterate over the target neurons and optimize the image for each one
            for i, target_neuron in enumerate(target_neurons):
                # Zero out gradients
                optimizer.zero_grad()

                # Compute the loss as the negative activation of the target neuron
                _, activations = self.model(image, return_logits=True)
                loss = self.__loss_function(activations, i, target_neurons, original_activations)

                # Compute the gradient of the loss with respect to the image
                for j in range(loss.shape[0]):
                    loss[j].backward(retain_graph=True)

                # Update the image using the gradient ascent algorithm
                optimizer.step()

                # Clamp the pixel values to be between 0 and 1
                image.data.clamp_(0.0, 1.0)

        return image.detach()

    def synthesize_testset(self, suspicousness_neurons_per_layer):
        synthesized_dataset = []
        for data, target in tqdm.tqdm(self.test_loader):
            data_ = data.clone()
            target_ = target.clone()

            outputs = self.model(data_)
            outputs = torch.softmax(outputs, 1)
            outputs = torch.argmax(outputs, 1)
            mask = outputs == target_

            data_ = data_[mask]
            target_ = target_[mask]

            if data_.shape[0] == 0: 
                continue

            target_neurons = list(map(lambda item: list(map(lambda item_: item_[0], item)), suspicousness_neurons_per_layer))
            perturbed_data = self.__generate_image(data_.clone(), target_neurons, num_iterations=self.num_iterations, learning_rate=self.learning_rate)

            for datam, perturbed_datam, label in zip(data_, perturbed_data, target_):
                synthesized_dataset.append((datam.permute(1, 2, 0).detach().numpy(), perturbed_datam.permute(1, 2, 0).numpy(), label.numpy()))

        return synthesized_dataset


class SynthesizedDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, transforms=None):
        self.dataset = dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, perturbed_data, label = self.dataset[idx]
        if perturbed_data.shape[2] == 3:
            perturbed_data = torch.tensor(perturbed_data).permute(2, 0, 1)
        else:
            perturbed_data = torch.tensor(perturbed_data)

        return perturbed_data, label


def evaluation(model_name, SFL_strategy, model, test_loader, parameters, suspiciousness_threshold):
    print(f"Evaluation synthesized dataset for {SFL_strategy}")

    with open(f"./pickles/synthesized_dataset_{model_name}_{SFL_strategy}_k{suspiciousness_threshold}.pickle", 'rb') as handle:
        synthesized_dataset = pickle.load(handle)

    synth_dataset = SynthesizedDataset(synthesized_dataset)
    synth_loader = torch.utils.data.DataLoader(
        synth_dataset,
        batch_size=128, shuffle=False)

    print("Evaluation on synthesized dataset:")
    test(model, synth_loader, parameters, scheduler=None)
