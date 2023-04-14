import os
import pickle

import numpy as np
import torch
import tqdm
from torch.autograd import Variable


class Verification:

    def __init__(self, deepcp, pickles_path):
        self.deepcp = deepcp
        self.pickles_path = pickles_path

    # def get_faulty_paths_vector(self, SFL_strategy, suspiciousness_threshold):
    #     with open(os.path.join(self.pickles_path, f"{self.deepcp.model_name}_{SFL_strategy}_objects.pickle"), 'rb') as handle:
    #         self.objects_dict = pickle.load(handle)

    #     layerwise_suspicousness_scores = self.objects_dict["layerwise_suspicousness_scores"]
    #     suspicousness_scores_vector = np.zeros(sum(list(map(lambda item: len(item), layerwise_suspicousness_scores))))
    #     indices = np.cumsum([0] + self.deepcp.layer_shapes)
    #     for i, layer_scores in zip(range(indices.shape[0]-1), layerwise_suspicousness_scores):
    #         print(layer_scores)
    #         layer_scores_ = list(filter(lambda item: not np.isnan(item[1]) and item[1] > suspiciousness_threshold, layer_scores))
    #         if len(layer_scores_) == 0:
    #             layer_scores_ = [max(layer_scores, key=lambda item: not np.isnan(item[1]) and item[1])]

    #         for neuron_index, _ in layer_scores_:
    #             suspicousness_scores_vector[indices[i]:indices[i+1]][neuron_index] = 1

    #     return suspicousness_scores_vector

    def get_faulty_paths_vector(self, SFL_strategy, suspiciousness_threshold):
        with open(os.path.join(self.pickles_path, f"{self.deepcp.model_name}_{SFL_strategy}_objects.pickle"), 'rb') as handle:
            self.objects_dict = pickle.load(handle)

        layerwise_suspicousness_scores = self.objects_dict["layerwise_suspicousness_scores"]
        suspicousness_scores_vectors = [np.zeros(num_neurons) for num_neurons in self.deepcp.layer_shapes]
        for i, layer_scores in zip(range(len(self.deepcp.layer_shapes)), layerwise_suspicousness_scores):
            # layer_scores_ = list(filter(lambda item: not np.isnan(item[1]) and item[1] > suspiciousness_threshold, layer_scores))
            layer_scores_ = list(filter(lambda item: not np.isnan(item[1]), layer_scores))
            layer_scores_ = sorted(layer_scores_, key=lambda item: item[1])
            layer_scores_ = layer_scores_[:suspiciousness_threshold]
            # print(layer_scores_)

            if len(layer_scores_) == 0:
                layer_scores_ = [max(layer_scores, key=lambda item: not np.isnan(item[1]) and item[1])]
            
            for neuron_index, _ in layer_scores_:
                suspicousness_scores_vectors[i][neuron_index] = 1

        return suspicousness_scores_vectors

    def calculate_tests_ratio(self, faulty_paths_vectors):
        num_failed_tests = 0
        num_total_tests_failed = 0
        num_passed_tests = 0
        num_total_tests_passed = 0
        num_activating_faulty_neurons = 0
        for data, target in tqdm.tqdm(self.test_loader):
            if self.deepcp.cuda:
                data, target = data.cuda(), target.cuda()

            data, target = Variable(data), Variable(target)

            cdp_representation, critical_neurons_layers_test, predicted_class, activation_mask = self.deepcp.generate_cdp_representation(data)
            if cdp_representation is None and critical_neurons_layers_test is None and predicted_class is None:
                continue

            layer_flags = []
            indices = np.cumsum([0] + self.deepcp.layer_shapes)
            for i, faulty_paths_vector in zip(range(indices.shape[0]-1), faulty_paths_vectors):
                mask = activation_mask[indices[i]:indices[i+1]][faulty_paths_vector.astype(bool)]
                layer_flags.append(any(mask))

            if all(layer_flags):
                num_activating_faulty_neurons += 1

            if target != predicted_class:
                num_total_tests_failed += 1

                if all(layer_flags):
                    num_failed_tests += 1
            else:
                num_total_tests_passed += 1

                if not all(layer_flags):
                    num_passed_tests += 1

        return num_failed_tests, num_total_tests_failed, num_passed_tests, num_total_tests_passed, num_activating_faulty_neurons

    def verify(self, SFL_strategy, suspiciousness_threshold):
        with open(f"./pickles/synthesized_dataset_{self.deepcp.model_name}_{SFL_strategy}.pickle", 'rb') as handle:
            synthesized_dataset = pickle.load(handle)

        synth_dataset = SynthesizedDataset(synthesized_dataset)
        self.test_loader = torch.utils.data.DataLoader(
            synth_dataset,
            batch_size=1, shuffle=False)

        faulty_paths_vectors = self.get_faulty_paths_vector(SFL_strategy, suspiciousness_threshold)
        num_failed_tests, num_total_tests_failed, num_passed_tests, num_total_tests_passed, num_activating_faulty_neurons = self.calculate_tests_ratio(faulty_paths_vectors)

        print("fails activating faulty paths:", num_failed_tests, num_total_tests_failed, num_failed_tests / num_total_tests_failed)
        # print("passes not activating faulty paths", num_passed_tests, num_total_tests_passed, num_passed_tests / num_total_tests_passed)
        print("synthesized samples activating faulty neurons:", num_activating_faulty_neurons, num_total_tests_failed+num_total_tests_passed, num_activating_faulty_neurons / (num_total_tests_failed+num_total_tests_passed))

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
