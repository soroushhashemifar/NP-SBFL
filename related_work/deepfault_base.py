import pickle
import sys

from synthesize import SynthesizeV1
sys.path.insert(0, "..")
import os

import numpy as np
import tqdm
from deepcp_base import DeepCP
from torch.autograd import Variable
from verification import SynthesizedsetVerification


class DeepFault(DeepCP):

    def __init__(self, get_relevancy_and_activations_fn=None, **kwargs):
        self.get_relevancy_and_activations = get_relevancy_and_activations_fn

        super().__init__(**kwargs)

    def calculate_hit_spectrums(self):
        neuron_hit_spectrums = {
            "A_P": 0, 
            "I_P": 0, 
            "A_F": 0, 
            "I_F": 0, 
        }
        for data, target in tqdm.tqdm(self.train_loader):
            if self.cuda:
                data, target = data.cpu(), target.cpu()

            data, target = Variable(data), Variable(target)

            cdp_representation, critical_neurons_layers, predicted_class, activation_mask = self.generate_cdp_representation(data)
            if cdp_representation is None and predicted_class is None:
                continue

            if predicted_class == target:
                neuron_hit_spectrums["A_P"] += activation_mask
                neuron_hit_spectrums["I_P"] += (1 - activation_mask)
            else:
                neuron_hit_spectrums["A_F"] += activation_mask
                neuron_hit_spectrums["I_F"] += (1 - activation_mask)

        return neuron_hit_spectrums

    def get_suspiciousness_scores(self, scores_vector):
        reformatted_suspicousness_scores = []
        indices = np.cumsum([0] + self.layer_shapes)
        for i in range(indices.shape[0]-1):
            layer_scores = scores_vector[indices[i]:indices[i+1]]
            layer_scores = list(zip([i]*len(layer_scores), range(len(layer_scores)), layer_scores))
            reformatted_suspicousness_scores.extend(layer_scores)

        return reformatted_suspicousness_scores

    def run(self):
        if not os.path.isdir(self.path_to_save_pickles):
            os.makedirs(self.path_to_save_pickles)

        neuron_hit_spectrums = self.calculate_hit_spectrums()

        scores_vector = self.get_scores_from_spectrums(neuron_hit_spectrums, "tarantula")
        scores = self.get_suspiciousness_scores(scores_vector)
        self.write_objects({
            "neuron_hit_spectrums": neuron_hit_spectrums,
            "scores_vector": scores_vector,
            "suspicousness_scores": scores,
            "layer_shapes": self.layer_shapes,
        }, "tarantula")

        scores_vector = self.get_scores_from_spectrums(neuron_hit_spectrums, "ochiai")
        scores = self.get_suspiciousness_scores(scores_vector)
        self.write_objects({
            "neuron_hit_spectrums": neuron_hit_spectrums,
            "scores_vector": scores_vector,
            "suspicousness_scores": scores,
            "layer_shapes": self.layer_shapes,
        }, "ochiai")

        scores_vector = self.get_scores_from_spectrums(neuron_hit_spectrums, "barinel")
        scores = self.get_suspiciousness_scores(scores_vector)
        self.write_objects({
            "neuron_hit_spectrums": neuron_hit_spectrums,
            "scores_vector": scores_vector,
            "suspicousness_scores": scores,
            "layer_shapes": self.layer_shapes,
        }, "barinel")


class Synthesize_DF(SynthesizeV1):
    
    def get_suspicious_neurons(self, SFL_strategy, suspiciousness_threshold):
        with open(os.path.join(self.pickles_path, f"{self.model_name}_{SFL_strategy}_objects.pickle"), 'rb') as handle:
            self.objects_dict = pickle.load(handle)

        suspicousness_scores = self.objects_dict["suspicousness_scores"]
        layer_shapes = self.objects_dict["layer_shapes"]

        filtered_scores = list(filter(lambda item: not np.isnan(item[2]), suspicousness_scores))

        sorted_scores = sorted(filtered_scores, key=lambda item: item[2], reverse=True)
        suspicousness_neurons = sorted_scores[:suspiciousness_threshold]
        
        layerwise_suspicousness_scores = [[] for _ in range(len(layer_shapes))]
        for suspicious_neuron in suspicousness_neurons:
            layerwise_suspicousness_scores[suspicious_neuron[0]].append(suspicious_neuron[1:])

        suspicousness_neurons_per_layer = []
        for layer_scores in layerwise_suspicousness_scores:
            layer_scores_ = list(filter(lambda item: not np.isnan(item[1]), layer_scores))
            layer_scores_ = sorted(layer_scores_, key=lambda item: item[1], reverse=True)
            layer_scores_ = layer_scores_[:suspiciousness_threshold]
            
            # if len(layer_scores_) == 0:
            #     layer_scores_ = [max(layer_scores, key=lambda item: not np.isnan(item[1]) and item[1])]

            suspicousness_neurons_per_layer.append(layer_scores_)

        return suspicousness_neurons_per_layer


class Verification_DF(SynthesizedsetVerification):
    
    def get_faulty_paths_vector(self, SFL_strategy, suspiciousness_threshold):
        with open(os.path.join(self.pickles_path, f"{self.deepcp.model_name}_{SFL_strategy}_objects.pickle"), 'rb') as handle:
            self.objects_dict = pickle.load(handle)

        suspicousness_scores = self.objects_dict["suspicousness_scores"]
        layer_shapes = self.objects_dict["layer_shapes"]

        filtered_scores = list(filter(lambda item: not np.isnan(item[2]), suspicousness_scores))

        sorted_scores = sorted(filtered_scores, key=lambda item: item[2], reverse=True)
        suspicousness_neurons = sorted_scores[:suspiciousness_threshold]
        
        layerwise_suspicousness_scores = [[] for _ in range(len(layer_shapes))]
        for suspicious_neuron in suspicousness_neurons:
            layerwise_suspicousness_scores[suspicious_neuron[0]].append(suspicious_neuron[1:])

        suspicousness_scores_vectors = [np.zeros(num_neurons) for num_neurons in self.deepcp.layer_shapes]
        for i, layer_scores in zip(range(len(self.deepcp.layer_shapes)), layerwise_suspicousness_scores):
            # layer_scores_ = list(filter(lambda item: not np.isnan(item[1]) and item[1] > suspiciousness_threshold, layer_scores))
            layer_scores_ = list(filter(lambda item: not np.isnan(item[1]), layer_scores))
            layer_scores_ = sorted(layer_scores_, key=lambda item: item[1])
            layer_scores_ = layer_scores_[:suspiciousness_threshold]
            # print(layer_scores_)

            # if len(layer_scores_) == 0:
            #     layer_scores_ = [max(layer_scores, key=lambda item: not np.isnan(item[1]) and item[1])]
            
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

                if mask.shape[0] == 0:
                    layer_flags.append(True)
                else:
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
