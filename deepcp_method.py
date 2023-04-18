import os
import pickle
import random
import time
from tracemalloc import start

import numpy as np
import torch
import tqdm
from scipy.spatial.distance import hamming
from sklearn.cluster import Birch
from sklearn.decomposition import IncrementalPCA
from sklearn.pipeline import Pipeline
from torch.autograd import Variable
from torchvision import datasets, transforms

from utils import (myLRPModel, get_best_parameters, jaccard_sim, min_subarray_with_sum_gt_target, 
                    get_tarantula_score, get_ochiai_score, get_BARINEL_score)


from models.train_model_6 import Net


class DeepCP:

    def __init__(self, model_name, model, layers_structure, alpha=0.99, beta=0.7, activation_threshold=0., input_size=(1, 28, 28), train_loader=None, 
                    path_to_save_pickles="pickles", device="cpu"):
        self.model_name = model_name
        self.input_size = input_size
        self.alpha = alpha
        self.beta = beta
        self.activation_threshold = activation_threshold

        self.path_to_save_pickles = path_to_save_pickles

        self.train_loader = train_loader
        
        self.cuda = True if device == "cuda" else False

        self.model = model
        self.lrp_model = myLRPModel(model, layers_structure)
        _, activations, _, _ = self.get_relevancy_and_activations(torch.randn((1, *self.input_size)))
        self.layer_shapes = [a.shape[1] for a in activations]
        print(self.layer_shapes)
        
    def get_relevancy_and_activations(self, data):
        pass

    def generate_cdp_representation(self, data):
        relevancy, activations, g_fx, predicted_class = self.get_relevancy_and_activations(data)

        # Path Extraction
        critical_neurons_layers_test = []
        for i in range(1, len(relevancy)):
            critical_neurons_layer = min_subarray_with_sum_gt_target(relevancy[i][0], self.alpha * g_fx)
            if critical_neurons_layer.shape[0] == 0:
                return None, None, None, None

            critical_neurons_layers_test.append(critical_neurons_layer)

        cdp_representation = np.zeros(sum(self.layer_shapes))
        indices = np.cumsum([0] + self.layer_shapes)
        for i in range(len(critical_neurons_layers_test)):
            cdp_representation[indices[i] + critical_neurons_layers_test[i]] = 1

        activations_vector = np.zeros(sum(self.layer_shapes))
        indices = np.cumsum([0] + self.layer_shapes)
        for i in range(indices.shape[0]-1):
            activations_vector[indices[i]:indices[i+1]] = activations[i]

        activation_mask = np.zeros(sum(self.layer_shapes))
        indices = np.cumsum([0] + self.layer_shapes)
        for i in range(indices.shape[0]-1):
            activation_mask[indices[i]:indices[i+1]] = np.where(activations_vector[indices[i]:indices[i+1]] > self.activation_threshold, 1, 0)
            if sum(activation_mask[indices[i]:indices[i+1]]) == 0:
                activation_mask[indices[i]:indices[i+1]][np.argmax(activations_vector[indices[i]:indices[i+1]])] = 1

        return cdp_representation, critical_neurons_layers_test, predicted_class, activation_mask

    def find_critical_neurons(self):
        cdp_representations_failure = 0
        num_failures = 0
        cdp_representations_pass = 0
        num_passes = 0
        for data, target in tqdm.tqdm(self.train_loader):
            if self.cuda:
                data, target = data.cpu(), target.cpu()

            data, target = Variable(data), Variable(target)

            cdp_representation, critical_neurons_layers, predicted_class, activation_mask = self.generate_cdp_representation(data)
            if cdp_representation is None and predicted_class is None:
                continue
            
            if predicted_class != target:
                cdp_representations_failure += cdp_representation
                num_failures += 1
            # else:
            #     cdp_representations_pass += cdp_representation
            #     num_passes += 1

        mean_cdp_representation_failure = cdp_representations_failure / num_failures
        # mean_cdp_representation_pass = cdp_representations_pass / num_passes
        
        criticals_mask = np.zeros_like(mean_cdp_representation_failure)
        indices = np.cumsum([0] + self.layer_shapes)
        for i in range(indices.shape[0]-1):
            mask_failure = np.where(mean_cdp_representation_failure[indices[i]:indices[i+1]] >= self.beta, 1, 0)
            # mask_pass = np.where(mean_cdp_representation_pass[indices[i]:indices[i+1]] <= 1 - self.beta, 1, 0)
            criticals_mask[indices[i]:indices[i+1]] = mask_failure #* mask_pass
            if sum(criticals_mask[indices[i]:indices[i+1]]) == 0:
                criticals_mask[indices[i]:indices[i+1]][np.argmax(mean_cdp_representation_failure[indices[i]:indices[i+1]])] = 1

        return criticals_mask

    def calculate_hit_spectrums(self):
        critical_neurons_vector = self.find_critical_neurons()

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

            # critical_neurons_vector = cdp_representation

            if predicted_class == target:
                neuron_hit_spectrums["A_P"] += activation_mask * critical_neurons_vector
                neuron_hit_spectrums["I_P"] += (1 - activation_mask) * critical_neurons_vector
            else:
                neuron_hit_spectrums["A_F"] += activation_mask * critical_neurons_vector
                neuron_hit_spectrums["I_F"] += (1 - activation_mask) * critical_neurons_vector

        return neuron_hit_spectrums

    def get_scores_from_spectrums(self, neuron_hit_spectrums, SFL_strategy):
        if SFL_strategy == "tarantula":
            scores = get_tarantula_score(neuron_hit_spectrums)
        elif SFL_strategy == "ochiai":
            scores = get_ochiai_score(neuron_hit_spectrums)
        elif SFL_strategy == "barinel":
            scores = get_BARINEL_score(neuron_hit_spectrums)
        else:
            raise Exception("wrong SFL strategy!")
            
        return scores

    def get_layerwise_suspiciousness_scores(self, scores_vector):
        layerwise_suspicousness_scores = []
        indices = np.cumsum([0] + self.layer_shapes)
        for i in range(indices.shape[0]-1):
            layer_scores = scores_vector[indices[i]:indices[i+1]]
            layer_scores = list(zip(range(layer_scores.shape[0]), layer_scores))
            layerwise_suspicousness_scores.append(layer_scores)

        return layerwise_suspicousness_scores

    def write_objects(self, object_dictionary, SFL_strategy):
        with open(os.path.join(self.path_to_save_pickles, f"{self.model_name}_{SFL_strategy}_objects.pickle"), 'wb') as handle1: 
            pickle.dump(object_dictionary, handle1, protocol=pickle.HIGHEST_PROTOCOL)

    def run(self):
        if not os.path.isdir(self.path_to_save_pickles):
            os.makedirs(self.path_to_save_pickles)

        neuron_hit_spectrums = self.calculate_hit_spectrums()

        scores_vector = self.get_scores_from_spectrums(neuron_hit_spectrums, "tarantula")
        layerwise_scores = self.get_layerwise_suspiciousness_scores(scores_vector)
        self.write_objects({
            "neuron_hit_spectrums": neuron_hit_spectrums,
            "scores_vector": scores_vector,
            "layerwise_suspicousness_scores": layerwise_scores,
        }, "tarantula")

        scores_vector = self.get_scores_from_spectrums(neuron_hit_spectrums, "ochiai")
        layerwise_scores = self.get_layerwise_suspiciousness_scores(scores_vector)
        self.write_objects({
            "neuron_hit_spectrums": neuron_hit_spectrums,
            "scores_vector": scores_vector,
            "layerwise_suspicousness_scores": layerwise_scores,
        }, "ochiai")

        scores_vector = self.get_scores_from_spectrums(neuron_hit_spectrums, "barinel")
        layerwise_scores = self.get_layerwise_suspiciousness_scores(scores_vector)
        self.write_objects({
            "neuron_hit_spectrums": neuron_hit_spectrums,
            "scores_vector": scores_vector,
            "layerwise_suspicousness_scores": layerwise_scores,
        }, "barinel")