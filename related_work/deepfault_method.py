import sys
sys.path.insert(0, "..")
from deepcp_method import DeepCP
import tqdm
from torch.autograd import Variable
import numpy as np


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
        index = 0
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

            index += 1
            if index == 100:
                break

        return neuron_hit_spectrums

    def get_suspicious_neurons(self, num_susp_neurons, suspicousness_scores):
        reformatted_suspicousness_scores = []
        indices = np.cumsum([0] + self.layer_shapes)
        for i in range(indices.shape[0]-1):
            layer_scores = suspicousness_scores[indices[i]:indices[i+1]]
            layer_scores = list(zip([i]*len(layer_scores), range(len(layer_scores)), layer_scores))
            reformatted_suspicousness_scores.extend(layer_scores)

        filtered_scores = list(filter(lambda item: not np.isnan(item[2]), reformatted_suspicousness_scores))

        sorted_scores = sorted(filtered_scores, key=lambda item: item[2], reverse=True)
        suspicousness_neurons = sorted_scores[:num_susp_neurons]
        
        return suspicousness_neurons

    def run(self, num_susp_neurons):
        neuron_hit_spectrums = self.calculate_hit_spectrums()

        scores_vector = self.get_scores_from_spectrums(neuron_hit_spectrums, "tarantula")
        suspicousness_neurons_tarantula = self.get_suspicious_neurons(num_susp_neurons, scores_vector)

        scores_vector = self.get_scores_from_spectrums(neuron_hit_spectrums, "ochiai")
        suspicousness_neurons_ochiai = self.get_suspicious_neurons(num_susp_neurons, scores_vector)

        scores_vector = self.get_scores_from_spectrums(neuron_hit_spectrums, "barinel")
        suspicousness_neurons_barinel = self.get_suspicious_neurons(num_susp_neurons, scores_vector)

        return suspicousness_neurons_tarantula, suspicousness_neurons_ochiai, suspicousness_neurons_barinel