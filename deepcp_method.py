import os
import pickle
import random

import numpy as np
import torch
import tqdm
from scipy.spatial.distance import hamming
from sklearn.cluster import Birch
from sklearn.decomposition import IncrementalPCA
from sklearn.pipeline import Pipeline
from torch.autograd import Variable

from utils import (myLRPModel, get_best_parameters, jaccard_sim, min_subarray_with_sum_gt_target, 
                    get_tarantula_score, get_ochiai_score, get_BARINEL_score)


class DeepCP:

    def __init__(self, model_name, model, layers_structure, input_size=(1, 28, 28), train_loader=None, test_loader=None, 
                    path_to_save_pickles="pickles", device="cpu", batch_size=16, alpha=0.9, 
                    beta=0.6, min_match=0.8, PCA_n_components=[], Birch_thresholds=[], Birch_n_clusters=[]):
        self.model_name = model_name
        self.input_size = input_size

        self.path_to_save_activations = self.model_name + "_activations"
        self.path_to_save_representations = self.model_name + "_representations"
        self.path_to_save_pickles = path_to_save_pickles

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.cuda = True if device == "cuda" else False
        self.batch_size = batch_size

        self.alpha = alpha
        self.beta = beta
        self.min_match = min_match
        self.PCA_n_components = PCA_n_components
        self.Birch_thresholds = Birch_thresholds
        self.Birch_n_clusters = Birch_n_clusters

        self.lrp_model = myLRPModel(model, layers_structure)
        relevancy, _ = self.lrp_model.forward(torch.randn((1, *input_size)))
        self.layer_shapes = [list(r[0].shape)[0] for r in relevancy[1:]]

    def get_relevancy_and_activations(self, data):
        pass

    def generate_cdp_representation(self, data):
        relevancy, activations = self.get_relevancy_and_activations(data)
        g_fx = torch.sum(relevancy[0]).item()

        # Path Extraction
        try:
            critical_neurons_layers_test = []
            for i in range(1, len(relevancy)):
                critical_neurons_layer = min_subarray_with_sum_gt_target(relevancy[i][0], self.alpha * g_fx)
                if critical_neurons_layer.shape[0] == 0:
                    return None, None, None, None

                critical_neurons_layers_test.append(critical_neurons_layer)
        except:
            return None, None, None, None

        predicted_class = torch.max(relevancy[-1], dim=1).indices.item()

        cdp_representation = torch.zeros(sum(self.layer_shapes))
        for i in range(len(critical_neurons_layers_test)):
            if i == 0:
                cdp_representation[critical_neurons_layers_test[i]] = 1
            else:
                cdp_representation[sum(self.layer_shapes[:i]) + critical_neurons_layers_test[i]] = 1

        return cdp_representation, critical_neurons_layers_test, predicted_class, activations

    def generate_class_separated_cdps(self):
        if not os.path.isdir(self.path_to_save_activations):
            os.mkdir(self.path_to_save_activations)

        if not os.path.isdir(self.path_to_save_representations):
            os.mkdir(self.path_to_save_representations)

        class_separated_samples = {}
        for data, target in tqdm.tqdm(self.train_loader):
            if self.cuda:
                data, target = data.cpu(), target.cpu()

            data, target = Variable(data), Variable(target)

            cdp_representation, _, predicted_class, activations = self.generate_cdp_representation(data)
            if cdp_representation is None and predicted_class is None:
                continue

            class_samples = class_separated_samples.get(predicted_class, [])
            filename = str(random.getrandbits(24)) + ".pickle"
            activations_name = os.path.join(self.path_to_save_activations, filename)
            with open(activations_name, "wb") as handle:
                pickle.dump(activations, handle, protocol=pickle.HIGHEST_PROTOCOL)

            representation_name = os.path.join(self.path_to_save_representations, filename)
            with open(representation_name, "wb") as handle:
                pickle.dump(cdp_representation.detach().numpy(), handle, protocol=pickle.HIGHEST_PROTOCOL)

            class_samples.append((representation_name, activations_name))
            class_separated_samples[predicted_class] = class_samples
            
        return class_separated_samples

    def generate_abstract_cdps(self, class_separated_samples):
        # TODO: reform the cdp representation because of its huge size for large networks

        decision_graph = {}
        decision_birch = {}
        for c in tqdm.tqdm(sorted(class_separated_samples.keys())):
            # Path Abstraction: Intra-Class Path Clustering
            if len(class_separated_samples[c]) < 4:
                continue

            cdp_representations = []
            sample_activations = []
            for cdp_reps, sample_activs in class_separated_samples[c]: #[:3000]: # due to out of memory!
                with open(cdp_reps, 'rb') as handle:
                    representation = pickle.load(handle)

                cdp_representations.append(representation)
                sample_activations.append(sample_activs)

            optimal_threshold, optimal_n_clusters, optimal_n_components = get_best_parameters(cdp_representations, self.PCA_n_components, self.Birch_thresholds, self.Birch_n_clusters, self.batch_size)
            decision_birch[c] = Pipeline([
                ('dim_red', IncrementalPCA(n_components=optimal_n_components, batch_size=self.batch_size)), 
                ('clustering', Birch(threshold=optimal_threshold, n_clusters=optimal_n_clusters))
                ])
            decision_birch[c].fit(cdp_representations)
            cluster_labels = decision_birch[c].predict(cdp_representations)

            # Path Abstraction: Path Merging
            for cluster_index in set(cluster_labels):
                cluster_samples = np.array(cdp_representations)[cluster_labels==cluster_index]
                neuron_criticality_weights = cluster_samples.mean(axis=0)

                s_hat_beta = []
                for i in range(len(self.layer_shapes)):
                    if i == 0:
                        start_index = 0
                        end_index = sum(self.layer_shapes[:i+1])
                    else:
                        start_index = sum(self.layer_shapes[:i])
                        end_index = sum(self.layer_shapes[:i+1])

                    abstract_critical_neurons = np.where(neuron_criticality_weights[start_index:end_index] >= self.beta)[0]
                    s_hat_beta.append(abstract_critical_neurons)

                cluster_samples_activations = [sample_activations[i] for i in np.where(cluster_labels == cluster_index)[0]]
                activations_filename = os.path.join(self.path_to_save_activations, f"cluster_samples_activations_{c}_{cluster_index}.pickle")
                with open(activations_filename, 'wb') as handle:
                    pickle.dump(cluster_samples_activations, handle, protocol=pickle.HIGHEST_PROTOCOL)

                decision_graph[(c, cluster_index)] = s_hat_beta

        return decision_graph, decision_birch

    def calculate_acdp_hit_spectrums(self, decision_graph, decision_birch):
        cdp_spectrums = {}
        for key in decision_graph.keys():
            cdp_spectrums[key] = {"A_P": 0, "I_P": 0, "A_F": 0, "I_F": 0}

        for data, target in tqdm.tqdm(self.test_loader):
            if self.cuda:
                data, target = data.cpu(), target.cpu()

            data, target = Variable(data), Variable(target)

            cdp_representation, critical_neurons_layers_test, predicted_class, activations = self.generate_cdp_representation(data)
            if (cdp_representation is None and critical_neurons_layers_test is None and predicted_class is None) or predicted_class not in decision_birch.keys():
                continue

            predicted_cluster = decision_birch[predicted_class].predict(cdp_representation[None, ...])[0]
            critical_neurons_layers_abstract_cdp = decision_graph[(predicted_class, predicted_cluster)]

            jacc_sims_of_cdps = [jaccard_sim(s_i.tolist(), s_hat_i.tolist()) for s_i, s_hat_i in zip(critical_neurons_layers_test, critical_neurons_layers_abstract_cdp)]
            mean_jacc_sim_structural = sum(jacc_sims_of_cdps) / len(jacc_sims_of_cdps)

            with open(os.path.join(self.path_to_save_activations, f"cluster_samples_activations_{predicted_class}_{predicted_cluster}.pickle"), 'rb') as handle:
                cluster_samples_activations = pickle.load(handle)

            # TODO: sample activations w.r.t a criteria
            subset_cluster_samples_activations = random.sample(cluster_samples_activations, min(10, len(cluster_samples_activations)))

            similarities = []
            for sample_activation_path in subset_cluster_samples_activations:
                with open(sample_activation_path, "rb") as handle:
                    sample_activation = pickle.load(handle)

                similarities_ = []
                for sample_layer_actv, layer_critical_neurons, test_layer_actv in zip(sample_activation, critical_neurons_layers_abstract_cdp, activations):
                    sample_layer_state = np.where(sample_layer_actv.detach().cpu() > 0., 1, 0)[0]
                    test_layer_state = np.where(test_layer_actv.detach().cpu() > 0., 1, 0)[0]

                    similarity = 1 - hamming(sample_layer_state[layer_critical_neurons].tolist(), test_layer_state[layer_critical_neurons].tolist())
                    similarities_.append(similarity)

                temp_ = sum(similarities_) / len(similarities_)
                similarities.append(temp_)

            mean_jacc_sim_activational = sum(similarities) / len(similarities) if len(similarities) != 0 else 0.

            if mean_jacc_sim_structural >= self.min_match and mean_jacc_sim_activational >= self.min_match:
                if predicted_class == target.item():
                    cdp_spectrums[(predicted_class, predicted_cluster)]["A_P"] = cdp_spectrums[(predicted_class, predicted_cluster)]["A_P"] + 1
                    # increase I_P for other clusters
                    for class_index in decision_birch.keys():
                        for cluster_index in range(decision_birch[class_index]["clustering"].n_clusters):
                            if (class_index, cluster_index) != (predicted_class, predicted_cluster):
                                cdp_spectrums[(class_index, cluster_index)]["I_P"] = cdp_spectrums[(class_index, cluster_index)]["I_P"] + 1
                else:
                    cdp_spectrums[(predicted_class, predicted_cluster)]["A_F"] = cdp_spectrums[(predicted_class, predicted_cluster)]["A_F"] + 1
                    # increase I_F for corresponding cluster in target class
                    clusters_of_target_class = list(range(decision_birch[target.item()]["clustering"].n_clusters))
                    for cluster_index in clusters_of_target_class:
                        cdp_spectrums[(target.item(), cluster_index)]["I_F"] = cdp_spectrums[(target.item(), cluster_index)]["I_F"] + 1

        return cdp_spectrums

    def get_scores_from_spectrums(self, cdp_spectrums, SFL_strategy):
        scores = []
        for key, path_spectrum in cdp_spectrums.items():
            if SFL_strategy == "tarantula":
                score = get_tarantula_score(path_spectrum)
            elif SFL_strategy == "ochiai":
                score = get_ochiai_score(path_spectrum)
            elif SFL_strategy == "barinel":
                score = get_BARINEL_score(path_spectrum)
            else:
                raise Exception("wrong SFL strategy!")
                
            scores.append((key, score))

        return scores

    def run(self):
        if not os.path.isdir(self.path_to_save_pickles):
            os.makedirs(self.path_to_save_pickles)

        class_separated_samples = self.generate_class_separated_cdps()
        decision_graph, decision_birch = self.generate_abstract_cdps(class_separated_samples)

        with open(os.path.join(self.path_to_save_pickles, f"{self.model_name}_class_separated_samples.pickle"), 'wb') as handle1, open(os.path.join(self.path_to_save_pickles, f"{self.model_name}_decision_graph.pickle"), 'wb') as handle2, open(os.path.join(self.path_to_save_pickles, f"{self.model_name}_decision_birch.pickle"), 'wb') as handle3:
            pickle.dump(class_separated_samples, handle1, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(decision_graph, handle2, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(decision_birch, handle3, protocol=pickle.HIGHEST_PROTOCOL)                

        cdp_spectrums = self.calculate_acdp_hit_spectrums(decision_graph, decision_birch)

        with open(os.path.join(self.path_to_save_pickles, f"{self.model_name}_cdp_spectrums.pickle"), 'wb') as handle:
            pickle.dump(cdp_spectrums, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if not os.path.isdir("results"):
            os.makedirs("results")

        scores = self.get_scores_from_spectrums(cdp_spectrums, "tarantula")
        scores = sorted(scores, key=lambda item: -item[1])
        with open(os.path.join("results", f"{self.model_name}_tarantula.txt"), "w") as file:
            for cdp_class, score in scores:
                cdp = str(list(map(lambda item: item.tolist(), decision_graph[cdp_class])))
                file.write(f"{cdp_class}\t{score}\t{cdp}\n")

        scores = self.get_scores_from_spectrums(cdp_spectrums, "ochiai")
        scores = sorted(scores, key=lambda item: -item[1])
        with open(os.path.join("results", f"{self.model_name}_ochiai.txt"), "w") as file:
            for cdp_class, score in scores:
                cdp = str(list(map(lambda item: item.tolist(), decision_graph[cdp_class])))
                file.write(f"{cdp_class}\t{score}\t{cdp}\n")

        scores = self.get_scores_from_spectrums(cdp_spectrums, "barinel")
        scores = sorted(scores, key=lambda item: -item[1])
        with open(os.path.join("results", f"{self.model_name}_barinel.txt"), "w") as file:
            for cdp_class, score in scores:
                cdp = str(list(map(lambda item: item.tolist(), decision_graph[cdp_class])))
                file.write(f"{cdp_class}\t{score}\t{cdp}\n")