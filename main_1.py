from ipaddress import collapse_addresses
import os
import pickle
import random
import time
from lrp_src.visualize import plot_relevance_scores

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from scipy.spatial.distance import hamming
from sklearn.cluster import KMeans, Birch
from sklearn.decomposition import IncrementalPCA
from sklearn.pipeline import Pipeline
from torch.autograd import Variable
from torchvision import datasets, transforms

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

import lrp_src
from config import args
from train_model import Net
from utils import (get_best_number_of_clusters, get_best_parameters, jaccard_sim,
                   min_subarray_with_sum_gt_target, myLRPModel, get_tarantula_score, get_ochiai_score, get_Dstar_score, get_BARINEL_score)

lrp_src.lrp_layers.top_k_percent = args['ALPHA']


def generate_cdp_representation(lrp_model, data, layer_shapes):
    # Layer-Wise Relevance Propagation (LRP)
    relevancy, activations = lrp_model.forward(data)
    g_fx = torch.sum(relevancy[0][0]).item()

    # Path Extraction
    try:
        critical_neurons_layers_test = []
        for i in range(1, len(relevancy)):
            critical_neurons_layer = min_subarray_with_sum_gt_target(relevancy[i][0], args['ALPHA'] * g_fx)
            if critical_neurons_layer.shape[0] == 0:
                return None, None, None, None

            critical_neurons_layers_test.append(critical_neurons_layer)
    except:
        return None, None, None, None

    predicted_class = torch.max(relevancy[-1], dim=1).indices.item()

    cdp_representation = torch.zeros(sum(layer_shapes))
    for i in range(len(critical_neurons_layers_test)):
        if i == 0:
            cdp_representation[critical_neurons_layers_test[i]] = 1
        else:
            cdp_representation[sum(layer_shapes[:i]) + critical_neurons_layers_test[i]] = 1

    return cdp_representation, critical_neurons_layers_test, predicted_class, activations

def generate_class_separated_cdps(new_train_loader, lrp_model, args, layer_shapes):
    class_separated_samples = {}
    # index = 0
    for data, target in tqdm.tqdm(new_train_loader):
        # if index == 10000: break
        if args['cuda']:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)

        cdp_representation, _, predicted_class, activations = generate_cdp_representation(lrp_model, data.view(-1, 784), layer_shapes)
        if cdp_representation is None and predicted_class is None:
            continue

        class_samples = class_separated_samples.get(predicted_class, [])
        activations_name = "activations/" + str(random.getrandbits(24)) + ".pickle"
        with open(activations_name, "wb") as handle:
            pickle.dump(activations, handle, protocol=pickle.HIGHEST_PROTOCOL)

        class_samples.append((cdp_representation.detach().numpy(), activations_name))
        class_separated_samples[predicted_class] = class_samples

        # index += 1
        
    return class_separated_samples

def generate_abstract_cdps(class_separated_samples, layer_shapes):
    # TODO: reform the cdp representation because of its huge size for large networks

    decision_graph = {}
    decision_kmeans = {}
    for c in tqdm.tqdm(sorted(class_separated_samples.keys())):
        # Path Abstraction: Intra-Class Path Clustering
        if len(class_separated_samples[c]) < 4:
            continue

        cdp_representations = []
        sample_activations = []
        for cdp_reps, sample_activs in class_separated_samples[c]:
            cdp_representations.append(cdp_reps)
            sample_activations.append(sample_activs)

        # optimal_n_clusters, optimal_n_components = get_best_number_of_clusters(cdp_representations)
        # decision_kmeans[c] = Pipeline([
        #     ('dim_red', IncrementalPCA(n_components=optimal_n_components, batch_size=args['batch_size'])), # Reason: NPC will failed with large networks with huge number of neurons
        #     ('clustering', KMeans(n_clusters=optimal_n_clusters, random_state=4)), 
        #     ])
        optimal_threshold, optimal_n_clusters, optimal_n_components = get_best_parameters(cdp_representations)
        decision_kmeans[c] = Pipeline([
            ('dim_red', IncrementalPCA(n_components=optimal_n_components, batch_size=args['batch_size'])), 
            ('clustering', Birch(threshold=optimal_threshold, n_clusters=optimal_n_clusters))
            ])
        decision_kmeans[c].fit(cdp_representations)
        cluster_labels = decision_kmeans[c].predict(cdp_representations)

        # print(np.unique(cluster_labels))
        # X_embedded = IncrementalPCA(n_components=2, batch_size=128).fit_transform(cdp_representations)
        # cdict = {0: 'blue', 1: 'orange', 2: 'green', 3: 'red', 4: 'purple', 5: 'brown', 6: 'pink', 7: 'gray', 8: 'olive', 9: 'cyan'}
        # fig, ax = plt.subplots()
        # for g in np.unique(cluster_labels):
        #     ix = np.where(cluster_labels == g)
        #     ax.scatter(X_embedded[ix, 0], X_embedded[ix, 1], c = cdict[g], label = g, s = 10)
        # ax.legend()
        # plt.savefig(f'cdp_representation_clusters_{c}.png')
        # continue

        # Path Abstraction: Path Merging
        for cluster_index in set(cluster_labels):
            cluster_samples = np.array(cdp_representations)[cluster_labels==cluster_index]
            neuron_criticality_weights = cluster_samples.mean(axis=0)

            s_hat_beta = []
            for i in range(len(layer_shapes)):
                if i == 0:
                    start_index = 0
                    end_index = sum(layer_shapes[:i+1])
                else:
                    start_index = sum(layer_shapes[:i])
                    end_index = sum(layer_shapes[:i+1])

                abstract_critical_neurons = np.where(neuron_criticality_weights[start_index:end_index] >= args['BETA'])[0]
                s_hat_beta.append(abstract_critical_neurons)

            cluster_samples_activations = [sample_activations[i] for i in np.where(cluster_labels == cluster_index)[0]]
            activations_filename = f"./activations/cluster_samples_activations_{c}_{cluster_index}.pickle"
            with open(activations_filename, 'wb') as handle:
                pickle.dump(cluster_samples_activations, handle, protocol=pickle.HIGHEST_PROTOCOL)

            decision_graph[(c, cluster_index)] = s_hat_beta

    return decision_graph, decision_kmeans

def calculate_acdp_hit_spectrums(new_test_loader, lrp_model, decision_graph, decision_kmeans, layer_shapes):
    cdp_spectrums = {}
    for key in decision_graph.keys():
        cdp_spectrums[key] = {"A_P": 0, "I_P": 0, "A_F": 0, "I_F": 0}

    for data, target in tqdm.tqdm(new_test_loader):
        if args['cuda']:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)

        cdp_representation, critical_neurons_layers_test, predicted_class, activations = generate_cdp_representation(lrp_model, data.view(-1, 784), layer_shapes)
        if (cdp_representation is None and critical_neurons_layers_test is None and predicted_class is None) or predicted_class not in decision_kmeans.keys():
            continue

        predicted_cluster = decision_kmeans[predicted_class].predict(cdp_representation[None, ...])[0]
        critical_neurons_layers_abstract_cdp = decision_graph[(predicted_class, predicted_cluster)]

        with open(f"./activations/cluster_samples_activations_{predicted_class}_{predicted_cluster}.pickle", 'rb') as handle:
            cluster_samples_activations = pickle.load(handle)

        jacc_sims_of_cdps = [jaccard_sim(s_i.tolist(), s_hat_i.tolist()) for s_i, s_hat_i in zip(critical_neurons_layers_test, critical_neurons_layers_abstract_cdp)]
        mean_jacc_sim_structural = sum(jacc_sims_of_cdps) / len(jacc_sims_of_cdps)

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

        if mean_jacc_sim_structural >= args['MIN_MATCH'] and mean_jacc_sim_activational >= args['MIN_MATCH']:
            if predicted_class == target.item():
                cdp_spectrums[(predicted_class, predicted_cluster)]["A_P"] = cdp_spectrums[(predicted_class, predicted_cluster)]["A_P"] + 1
                # increase I_P for other clusters
                for class_index in decision_kmeans.keys():
                    for cluster_index in range(decision_kmeans[class_index]["clustering"].n_clusters):
                        if (class_index, cluster_index) != (predicted_class, predicted_cluster):
                            cdp_spectrums[(class_index, cluster_index)]["I_P"] = cdp_spectrums[(class_index, cluster_index)]["I_P"] + 1
            else:
                cdp_spectrums[(predicted_class, predicted_cluster)]["A_F"] = cdp_spectrums[(predicted_class, predicted_cluster)]["A_F"] + 1
                # increase I_F for corresponding cluster in target class
                clusters_of_target_class = list(range(decision_kmeans[target.item()]["clustering"].n_clusters))
                for cluster_index in clusters_of_target_class:
                    cdp_spectrums[(target.item(), cluster_index)]["I_F"] = cdp_spectrums[(target.item(), cluster_index)]["I_F"] + 1

    return cdp_spectrums

if __name__ == "__main__":
    start_time = time.time()
    new_train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=1, shuffle=True)
    new_test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=1, shuffle=True)

    model = Net()
    model.load_state_dict(torch.load("./mymodel_mnist.pth", map_location=torch.device("cuda" if args['cuda'] else "cpu")))
    model = model.to("cuda" if args['cuda'] else "cpu")
    model.eval()

    layers_structure = [
        torch.nn.Flatten(1), 
        model.fc1, torch.nn.ReLU(),
        model.fc2, torch.nn.ReLU(),
        model.fc3
    ]

    layer_shapes = [128, 32, 10]

    lrp_model = myLRPModel(model, layers_structure)

    # Visualizing LRP outputs
    # for i, (x, y) in enumerate(new_test_loader):
    #     if args['cpu']:
    #         x, y = x.cpu(), y.cpu()

    #     r, _ = lrp_model.forward(x)
    #     r[0] = r[0].cpu()
    #     plot_relevance_scores(x=x, r=r[0], name=str(i), config={"output_dir": "/home/soroushh/Downloads/Faulty-CDP-localization"})

    #     if i == 10:
    #         break
    
    # exit()

    if not os.path.isfile("./decision_graph.pickle"):
        class_separated_samples = generate_class_separated_cdps(new_train_loader, lrp_model, args, layer_shapes)

        with open('./class_separated_samples.pickle', 'wb') as handle:
            pickle.dump(class_separated_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # with open('./class_separated_samples.pickle', 'rb') as handle:
        #     class_separated_samples = pickle.load(handle)

        # # Visualizing cdp representations
        # total_cdps = []
        # total_labels = []
        # for class_idx in class_separated_samples.keys():
        #     cdps = list(map(lambda item: item[0], class_separated_samples[class_idx]))
        #     labels = [class_idx] * len(cdps)
        #     total_cdps.extend(cdps)
        #     total_labels.extend(labels)

        # X_embedded = IncrementalPCA(n_components=2, batch_size=128).fit_transform(total_cdps)
        # cdict = {0: 'blue', 1: 'orange', 2: 'green', 3: 'red', 4: 'purple', 5: 'brown', 6: 'pink', 7: 'gray', 8: 'olive', 9: 'cyan'}
        # fig, ax = plt.subplots()
        # for g in np.unique(total_labels):
        #     ix = np.where(total_labels == g)
        #     ax.scatter(X_embedded[ix, 0], X_embedded[ix, 1], c = cdict[g], label = g, s = 10)
        # ax.legend()
        # plt.savefig('class_separated_samples.png')

        decision_graph, decision_kmeans = generate_abstract_cdps(class_separated_samples, layer_shapes)

        with open('./decision_graph.pickle', 'wb') as handle, open('./decision_kmeans.pickle', 'wb') as handle2:
            pickle.dump(decision_graph, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(decision_kmeans, handle2, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('./decision_graph.pickle', 'rb') as handle, open('./decision_kmeans.pickle', 'rb') as handle2:
            decision_graph = pickle.load(handle)
            decision_kmeans = pickle.load(handle2)
            
    if not os.path.isfile("./cdp_spectrums.pickle"):
        cdp_spectrums = calculate_acdp_hit_spectrums(new_test_loader, lrp_model, decision_graph, decision_kmeans, layer_shapes)

        with open('cdp_spectrums.pickle', 'wb') as handle:
            pickle.dump(cdp_spectrums, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('./cdp_spectrums.pickle', 'rb') as handle:
            cdp_spectrums = pickle.load(handle)

    print("Total runtime:", time.time()-start_time)

    scores = []
    for key, path_spectrum in cdp_spectrums.items():
        if args['SFL_strategy'] == "tarantula":
            score = get_tarantula_score(path_spectrum)
        elif args['SFL_strategy'] == "ochiai":
            score = get_ochiai_score(path_spectrum)
        elif args['SFL_strategy'] == "d_star":
            score = get_Dstar_score(path_spectrum, 0.8)
        elif args['SFL_strategy'] == "barinel":
            score = get_BARINEL_score(path_spectrum)
        else:
            raise Exception("wrong SFL strategy!")
            
        scores.append((key, score))

    scores = sorted(scores, key=lambda item: -item[1])
    SFL_strategy = args['SFL_strategy']
    with open(f"results_{SFL_strategy}.txt", "w") as file:
        for cdp, score in scores:
            file.write(f"{cdp}, score = {score}\n")
            _cdp = str(list(map(lambda item: item.tolist(), decision_graph[cdp])))
            file.write(f"{_cdp}\n\n")

