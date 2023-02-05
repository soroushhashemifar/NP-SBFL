import pickle
import numpy as np
from utils import get_BARINEL_score, get_ochiai_score, get_tarantula_score
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
import pandas as pd


with open('../pickles/Model_1_decision_graph.pickle', 'rb') as handle1, open('../pickles/Model_1_cdp_spectrums.pickle', 'rb') as handle2:
    decision_graph = pickle.load(handle1)
    cdp_spectrums = pickle.load(handle2)

layers_units = np.array([128, 32])

num_neurons_per_layers = []
for metric, metric_threshold in [("tarantula", 0.90), ("ochiai", 0.14), ("barinel", 0.043)]:
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

    fpl_detected_neurons = [(layer_index, neuron_index) for faulty_cdp in faulty_cdps for layer_index in range(2) for neuron_index in decision_graph[faulty_cdp][layer_index]]
    fpl_detected_neurons = set(fpl_detected_neurons)
    num_neurons_per_layer = list(map(lambda item: item[0], fpl_detected_neurons))
    num_neurons_per_layer = list(np.unique(num_neurons_per_layer, return_counts=True))
    num_neurons_per_layer[1] = num_neurons_per_layer[1] / layers_units
    num_neurons_per_layer = [(num_neurons_per_layer[0][i], num_neurons_per_layer[1][i]) for i in range(num_neurons_per_layer[0].shape[0])]
    num_neurons_per_layer = [(i, value, metric.capitalize()) for i, value in num_neurons_per_layer]
    num_neurons_per_layers.extend(num_neurons_per_layer)

num_neurons_per_layer_df = pd.DataFrame.from_dict({
    "Dense Layers": [f"Layer {i+1}" for i, value, met in num_neurons_per_layers], 
    "Suspicious Neurons ratio": [value for i, value, met in num_neurons_per_layers], 
    "metric": [met for i, value, met in num_neurons_per_layers]})

print(num_neurons_per_layer_df)

g = sns.barplot(
    data=num_neurons_per_layer_df,
    x="Dense Layers", y="Suspicious Neurons ratio",
    hue="metric", width=0.15
)
sns.move_legend(g, "upper center")
plt.savefig('Model_1_dnn_units.png')
plt.clf()

with open('../pickles/Model_2_decision_graph.pickle', 'rb') as handle1, open('../pickles/Model_2_cdp_spectrums.pickle', 'rb') as handle2:
    decision_graph = pickle.load(handle1)
    cdp_spectrums = pickle.load(handle2)

layers_units = np.array([256, 128, 64, 32])

num_neurons_per_layers = []
for metric, metric_threshold in [("tarantula", 0.87), ("ochiai", 0.05), ("barinel", 0.01)]:
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

    fpl_detected_neurons = [(layer_index, neuron_index) for faulty_cdp in faulty_cdps for layer_index in range(4) for neuron_index in decision_graph[faulty_cdp][layer_index]]
    fpl_detected_neurons = set(fpl_detected_neurons)
    num_neurons_per_layer = list(map(lambda item: item[0], fpl_detected_neurons))
    num_neurons_per_layer = list(np.unique(num_neurons_per_layer, return_counts=True))
    num_neurons_per_layer[1] = num_neurons_per_layer[1] / layers_units
    num_neurons_per_layer = [(num_neurons_per_layer[0][i], num_neurons_per_layer[1][i]) for i in range(num_neurons_per_layer[0].shape[0])]
    num_neurons_per_layer = [(i, value, metric.capitalize()) for i, value in num_neurons_per_layer]
    num_neurons_per_layers.extend(num_neurons_per_layer)

num_neurons_per_layer_df = pd.DataFrame.from_dict({
    "Dense Layers": [f"Layer {i+1}" for i, value, met in num_neurons_per_layers], 
    "Suspicious Neurons ratio": [value for i, value, met in num_neurons_per_layers], 
    "metric": [met for i, value, met in num_neurons_per_layers]})

print(num_neurons_per_layer_df)

g = sns.barplot(
    data=num_neurons_per_layer_df,
    x="Dense Layers", y="Suspicious Neurons ratio",
    hue="metric", width=0.30
)
sns.move_legend(g, "upper center")
plt.savefig('mnist_dnn2_units.png')
plt.clf()

with open('../pickles/Model_3_decision_graph.pickle', 'rb') as handle1, open('../pickles/Model_3_cdp_spectrums.pickle', 'rb') as handle2:
    decision_graph = pickle.load(handle1)
    cdp_spectrums = pickle.load(handle2)

layers_units = np.array([16, 32, 64, 64, 128, 64])

num_neurons_per_layers = []
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
    num_neurons_per_layer = list(map(lambda item: item[0], fpl_detected_neurons))
    num_neurons_per_layer = list(np.unique(num_neurons_per_layer, return_counts=True))
    print(num_neurons_per_layer)
    num_neurons_per_layer[1] = num_neurons_per_layer[1] / layers_units
    num_neurons_per_layer = [(num_neurons_per_layer[0][i], num_neurons_per_layer[1][i]) for i in range(num_neurons_per_layer[0].shape[0])]
    num_neurons_per_layer = [(i, value, metric.capitalize()) for i, value in num_neurons_per_layer]
    num_neurons_per_layers.extend(num_neurons_per_layer)

num_neurons_per_layer_df = pd.DataFrame.from_dict({
    "Dense Layers": [f"Layer {i+1}" for i, value, met in num_neurons_per_layers], 
    "Suspicious Neurons ratio": [value for i, value, met in num_neurons_per_layers], 
    "metric": [met for i, value, met in num_neurons_per_layers]})

print(num_neurons_per_layer_df)

g = sns.barplot(
    data=num_neurons_per_layer_df,
    x="Dense Layers", y="Suspicious Neurons ratio",
    hue="metric", width=0.5
)
sns.move_legend(g, "upper center")
plt.savefig('cifar10_cnn_units.png')