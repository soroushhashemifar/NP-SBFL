import pickle
import numpy as np
# from utils import get_BARINEL_score, get_ochiai_score, get_tarantula_score
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
import pandas as pd
import os
import sys


def get_suspicious_neurons(decision_graph, model_name, SFL_strategy, metric_threshold):
    with open(os.path.join("..", "results", f"{model_name}_{SFL_strategy}.txt")) as file:
        content = file.readlines()
        content = list(map(lambda item: item.strip(), content))
        content = list(map(lambda item: item.split("\t"), content))
        scores = list(map(lambda item: (eval(item[0]), float(item[1]), eval(item[2])), content))

    faulty_scores = list(filter(lambda item: item[1] >= metric_threshold, scores))
    faulty_cdps = list(map(lambda item: item[0], faulty_scores))
    fpl_detected_neurons = [(layer_index, neuron_index) for faulty_cdp in faulty_cdps for layer_index in range(len(decision_graph[faulty_cdp])-1) for neuron_index in decision_graph[faulty_cdp][layer_index]]
    fpl_detected_neurons = set(fpl_detected_neurons)

    return fpl_detected_neurons

def save_susp_distributions_plot(model_name, metric_thresholds, width):
    with open(f'../pickles/{model_name}_decision_graph.pickle', 'rb') as handle:
        decision_graph = pickle.load(handle)

    num_neurons_per_layers_per_metric = []
    for metric, metric_threshold in metric_thresholds:
        fpl_detected_neurons = get_suspicious_neurons(decision_graph, model_name, metric, metric_threshold)

        num_neurons_per_layer = list(map(lambda item: item[0], fpl_detected_neurons))
        num_neurons_per_layer = list(np.unique(num_neurons_per_layer, return_counts=True))
        num_neurons_per_layer[1] = num_neurons_per_layer[1] / layers_num_units
        num_neurons_per_layer = [(num_neurons_per_layer[0][i], num_neurons_per_layer[1][i]) for i in range(num_neurons_per_layer[0].shape[0])]
        num_neurons_per_layer = [(i, value, metric.capitalize()) for i, value in num_neurons_per_layer]
        num_neurons_per_layers_per_metric.extend(num_neurons_per_layer)

    num_neurons_per_layer_df = pd.DataFrame.from_dict({
        "Dense Layers": [f"Layer {i+1}" for i, value, met in num_neurons_per_layers_per_metric], 
        "Suspicious Neurons ratio": [value for i, value, met in num_neurons_per_layers_per_metric], 
        "metric": [met for i, value, met in num_neurons_per_layers_per_metric]})

    print(num_neurons_per_layer_df)

    g = sns.barplot(
        data=num_neurons_per_layer_df,
        x="Dense Layers", y="Suspicious Neurons ratio",
        hue="metric", width=width
    )
    sns.move_legend(g, "best")
    filename = f'figures/{model_name}_susp_distribution.png'
    plt.savefig(filename)
    plt.clf()


# layers_num_units = np.array([128, 32])
# metric_thresholds = [("tarantula", 0.99), ("ochiai", 0.26), ("barinel", 0.18)]
# save_susp_distributions_plot("Model_1", metric_thresholds, width=0.15)

# layers_num_units = np.array([256, 128, 64, 32])
# metric_thresholds = [("tarantula", 0.87), ("ochiai", 0.05), ("barinel", 0.014)]
# save_susp_distributions_plot("Model_2", metric_thresholds, width=0.30)

layers_num_units = np.array([16, 16, 32, 32, 64, 64, 64, 64, 128, 64])
metric_thresholds = [("tarantula", 0.90802413), ("ochiai", 0.37238748), ("barinel", 0.3964497)]
save_susp_distributions_plot("Model_3", metric_thresholds, width=0.60)