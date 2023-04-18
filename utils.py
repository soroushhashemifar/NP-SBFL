import math

import numpy as np
import torch
from sklearn.cluster import Birch
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from torch.autograd import Variable
from torch.nn import functional as F

from lrp_src.lrp import LRPModel


def train(epoch, model, train_loader, optimizer, parameters):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if parameters['cuda']:
            data, target = data.cuda(), target.cuda()
        #Variables in Pytorch are differenciable. 
        data, target = Variable(data), Variable(target)
        #This will zero out the gradients for this batch. 
        optimizer.zero_grad()
        output = model(data)
        # Calculate the loss The negative log likelihood loss. It is useful to train a classification problem with C classes.
        loss = parameters['loss'](output, target)
        #dloss/dx for every Variable 
        loss.backward()
        #to do a one-step update on our parameter.
        optimizer.step()
        #Print out the loss periodically. 
        if batch_idx % parameters['log_interval'] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data))

def test(model, test_loader, parameters, scheduler=None, log=True):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if parameters['cuda']:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        test_loss += parameters['loss'](output, target).data # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)

    accuracy = 100. * correct / len(test_loader.dataset)
    if log:
        print(f'Test set: Average loss: {test_loss}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy}%)')

    if scheduler is not None:
        scheduler.step(test_loss)

    return accuracy, test_loss


class myLRPModel(LRPModel):

    def __init__(self, model: torch.nn.Module, layers_structure: list) -> None:
        self.layers_structure = layers_structure
        self.relevancy_layers_to_filter = ["RelevancePropagationMaxPool2d", "RelevancePropagationFlatten", "RelevancePropagationReLU", "RelevancePropagationDropout", "RelevancePropagationIdentity"]

        super().__init__(model)
        
    def _get_layer_operations(self) -> torch.nn.ModuleList:
        """Get all network operations and store them in a list.
        This method is adapted to VGG networks from PyTorch's Model Zoo.
        Modify this method to work also for other networks.
        Returns:
            Layers of original model stored in module list.
        """
        layers = torch.nn.ModuleList(self.layers_structure)

        return layers

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Forward method that first performs standard inference followed by layer-wise relevance propagation.
        Args:
            x: Input tensor representing an image / images (N, C, H, W).
        Returns:
            Tensor holding relevance scores with dimensions (N, 1, H, W).
        """
        activations = list()

        # Run inference and collect activations.
        with torch.no_grad():
            # Replace image with ones avoids using image information for relevance computation.
            activations.append(torch.ones_like(x))
            for layer in self.layers:
                x = layer.forward(x)
                activations.append(x)

        # Reverse order of activations to run backwards through model
        activations = activations[::-1]
        activations_to_return = activations.copy()
        activations_to_return = [(layer.__class__.__name__, activations_to_return[i]) for i, layer in enumerate(self.lrp_layers)]
        activations = [a.data.requires_grad_(True) for a in activations]

        # Initial relevance scores are the network's output activations
        relevance = torch.softmax(activations.pop(0), dim=-1)  # Unsupervised

        # Perform relevance propagation
        relevances = [("Softmax", relevance)]
        for i, layer in enumerate(self.lrp_layers):
            relevance = layer.forward(activations.pop(0), relevance)

            # if layer.__class__.__name__ == "RelevancePropagationReLU":
            relevances.append((layer.__class__.__name__, relevance))

        return relevances[::-1], activations_to_return[::-1]


def min_subarray_with_sum_gt_target(arr, target):
    positive_relevancy_indices = torch.where(arr > 0)[0]

    values = np.vstack([positive_relevancy_indices.cpu().detach().numpy()[np.newaxis, ...], -arr[positive_relevancy_indices].cpu().detach().numpy()[np.newaxis, ...]])
    sorted_values = np.sort(values)
    indices = sorted_values[0, :]
    values = -1 * sorted_values[1, :]

    cs = np.cumsum(values)
    target_index = np.where(cs > target)[0]
    if len(target_index) == 0:
        return torch.tensor(sorted(indices[:int(len(indices)*0.1)].tolist()), dtype=torch.long)

    target_index = target_index[0]

    if target_index != 0:
        selected_indices = indices[:target_index]
    else:
        selected_indices = [indices[0]]

    selected_indices = torch.tensor(sorted(selected_indices), dtype=torch.long)

    return selected_indices

def jaccard_sim(list1, list2):
    """Define Jaccard Similarity function for two sets"""
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

def get_best_parameters(data, PCA_n_components, Birch_thresholds, Birch_n_clusters, batch_size):
    # print("len(data)", len(data), data[0].shape)
    results = []
    for n_components in PCA_n_components:
        for threshold in Birch_thresholds:
            for n_clusters in Birch_n_clusters:
                # print(n_components)
                clustering = Pipeline([
                    ('dim_red', IncrementalPCA(n_components=n_components, batch_size=batch_size)), 
                    ('clustering', Birch(threshold=threshold, n_clusters=n_clusters))
                    ])
                cluster_labels = clustering.fit_predict(data)
                if len(np.unique(cluster_labels)) < 2: continue
                silhouette_avg = silhouette_score(data, cluster_labels)

                results.append((threshold, n_clusters, n_components, silhouette_avg))
        
    optimal_threshold, optimal_n_clusters, optimal_n_components, _ = max(results, key=lambda item: item[3])

    return optimal_threshold, optimal_n_clusters, optimal_n_components

def get_tarantula_score(path_spectrum):
    a_f_ratio = path_spectrum['A_F'] / (path_spectrum['A_F'] + path_spectrum['I_F'])
    a_p_ratio = path_spectrum['A_P'] / (path_spectrum['A_P'] + path_spectrum['I_P'])
    return a_f_ratio / (a_f_ratio + a_p_ratio)

def get_ochiai_score(path_spectrum):
    total_faileds =  path_spectrum['A_F'] + path_spectrum['I_F']
    total_actives = path_spectrum['A_P'] + path_spectrum['A_F']
    return path_spectrum['A_F'] / (np.sqrt(total_faileds * total_actives))

def get_BARINEL_score(path_spectrum):
    return 1 - path_spectrum["A_P"] / (path_spectrum["A_P"] + path_spectrum["A_F"])
