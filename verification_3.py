import pickle
import time
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import tqdm
from torch.autograd import Variable
from torchvision import datasets, transforms
import cv2
from PIL import Image

from config import args
from main_conv import generate_cdp_representation
from utils import myLRPModel, get_tarantula_score, get_ochiai_score, get_Dstar_score, get_BARINEL_score
from train_model_conv import Net


model = Net()
model.load_state_dict(torch.load("./mymodel_cifar10.pth"))
model = model.to("cuda" if args['cuda'] else "cpu")
model.eval()

layers_structure = [
        # [1, 3, 32, 32]
        model.conv1_1, torch.nn.ReLU(), model.pool, 
        model.conv2_1, torch.nn.ReLU(), model.pool, 
        model.conv3_1, torch.nn.ReLU(), model.pool,
        model.conv4_1, torch.nn.ReLU(), model.pool, 
        torch.nn.Flatten(1), 
        model.fc1, torch.nn.ReLU(), 
        model.fc2, torch.nn.ReLU(), 
        model.fc3 
    ]

lrp_model = myLRPModel(model, layers_structure)

relevancy, _ = lrp_model.forward(torch.randn((1, 3, 32, 32)))
layer_shapes = [list(r[0].shape)[0] for r in relevancy[1:]]

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])),
    batch_size=1, shuffle=True)

with open('./decision_kmeans_cifar10.pickle', 'rb') as handle2, open('./cdp_spectrums_cifar10.pickle', 'rb') as handle3:
    decision_kmeans = pickle.load(handle2)
    cdp_spectrums = pickle.load(handle3)

available_transformations = [
    transforms.RandomRotation(120),
    transforms.RandomCrop((32, 32)),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomHorizontalFlip(0.5),
    transforms.Lambda(lambda x : x + torch.randn_like(x)),
    transforms.ColorJitter(brightness=0, contrast=0.4, saturation=0, hue=0),
]

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

# tarantula = 0.72, 0.78, 0.786, 0.80, 0.82, 0.84, 0.85, 0.86, 0.866, 0.87, 0.875, 0.876, 0.88, 0.89, 0.90, 0.91, 0.917, 0.93, 0.935, 0.96
# ochiai = 0.12, 0.13, 0.136, 0.14, 0.18, 0.19, 0.22, 0.23, 0.236, 0.24, 0.26, 0.32, 0.326, 0.34, 0.346, 0.35, 0.37, 0.372, 0.38, 0.44

SFL_strategy = args['SFL_strategy']
with open(f"results_cifar_{SFL_strategy}.txt", "a") as file:
    for threshold in [0.08, 0.09, 0.10, 0.105, 0.12, 0.15, 0.18, 0.19, 0.193, 0.2, 0.26, 0.28, 0.288, 0.3, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4]:
        print("THRESHOLD ==", threshold)
        file.write(f"THRESHOLD == {threshold}\n")
        args['MIN_SCORE_TO_FILTER'] = threshold
        scores = list(filter(lambda item: item[1] > args['MIN_SCORE_TO_FILTER'], scores))
        faulty_cdps = list(map(lambda item: item[0], scores))

        num_corrections = 0
        num_total_samples = 0
        for data, target in tqdm.tqdm(test_loader):
            if args['cuda']:
                data, target = data.cuda(), target.cuda()

            data, target = Variable(data), Variable(target)

            cdp_representation, critical_neurons_layers_test, predicted_class, _ = generate_cdp_representation(lrp_model, data, layer_shapes)
            if (cdp_representation is None and critical_neurons_layers_test is None and predicted_class is None) or predicted_class not in decision_kmeans.keys():
                continue
                
            predicted_cluster = decision_kmeans[predicted_class].predict(cdp_representation[None, ...])[0]

            if (predicted_class, predicted_cluster) in faulty_cdps and target != predicted_class:
                num_total_samples += 1

                predicted_class_ = -1
                predicted_cluster_ = -1
                start_time = time.time()
                OUT_OF_TIME = False
                while (predicted_class_, predicted_cluster_) in faulty_cdps or predicted_class_ != target:
                    if time.time() - start_time >= args['TIME_BUDGET']:
                        OUT_OF_TIME = True
                        break

                    transformator = transforms.Compose(random.sample(available_transformations, 3))
                    data_ = transformator(data)

                    cdp_representation_, critical_neurons_layers_test_, predicted_class_, _ = generate_cdp_representation(lrp_model, data_, layer_shapes)
                    if (cdp_representation_ is None and critical_neurons_layers_test_ is None and predicted_class_ is None) or predicted_class_ not in decision_kmeans.keys():
                        continue

                    predicted_cluster_ = decision_kmeans[predicted_class_].predict(cdp_representation_[None, ...])[0]
                
                if OUT_OF_TIME:
                    pass
                else:
                    num_corrections += 1

        rate = num_corrections / num_total_samples if num_total_samples != 0 else 0
        file.write(f"FROM faulty CDP with wrong prediction TO healty CDP with correct prediction: {num_corrections} {num_total_samples} {rate}\n")

        num_failures = 0
        num_total_samples = 0
        for data, target in tqdm.tqdm(test_loader):
            if args['cuda']:
                data, target = data.cuda(), target.cuda()

            data, target = Variable(data), Variable(target)

            cdp_representation, critical_neurons_layers_test, predicted_class, _ = generate_cdp_representation(lrp_model, data, layer_shapes)
            if (cdp_representation is None and critical_neurons_layers_test is None and predicted_class is None) or predicted_class not in decision_kmeans.keys():
                continue

            predicted_cluster = decision_kmeans[predicted_class].predict(cdp_representation[None, ...])[0]

            if (predicted_class, predicted_cluster) not in faulty_cdps and target == predicted_class:
                num_total_samples += 1

                predicted_class_ = -1
                predicted_cluster_ = -1
                start_time = time.time()
                OUT_OF_TIME = False
                while (predicted_class_, predicted_cluster_) not in faulty_cdps or predicted_class_ == target:
                    if time.time() - start_time >= args['TIME_BUDGET']:
                        OUT_OF_TIME = True
                        break

                    transformator = transforms.Compose(random.sample(available_transformations, 3))
                    data_ = transformator(data)

                    cdp_representation_, critical_neurons_layers_test_, predicted_class_, _ = generate_cdp_representation(lrp_model, data_, layer_shapes)
                    if (cdp_representation_ is None and critical_neurons_layers_test_ is None and predicted_class_ is None) or predicted_class_ not in decision_kmeans.keys():
                        continue

                    predicted_cluster_ = decision_kmeans[predicted_class_].predict(cdp_representation_[None, ...])[0]
                
                if OUT_OF_TIME:
                    pass
                else:
                    num_failures += 1

        rate = num_failures / num_total_samples if num_total_samples != 0 else 0
        file.write(f"FROM healty CDP with correct prediction TO faulty CDP with wrong prediction: {num_failures} {num_total_samples} {rate}\n\n")
