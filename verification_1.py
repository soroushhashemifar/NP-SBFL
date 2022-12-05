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
from main import generate_cdp_representation
from utils import myLRPModel, get_tarantula_score, get_ochiai_score, get_Dstar_score, get_BARINEL_score
from train_model import Net


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

lrp_model = myLRPModel(model, layers_structure)

layer_shapes = [128, 32, 10]

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])),
    batch_size=1, shuffle=True)

with open('./decision_kmeans.pickle', 'rb') as handle2, open('./cdp_spectrums.pickle', 'rb') as handle3:
    decision_kmeans = pickle.load(handle2)
    cdp_spectrums = pickle.load(handle3)

available_transformations = [
    transforms.RandomRotation(120),
    transforms.RandomCrop((28, 28)),
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

# tarantula = 0.65, 0.74, 0.75, 0.76, 0.769, 0.79, 0.83, 0.84, 0.849, 0.87, 0.94, 0.948, 0.97, 0.978, 0.98, 0.989, 0.99, 0.997, 0.998, 0.999
# ochiai = 0.02, 0.03, 0.0367, 0.05, 0.06, 0.068, 0.07, 0.077, 0.08, 0.085, 0.09, 0.099, 0.13, 0.18, 0.24, 0.26, 0.27, 0.31, 0.35, 0.43

SFL_strategy = args['SFL_strategy']
with open(f"results_{SFL_strategy}.txt", "a") as file:
    for threshold in [0.002, 0.008, 0.0084, 0.01, 0.011, 0.012, 0.013, 0.0135, 0.0138, 0.018, 0.025, 0.03, 0.05, 0.1, 0.16, 0.18, 0.19, 0.6, 0.8, 0.999]:
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

            cdp_representation, critical_neurons_layers_test, predicted_class, _ = generate_cdp_representation(lrp_model, data.view(-1, 784), layer_shapes)
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

                    cdp_representation_, critical_neurons_layers_test_, predicted_class_, _ = generate_cdp_representation(lrp_model, data_.view(-1, 784), layer_shapes)
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

            cdp_representation, critical_neurons_layers_test, predicted_class, _ = generate_cdp_representation(lrp_model, data.view(-1, 784), layer_shapes)
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

                    cdp_representation_, critical_neurons_layers_test_, predicted_class_, _ = generate_cdp_representation(lrp_model, data_.view(-1, 784), layer_shapes)
                    if (cdp_representation_ is None and critical_neurons_layers_test_ is None and predicted_class_ is None) or predicted_class_ not in decision_kmeans.keys():
                        continue

                    predicted_cluster_ = decision_kmeans[predicted_class_].predict(cdp_representation_[None, ...])[0]
                
                if OUT_OF_TIME:
                    pass
                else:
                    num_failures += 1

        rate = num_failures / num_total_samples if num_total_samples != 0 else 0
        file.write(f"FROM healty CDP with correct prediction TO faulty CDP with wrong prediction: {num_failures} {num_total_samples} {rate}\n\n")