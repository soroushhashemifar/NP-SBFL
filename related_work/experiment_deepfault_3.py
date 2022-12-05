import pickle
from config import args
from train_model_conv import Net
import torch
from torchvision import datasets, transforms
import tqdm
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

from utils import get_BARINEL_score, get_tarantula_score


if __name__ == "__main__":
    new_test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])),
        batch_size=1, shuffle=True)

    model = Net()
    model.load_state_dict(torch.load("./mymodel_cifar10.pth", map_location=torch.device('cpu')))
    model = model.to("cuda" if args['cuda'] else "cpu")
    model.eval()

    logits = []
    def hook(model, input, output):
        logits.append(output[0].cpu())

    model.conv1_1.register_forward_hook(hook)
    model.conv2_1.register_forward_hook(hook)
    model.conv3_1.register_forward_hook(hook)
    model.conv4_1.register_forward_hook(hook)
    model.fc1.register_forward_hook(hook)
    model.fc2.register_forward_hook(hook)
    model.fc3.register_forward_hook(hook)

    hit_spectrums = {}
    for data, target in tqdm.tqdm(new_test_loader):
        if args['cuda']:
            data, target = data.cpu(), target.cpu()

        data, target = Variable(data), Variable(target)

        outputs = model(data)
        outputs = torch.softmax(outputs, 1)
        outputs = torch.argmax(outputs, 1)
        test_pass = (outputs == target).item()

        for layer_index, layer_logits_values in enumerate(logits[:-1]):
            activations = F.relu(layer_logits_values)
            if len(activations.shape) == 3: # conv layers
                num_elements = activations.shape[1] * activations.shape[2]
                activations = torch.norm(activations, p='fro', dim=(1, 2))
                activations = activations / num_elements
                active_neurons = torch.where(activations > 0.5)[0]
            else: # fully connected layers
                active_neurons = torch.where(activations > 0.5)[0]

            for i in range(layer_logits_values.shape[0]):
                unit_name = f"layer_{layer_index}_unit_{i}"
                unit_spectrum = hit_spectrums.get(unit_name, {})

                if i in active_neurons and test_pass:
                    unit_spectrum["A_P"] = unit_spectrum.get("A_P", 0) + 1
                elif i in active_neurons and not test_pass:
                    unit_spectrum["A_F"] = unit_spectrum.get("A_F", 0) + 1
                elif i not in active_neurons and test_pass:
                    unit_spectrum["I_P"] = unit_spectrum.get("I_P", 0) + 1
                elif i not in active_neurons and not test_pass:
                    unit_spectrum["I_F"] = unit_spectrum.get("I_F", 0) + 1

                hit_spectrums[unit_name] = unit_spectrum

        # exit()

        logits = []

    scores = []
    for unit_name in hit_spectrums.keys():
        unit_spectrum = hit_spectrums[unit_name]
        score = get_tarantula_score({ # reference: deepfault paper
            "A_P": unit_spectrum.get("A_P", 0), 
            "I_P": unit_spectrum.get("I_P", 0), 
            "A_F": unit_spectrum.get("A_F", 0), 
            "I_F": unit_spectrum.get("I_F", 0), 
        })
        scores.append((unit_name, score))

    deepfault_scores = []
    for layer_index in range(6):
        layer_scores = list(filter(lambda item: item[0].startswith(f"layer_{layer_index}"), scores))
        layer_scores = sorted(layer_scores, key=lambda item: -item[1])
        layer_scores = layer_scores[:3]
        deepfault_scores.extend(layer_scores)

    with open('./cdp_spectrums_cifar10.pickle', 'rb') as handle:
        cdp_spectrums = pickle.load(handle)

    fpl_scores = []
    for key, path_spectrum in cdp_spectrums.items():
        score = get_tarantula_score(path_spectrum)
        fpl_scores.append((key, score))

    fpl_scores = list(filter(lambda item: item[1] > 0.91, fpl_scores))
    faulty_cdps = list(map(lambda item: item[0], fpl_scores))

    with open('./decision_graph_cifar10.pickle', 'rb') as handle:
        decision_graph = pickle.load(handle)

    deepfault_detected_neurons = [(int(score_tuple[0].split("_")[1]), int(score_tuple[0].split("_")[3])) for score_tuple in deepfault_scores]
    # deepfault_detected_neurons = list(filter(lambda item: item[0] in [4, 5], deepfault_detected_neurons))
    deepfault_detected_neurons = set(deepfault_detected_neurons)
    fpl_detected_neurons = [(layer_index, neuron_index) for faulty_cdp in faulty_cdps for layer_index in range(0, 6) for neuron_index in decision_graph[faulty_cdp][layer_index]]
    fpl_detected_neurons = set(fpl_detected_neurons)

    print(sorted(list(deepfault_detected_neurons)))
    print(sorted(list(fpl_detected_neurons)))

    print("common detected neurons ratio:", len(deepfault_detected_neurons.intersection(fpl_detected_neurons)) / len(deepfault_detected_neurons))
    print("DeepFault - FPL:", len(deepfault_detected_neurons - fpl_detected_neurons) / len(deepfault_detected_neurons))
    print("FPL - DeepFault:", len(fpl_detected_neurons - deepfault_detected_neurons) / len(fpl_detected_neurons))