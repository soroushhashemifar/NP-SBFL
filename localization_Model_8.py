import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from deepcp_base import DeepCP
from models.train_model_8 import Net, FER2013
from synthesize import SynthesizeV1, SynthesizeV2, evaluation
from utils import report_common_neurons_SFLs
from verification import SynthesizedsetVerification


class Model8(DeepCP):
    
    def get_relevancy_and_activations(self, data):
        relevancies, activations = self.lrp_model.forward(data)

        predicted_class = torch.argmax(torch.softmax(activations[-1][1], 1)).item()

        relevancy = [(r[0], r[1].flatten(1)) for r in relevancies]
        activations = [(a[0], a[1].flatten(1)) for a in activations[:-1]]

        relevancy = list(filter(lambda item: item[0] in ["RelevancePropagationConv2d", "RelevancePropagationMaxPool2d", "RelevancePropagationLinear"], relevancy))
        relevancy = list(map(lambda item: item[1], relevancy))
        activations = list(filter(lambda item: item[0] in ["RelevancePropagationConv2d", "RelevancePropagationMaxPool2d", "RelevancePropagationLinear"], activations))
        activations = list(map(lambda item: F.relu(item[1]) if item[0] in ["RelevancePropagationConv2d", "RelevancePropagationLinear"] else item[1], activations))

        g_fx = torch.sum(relevancies[0][1]).item()

        return relevancy, activations, g_fx, predicted_class


if __name__ == "__main__":
    train_loader = torch.utils.data.DataLoader(
        FER2013('models/data/fer2013/fer2013.csv', split="Training",
                    transform=transforms.Compose([
                        transforms.Resize((48, 48)),
                        transforms.ToTensor(),
                    ])),
        batch_size=1, shuffle=False)

    test_loader = torch.utils.data.DataLoader(
        FER2013('models/data/fer2013/fer2013.csv', split="PublicTest", transform=transforms.Compose([
                        transforms.Resize((48, 48)),
                        transforms.ToTensor(),
                    ])),
        batch_size=1, shuffle=False)

    model = Net()
    model.load_state_dict(torch.load("models/Model_fer2013_1.pth", map_location="cpu"))
    model = model.to("cpu")
    model.eval()

    layers_structure = [
        model.conv1_1, torch.nn.ReLU(),
        model.conv1_2, torch.nn.ReLU(), 
        model.pool, 
        model.conv2_1, torch.nn.ReLU(), 
        model.conv2_2, torch.nn.ReLU(), 
        model.pool, 
        torch.nn.Flatten(1), 
        model.fc1, torch.nn.ReLU(), 
        model.fc2, torch.nn.ReLU(), 
        model.out 
    ]

    deepcp8 = Model8(
        model_name="Model_fer2013_1",
        model=model,
        layers_structure=layers_structure, 
        input_size=(1, 48, 48), 
        train_loader=train_loader, 
        device="cpu",
        alpha=0.7, 
        activation_threshold=0.
    )

    # model_8_synthsizer = SynthesizeV1(deepcp8.model_name, model, test_loader, pickles_path=deepcp8.path_to_save_pickles, output_path=os.path.join(deepcp8.path_to_save_pickles, "synth_v1"), step_size=10, distance=0.1)
    model_8_synthsizer = SynthesizeV2(deepcp8.model_name, model, test_loader, pickles_path=deepcp8.path_to_save_pickles, output_path=os.path.join(deepcp8.path_to_save_pickles, "synth_v2"), num_iterations=5, learning_rate=0.02)
    verification = SynthesizedsetVerification(deepcp8)

    suspiciousness_threshold = 50

    print("Localizing faults in model 8")
    deepcp8.run()

    report_common_neurons_SFLs(model_8_synthsizer, suspiciousness_threshold, 8)

    print("Synthesizing dataset for model 8")
    parameters = {
            "cuda": False,
            "loss": nn.CrossEntropyLoss(),
        }
    model_8_synthsizer.run("tarantula", suspiciousness_threshold=suspiciousness_threshold)
    evaluation(deepcp8.model_name, "tarantula", model, model_8_synthsizer.output_path, parameters, suspiciousness_threshold=suspiciousness_threshold)
    verification.verify("tarantula", suspiciousness_threshold=suspiciousness_threshold, synth_dataset_path=model_8_synthsizer.output_path)

    model_8_synthsizer.run("ochiai", suspiciousness_threshold=suspiciousness_threshold)
    evaluation(deepcp8.model_name, "ochiai", model, model_8_synthsizer.output_path, parameters, suspiciousness_threshold=suspiciousness_threshold)
    verification.verify("ochiai", suspiciousness_threshold=suspiciousness_threshold, synth_dataset_path=model_8_synthsizer.output_path)

    model_8_synthsizer.run("barinel", suspiciousness_threshold=suspiciousness_threshold)
    evaluation(deepcp8.model_name, "barinel", model, model_8_synthsizer.output_path, parameters, suspiciousness_threshold=suspiciousness_threshold)
    verification.verify("barinel", suspiciousness_threshold=suspiciousness_threshold, synth_dataset_path=model_8_synthsizer.output_path)
