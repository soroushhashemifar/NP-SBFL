import torch
from torchvision import datasets, transforms

import lrp_src
from deepcp_class import DeepCP
from train_model_3 import Net
from utils import myLRPModel


class Model3(DeepCP):
    
    def get_relevancy_and_activations(self, data):
        relevancy, activations = lrp_model.forward(data)

        relevancy = [r.view(r.shape[0], r.shape[1], -1).sum(2) for r in relevancy]
        relevancy[8] = relevancy[8].view(-1, 64, 4).sum(2) # input of fc1, output of flatten!
        activations = [torch.norm(a, p='fro', dim=(2, 3)) if len(a.shape) == 4 else a for a in activations]

        return relevancy, activations


if __name__ == "__main__":
    ALPHA = 0.9

    lrp_src.lrp_layers.top_k_percent = ALPHA

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])),
        batch_size=1, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])),
        batch_size=1, shuffle=True)

    model = Net()
    model.load_state_dict(torch.load("./models/mymodel_3.pth", map_location="cpu"))
    model = model.to("cpu")
    model.eval()

    layers_structure = [
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

    model3 = Model3(
        model_name="Model_3",
        lrp_model=lrp_model, 
        layer_shapes=layer_shapes, 
        path_to_save_activations="temp", 
        path_to_save_representations="temp2", 
        train_loader=train_loader, 
        test_loader=test_loader,
        device="cpu",
        batch_size=128,
        alpha=ALPHA, beta=0.7, min_match=0.8, 
        PCA_n_components=[4],#[64, 128], 
        Birch_thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        Birch_n_clusters=[2, 5, 7, 10],
        SFL_strategy="tarantula",
    )

    model3.run()
