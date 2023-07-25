import pickle
import numpy as np
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
import torch


synth_dataset_path = "./pickles/synth_v2"

print(f"model_name \t SFL_strategy \t k \t num_samples \t mean L1 \t mean L2 \t mean L_inf \t IS natural (mean) \t IS natural (std) \t FID natural")
for model_name in ["Model_mnist_1", "Model_mnist_2", "Model_mnist_3"]:
    for SFL_strategy in ["tarantula", "ochiai", "barinel"]: 
        for suspiciousness_threshold in [1, 5, 10]:
            with open(f"{synth_dataset_path}/synthesized_dataset_{model_name}_{SFL_strategy}_k{suspiciousness_threshold}.pickle", 'rb') as handle:
                synthesized_dataset = pickle.load(handle)

            mean_l1 = []
            mean_l2 = []
            mean_l_inf = []
            for triple in synthesized_dataset:
                image, perturbed_image, _ = triple
                image *= 255
                perturbed_image *= 255
                delta = (image - perturbed_image).ravel()
                l1 = np.linalg.norm(delta, ord=1)
                mean_l1.append(l1)
                l2 = np.linalg.norm(delta, ord=2)
                mean_l2.append(l2)
                l_inf = np.linalg.norm(delta, ord=np.inf)
                mean_l_inf.append(l_inf)

            mean_l1 = np.mean(mean_l1)
            mean_l1 = "{:.3f}".format(mean_l1)
            mean_l2 = np.mean(mean_l2)
            mean_l2 = "{:.3f}".format(mean_l2)
            mean_l_inf = np.mean(mean_l_inf)
            mean_l_inf = "{:.3f}".format(mean_l_inf)

            print(f"{model_name} \t {SFL_strategy} \t {suspiciousness_threshold} \t {len(synthesized_dataset)} \t {mean_l1} \t {mean_l2} \t {mean_l_inf} \t - \t - \t -")

for model_name in ["Model_cifar_1", "Model_cifar_2", "Model_cifar_3"]:
    for SFL_strategy in ["tarantula", "ochiai", "barinel"]: 
        for suspiciousness_threshold in [10, 30, 50]:
            with open(f"{synth_dataset_path}/synthesized_dataset_{model_name}_{SFL_strategy}_k{suspiciousness_threshold}.pickle", 'rb') as handle:
                synthesized_dataset = pickle.load(handle)

            mean_l1 = []
            mean_l2 = []
            mean_l_inf = []
            for triple in synthesized_dataset:
                image, perturbed_image, _ = triple
                image *= 255
                perturbed_image *= 255
                delta = (image - perturbed_image).ravel()
                l1 = np.linalg.norm(delta, ord=1)
                mean_l1.append(l1)
                l2 = np.linalg.norm(delta, ord=2)
                mean_l2.append(l2)
                l_inf = np.linalg.norm(delta, ord=np.inf)
                mean_l_inf.append(l_inf)

            mean_l1 = np.mean(mean_l1)
            mean_l1 = "{:.3f}".format(mean_l1)
            mean_l2 = np.mean(mean_l2)
            mean_l2 = "{:.3f}".format(mean_l2)
            mean_l_inf = np.mean(mean_l_inf)
            mean_l_inf = "{:.3f}".format(mean_l_inf)

            metric_IS = InceptionScore(feature=64, normalize=True)
            metric_FID = FrechetInceptionDistance(feature=64, normalize=True)

            perturbed_images = torch.concat([torch.tensor(triple[1]).permute(2, 0 ,1).unsqueeze(0) for triple in synthesized_dataset])
            for i in range(0, perturbed_images.shape[0], 500):
                metric_IS.update(perturbed_images[i:i+500])
            
            is_1, is_2 = metric_IS.compute()
            is_1 = "{:.3f}".format(is_1.item())
            is_2 = "{:.3f}".format(is_2.item())

            real_images = torch.concat([torch.tensor(triple[0]).permute(2, 0 ,1).unsqueeze(0) for triple in synthesized_dataset])
            for i in range(0, real_images.shape[0], 500):
                metric_FID.update(real_images[i:i+500], real=True)

            for i in range(0, perturbed_images.shape[0], 500):
                metric_FID.update(perturbed_images[i:i+500], real=False)

            fid = metric_FID.compute()
            fid = "{:.3f}".format(fid.item())

            print(f"{model_name} \t {SFL_strategy} \t {suspiciousness_threshold} \t {len(synthesized_dataset)} \t {mean_l1} \t {mean_l2} \t {mean_l_inf} \t {is_1} \t {is_2} \t {fid}")
