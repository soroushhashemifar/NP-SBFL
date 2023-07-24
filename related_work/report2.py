import pickle
import numpy as np


synth_dataset_path = "./pickles_deepfault/synth_v1"

print(f"model_name \t SFL_strategy \t k \t num_samples \t mean L1 \t mean L2 \t mean L_inf")
for model_name, num_layers in [("Model_mnist_1", 5), ("Model_mnist_2", 6), ("Model_mnist_3", 8)]:
    for SFL_strategy in ["tarantula", "ochiai", "barinel"]: 
        for suspiciousness_threshold in [1, 5, 10]:
            with open(f"{synth_dataset_path}/synthesized_dataset_{model_name}_{SFL_strategy}_k{suspiciousness_threshold*num_layers}.pickle", 'rb') as handle:
                synthesized_dataset = pickle.load(handle)

            mean_l1 = []
            mean_l2 = []
            mean_l_inf = []
            for triple in synthesized_dataset:
                image, perturbed_image, _ = triple
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

            print(f"{model_name} \t {SFL_strategy} \t {suspiciousness_threshold} \t {len(synthesized_dataset)} \t {mean_l1} \t {mean_l2} \t {mean_l_inf}")

for model_name, num_layers in [("Model_cifar_1", 10), ("Model_cifar_2", 8), ("Model_cifar_3", 7)]:
    for SFL_strategy in ["tarantula", "ochiai", "barinel"]: 
        for suspiciousness_threshold in [10, 30, 50]:
            with open(f"{synth_dataset_path}/synthesized_dataset_{model_name}_{SFL_strategy}_k{suspiciousness_threshold*num_layers}.pickle", 'rb') as handle:
                synthesized_dataset = pickle.load(handle)

            mean_l1 = []
            mean_l2 = []
            mean_l_inf = []
            for triple in synthesized_dataset:
                image, perturbed_image, _ = triple
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

            print(f"{model_name} \t {SFL_strategy} \t {suspiciousness_threshold} \t {len(synthesized_dataset)} \t {mean_l1} \t {mean_l2} \t {mean_l_inf}")
