import enum
import pickle
import matplotlib.pyplot as plt
import random
import cv2
from PIL import Image


def plot_sample_synthesized_images(k, models, SFLs):
    for model_name in models:
        for SFL_strategy in SFLs:
            with open(f"../pickles/synth_v2/synthesized_dataset_{model_name}_{SFL_strategy}_k{k}.pickle", 'rb') as handle:
                synthesized_dataset = pickle.load(handle)

            random.shuffle(synthesized_dataset)

            fig, ax = plt.subplots(2, 5, figsize=(14, 6))
            fig.tight_layout()
            for i in range(5):
                for j in range(2):
                    img = synthesized_dataset[i][j]
                    img = cv2.resize(img, (128, 128))
                    ax[j, i].imshow(img)
                    ax[j, i].axis('off')

            plt.subplots_adjust(wspace=0, hspace=0)
            plt.savefig(f"./figures/synthesized_samples_{model_name}_{SFL_strategy}.png")

            if "mnist" in model_name:
                image = Image.open(f"./figures/synthesized_samples_{model_name}_{SFL_strategy}.png").convert("L")
                image.save(f"./figures/synthesized_samples_{model_name}_{SFL_strategy}.png")


models = ["Model_mnist_1", "Model_mnist_2", "Model_mnist_3"]
SFLs = ["tarantula", "ochiai", "barinel"]
k = 10
plot_sample_synthesized_images(k, models, SFLs)

models = ["Model_cifar_1", "Model_cifar_2", "Model_cifar_3"]
SFLs = ["tarantula", "ochiai", "barinel"]
k = 50
plot_sample_synthesized_images(k, models, SFLs)
