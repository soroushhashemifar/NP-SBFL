import enum
import pickle
import matplotlib.pyplot as plt
import random
import cv2


for model_name in ["Model_cifar_1"]:
    for SFL_strategy in ["tarantula", "ochiai", "barinel"]:
        with open(f"../pickles/synthesized_dataset_{model_name}_{SFL_strategy}.pickle", 'rb') as handle:
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