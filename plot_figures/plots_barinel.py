import numpy as np
import matplotlib.pyplot as plt
import re


def get_data(filename):
    with open(filename) as file:
        content = file.readlines()
        content = list(map(lambda item: item.strip(), content))
        content = list(filter(lambda item: len(item) != 0, content))

    data = []

    i = 0
    while i < len(content):
        if "THRESHOLD" in content[i]:
            threshold = content[i].split(" ")[-1]
            FHR = content[i+1].split(" ")[-1]
            HFR = content[i+2].split(" ")[-1]
            data.append([float(threshold), float(FHR), float(HFR)])
            i += 2

        i += 1

    data = np.array(data)

    return data

fig, ax = plt.subplots(1, 3, figsize=(16, 5))
fig.tight_layout(pad=5.0)

results_barinel = get_data("../results/Model_1_barinel_verif.txt")
harmonic_mean = 2/(1/(results_barinel[:, 1]+1e-8) + 1/(results_barinel[:, 2]+1e-8))
max_idx = np.argmax(harmonic_mean)

print("Model 1", results_barinel[max_idx], harmonic_mean[max_idx])

ax[0].plot(results_barinel[:, 0], results_barinel[:, 1], label="FHR")
ax[0].plot(results_barinel[:, 0], results_barinel[:, 2], label="HFR")
ax[0].plot(results_barinel[:, 0], harmonic_mean, label="Harmonic mean")
ax[0].set_xlabel("Threshold")
ax[0].set_ylabel("Rate")
ax[0].legend()

results2_barinel = get_data("../results/Model_2_barinel_verif.txt")
harmonic_mean = 2/(1/(results2_barinel[:, 1]+1e-8) + 1/(results2_barinel[:, 2]+1e-8))
max_idx = np.argmax(harmonic_mean)

print("Model 2", results2_barinel[max_idx], harmonic_mean[max_idx])

ax[1].plot(results2_barinel[:, 0], results2_barinel[:, 1], label="FHR")
ax[1].plot(results2_barinel[:, 0], results2_barinel[:, 2], label="HFR")
ax[1].plot(results2_barinel[:, 0], harmonic_mean, label="Harmonic mean")
ax[1].set_xlabel("Threshold")
ax[1].set_ylabel("Rate")
ax[1].legend()

results_cifar_barinel = get_data("../results/Model_3_barinel_verif.txt")
harmonic_mean = 2/(1/(results_cifar_barinel[:, 1]+1e-8) + 1/(results_cifar_barinel[:, 2]+1e-8))
max_idx = np.argmax(harmonic_mean)

print("Model 3", results_cifar_barinel[max_idx], harmonic_mean[max_idx])

ax[2].plot(results_cifar_barinel[:, 0], results_cifar_barinel[:, 1], label="FHR")
ax[2].plot(results_cifar_barinel[:, 0], results_cifar_barinel[:, 2], label="HFR")
ax[2].plot(results_cifar_barinel[:, 0], harmonic_mean, label="Harmonic mean")
ax[2].set_xlabel("Threshold")
ax[2].set_ylabel("Rate")
ax[2].legend()

plt.savefig('figures/barinel_plots.png')