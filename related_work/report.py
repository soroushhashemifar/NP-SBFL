import sys
sys.path.insert(0, "..")
import os
import pickle
import numpy as np

from synthesize import SynthesizeV2
from deepfault_base import Synthesize_DF


def report_common_neurons(model_name, synthesizer, synthesizer_deepfault, suspiciousness_threshold, num_susp_neurons, num_layers):
    inters = []
    suspicious_neurons = synthesizer.get_suspicious_neurons(SFL_strategy, suspiciousness_threshold)
    suspicious_neurons_deepfault = synthesizer_deepfault.get_suspicious_neurons(SFL_strategy_deepfault, num_susp_neurons)
    for layer in range(num_layers):
        suspicious_neurons_ = suspicious_neurons[layer]
        suspicious_neurons_ = list(map(lambda item: item[0], suspicious_neurons_))
        suspicious_neurons_deepfault_ = suspicious_neurons_deepfault[layer]
        
        if len(suspicious_neurons_deepfault_) == 0:
            continue
        
        suspicious_neurons_deepfault_ = list(map(lambda item: item[0], suspicious_neurons_deepfault_))
        inter = len(set(suspicious_neurons_).intersection(suspicious_neurons_deepfault_)) / suspiciousness_threshold
        inters.append(inter)

    print(f"#common neurons ({model_name}):", np.median(inters), np.mean(inters))

pickles_path = "../pickles"
deepfault_pickles_path = "./pickles_deepfault"

if __name__ == "__main__":
    model_name = "Model_mnist_1"
    num_layers = 5

    SFL_strategy = "tarantula"
    suspiciousness_threshold = 10
    SFL_strategy_deepfault = "ochiai"
    num_susp_neurons = 10 * num_layers

    model_synthsizer = SynthesizeV2(model_name, None, None, pickles_path=pickles_path)
    model_synthsizer_deepfault = Synthesize_DF(model_name, None, None, pickles_path=deepfault_pickles_path)

    report_common_neurons(model_name, model_synthsizer, model_synthsizer_deepfault, suspiciousness_threshold, num_susp_neurons, num_layers)
    
    model_name = "Model_mnist_2"
    num_layers = 6

    SFL_strategy = "tarantula"
    suspiciousness_threshold = 10
    SFL_strategy_deepfault = "ochiai"
    num_susp_neurons = 10 * num_layers

    model_synthsizer = SynthesizeV2(model_name, None, None, pickles_path=pickles_path)
    model_synthsizer_deepfault = Synthesize_DF(model_name, None, None, pickles_path=deepfault_pickles_path)

    report_common_neurons(model_name, model_synthsizer, model_synthsizer_deepfault, suspiciousness_threshold, num_susp_neurons, num_layers)

    model_name = "Model_mnist_3"
    num_layers = 8

    SFL_strategy = "tarantula"
    suspiciousness_threshold = 10
    SFL_strategy_deepfault = "ochiai"
    num_susp_neurons = 10 * num_layers

    model_synthsizer = SynthesizeV2(model_name, None, None, pickles_path=pickles_path)
    model_synthsizer_deepfault = Synthesize_DF(model_name, None, None, pickles_path=deepfault_pickles_path)

    report_common_neurons(model_name, model_synthsizer, model_synthsizer_deepfault, suspiciousness_threshold, num_susp_neurons, num_layers)

    model_name = "Model_cifar_1"
    num_layers = 10

    SFL_strategy = "tarantula"
    suspiciousness_threshold = 50
    SFL_strategy_deepfault = "ochiai"
    num_susp_neurons = 50 * num_layers

    model_synthsizer = SynthesizeV2(model_name, None, None, pickles_path=pickles_path)
    model_synthsizer_deepfault = Synthesize_DF(model_name, None, None, pickles_path=deepfault_pickles_path)

    report_common_neurons(model_name, model_synthsizer, model_synthsizer_deepfault, suspiciousness_threshold, num_susp_neurons, num_layers)

    model_name = "Model_cifar_2"
    num_layers = 8

    SFL_strategy = "tarantula"
    suspiciousness_threshold = 50
    SFL_strategy_deepfault = "ochiai"
    num_susp_neurons = 50 * num_layers

    model_synthsizer = SynthesizeV2(model_name, None, None, pickles_path=pickles_path)
    model_synthsizer_deepfault = Synthesize_DF(model_name, None, None, pickles_path=deepfault_pickles_path)

    report_common_neurons(model_name, model_synthsizer, model_synthsizer_deepfault, suspiciousness_threshold, num_susp_neurons, num_layers)

    model_name = "Model_cifar_3"
    num_layers = 7

    SFL_strategy = "tarantula"
    suspiciousness_threshold = 50
    SFL_strategy_deepfault = "ochiai"
    num_susp_neurons = 50 * num_layers

    model_synthsizer = SynthesizeV2(model_name, None, None, pickles_path=pickles_path)
    model_synthsizer_deepfault = Synthesize_DF(model_name, None, None, pickles_path=deepfault_pickles_path)

    report_common_neurons(model_name, model_synthsizer, model_synthsizer_deepfault, suspiciousness_threshold, num_susp_neurons, num_layers)