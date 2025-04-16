from scipy.io import loadmat

import numpy as np
import pandas as pd
import os

mat_path = os.path.join(os.path.dirname(__file__), "mnist-original.mat")
mat = loadmat(mat_path)

images = mat["data"].T
labels = mat["label"][0]

full_data = np.column_stack((labels, images))

df = pd.DataFrame(full_data)

folder = os.path.dirname(__file__)
csv_path = os.path.join(folder, "mnist.csv")

df.to_csv(csv_path, index=False, header=False)

class Layer:
    def __init__(self, num_input, num_neuron, learning_rate = 0.1):
        self.weight = np.random.randn(num_neuron, num_input)
        self.bias = np.random.randn(num_neuron, 1)
        self.learning_rate = learning_rate        
