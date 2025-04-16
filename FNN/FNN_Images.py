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

    def forward(self, input):
        self.input = input
        z = np.dot(self.weight, self.input) + self.bias
        self.z = z
        self.output = self.ReLu(z)
        return self.output

    def ReLu(self, input):
        return np.maximum(0, input)
    
    def ReLu_derivative(self, x):
        return (x > 0).astype(float)
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    
    def backpropagation(self, output):
        dc_dz = self.softmax(self.z) - output
        dc_db = dc_dz

        dc_dw = np.dot(dc_dz, self.input.T)

        self.weight -= self.learning_rate * dc_dw
        self.bias -= self.learning_rate * dc_db

        return dc_dz

ms_file = os.path.dirname(__file__)
mnist_csv = os.path.join(ms_file, "mnist.csv")

data = pd.read_csv(mnist_csv, header=None).values

X = data[:, 1:] / 255.0
y = data[:, 0].astype(int)


def one_hot(label, num_classes=10):
    vec = np.zeros((num_classes, 1))
    vec[label] = 1
    return vec

input_size = 784
hidden_size = 64
output_size = 10

hidden = Layer(input_size, hidden_size)
output = Layer(hidden_size, output_size)

iterations = 5

for iteration in range(iterations):
    total_loss = 0
    correct = 0

    for i in range(len(X)):
        x_sample = X[i].reshape(-1, 1)
        y_sample = one_hot(y[i])

        a1 = hidden.forward(x_sample)
        a2 = output.forward(a1)


        pred_label = np.argmax(a2)
        if pred_label == y[i]:
            correct += 1

        loss = -np.sum(y_sample * np.log(a2 + 1e-9))
        total_loss += loss

        dc_dz2 = output.backpropagation(y_sample)
        dc_dz1 = np.dot(output.weights.T, dc_dz2) * hidden.ReLu_derivative(hidden.z)
        hidden.backpropagation(dc_dz1)

    avg_loss = total_loss / len(X)
    accuracy = correct / len(X)
    print(f"Epoch {iteration+1} | Loss: {avg_loss:.4f} | Accuracy: {accuracy*100:.2f}%")
