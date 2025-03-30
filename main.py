import random as rand
import numpy as np


class Layer:
    def __init__(self, num_input, num_neuron, learning_rate = 0.1):
        self.weight = np.random.randn(num_neuron, num_input)
        self.bias = np.random.randn(num_neuron, 1)
        self.learning_rate = learning_rate

    def forward(self, input):
        self.input = input.reshape(-1, 1)
        z = np.dot(self.weight, self.input) + self.bias
        self.output = self.sigmoid(z)
        return self.sigmoid(z)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_der(self, x):
        return x * (1 - x)
    
    def backpropagation(self, output):
        dout_dz = self.sigmoid_der(self.output)  
        dz_dw = self.input.T                    
        dz_db = 1                                
        dz_dinput = self.weight.T          

        dL_dz = output * dout_dz
        dL_dw = np.dot(dL_dz, dz_dw) 
        dL_db = dL_dz              

        self.weight -= self.learning_rate * dL_dw
        self.bias -= self.learning_rate * dL_db

        return np.dot(dz_dinput, dL_dz)

layer = Layer(3, 5)

input_data = np.array([1, 2, 3])
target_output = np.array([[0.5], [0.8], [0.1], [0.3], [0.9]])

for epoch in range(100000):

    predicted_output = layer.forward(input_data)

    error = predicted_output - target_output
    
    layer.backpropagation(error)

    if epoch % 100 == 0:
        loss = np.mean(np.square(error))
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

print("Final output after training:\n", layer.forward(input_data))

# print(f"The layer is: {layer}")
# input = np.array([[0.2], [0.5], [0.2]])
# output = layer.forward(input)
# print(f"output is: {output}")
