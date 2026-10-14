import numpy as np
from data.cd import create_data

np.random.seed(0)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# A linear activation function can match linear but not like sine
# use a ReLU activation function
# Rectified Linear activation function
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

# softmax activation --> mainp6.py (input -> Exponentiate -> Normalize -> Output)
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

# Defining Loss (p7, p8)
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        # clip off zeroes
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        # scalar values
        # EX: y_true = [0, 1, 1]
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        # one hot encoded vector
        # EX:
        #     y_true = [
        #         [1, 0, 0],
        #         [0, 1, 0],
        #         [0, 1, 0],
        #       ]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
        negative_log_likelihood = -np.log(correct_confidences)
        return negative_log_likelihood
X, y = create_data(points=100, classes=3)

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])

loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(activation2.output, y)
print("Loss: ", loss)