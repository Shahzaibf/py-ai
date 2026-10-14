import math
import numpy as np

# layer_outputs = [4.8, 1.21, 2.385]

E = math.e
# exponentiate to convert neg to pos without losing the meaning of the neg value
'''
exp_values = []
for output in layer_outputs:
    exp_values.append(E**output)

print(exp_values)

# normalized exponentiated values
norm_base = sum(exp_values)
norm_values = []
for value in exp_values:
    norm_values.append(value / norm_base)

print(norm_values)
print(sum(norm_values))
'''

# with numpy
# overflow subtract max value from each so each value can only be max 1 after exponentiating
layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.026]]

exp_values = np.exp(layer_outputs)

norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)
print(norm_values)
# input -> Exponentiate -> Normalize -> Output (Softmax activation)