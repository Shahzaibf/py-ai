# modeling of a single nueron
import sys
import numpy as np
import matplotlib

# print("Python: ", sys.version)
# print("Numpy: ", np.__version__)
# print("Matplotlib: ", matplotlib.__version__)
# source venv/bin/activate

inputs = [1.2, 5.1, 2.1]
weights = [3.1, 2.1, 8.7]
bias = 3

# broken down neural networks just add inputs * weights + bias

output = inputs[0] * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2] + bias
print(output)