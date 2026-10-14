import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
# https://cs231n.github.io/neural-networks-case-study/
def create_data(points, classes):
    N = points
    K = classes
    D = 2
    X = np.zeros((N*K,D)) # data matrix (each row = single example)
    y = np.zeros(N*K, dtype='uint8') # class labels
    for j in range(K):
        ix = range(N*j,N*(j+1))
        r = np.linspace(0.0,1,N) # radius
        t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = j
    return X, y

if __name__ == "__main__":
    X, y = create_data(100, 3)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg', s=20)
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Spiral Data")
    plt.show()