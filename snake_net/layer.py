"""Snake-Net Project."""

import numpy as np


class Layer_Dense:
    """Dense layer of neurons."""

    def __init__(self, n_inputs: int, n_neurons: int):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


def main():
    input = np.array([[1, 2, 3, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]])

    layer1 = Layer_Dense(4, 5)
    layer2 = Layer_Dense(5, 2)

    layer1.forward(input)
    print(layer1.output)
    layer2.forward(layer1.output)
    print(layer2.output)


if __name__ == "__main__":
    np.random.seed(0)
    main()
