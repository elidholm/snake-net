"""Snake-Net Project."""

import numpy as np
import rich


class LayerDense:
    """Dense layer of neurons."""

    def __init__(self, n_inputs: int, n_neurons: int):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.output = np.zeros((1, n_neurons))

    def forward(self, inputs: np.ndarray) -> None:
        """Forward propagation
        Args:
            inputs (np.ndarray): Input data
        """
        self.output = np.dot(inputs, self.weights) + self.biases


def main():
    """Main function."""
    inputs = np.array([[1, 2, 3, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]])
    rich.print(f"Inputs:\n{inputs}")

    layer1 = LayerDense(4, 5)
    layer2 = LayerDense(5, 2)

    layer1.forward(inputs)
    rich.print(f"\nOutput after first layer:\n{layer1.output}")
    layer2.forward(layer1.output)
    rich.print(f"\nOutput after second layer:\n{layer2.output}")


if __name__ == "__main__":
    np.random.seed(0)
    main()
