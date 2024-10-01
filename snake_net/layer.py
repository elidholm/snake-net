"""Snake-Net Project."""

import logging

import numpy as np
from rich.logging import RichHandler

logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="%Y-%m-%d %H:%M:%S", handlers=[RichHandler()])
_log = logging.getLogger(__name__)


class LayerDense:
    """Dense layer of neurons."""

    def __init__(self, n_neurons: int, n_inputs: int):
        self.weights = 0.50 * np.random.randn(n_neurons, n_inputs)
        self.biases = 0.50 * np.random.randn(n_neurons, 1)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward propagation
        Args:
            inputs (np.ndarray): Input data
        Returns:
            np.ndarray: Output data
        """
        return self.weights.dot(inputs) + self.biases

    def update_params(self, d_w: np.ndarray, d_b: np.ndarray, alpha: float = 0.01) -> None:
        """Update weights and biases.
        Args:
            d_w (np.ndarray): Gradient of the weights
            d_b (np.ndarray): Gradient of the biases
            alpha (float, optional): Learning rate. Defaults to 0.01.
        """
        self.weights = self.weights - alpha * d_w
        self.biases = self.biases - alpha * d_b


def main():
    """Main function."""
    inputs = np.array([[1, 2, -1.5], [2, 5, 2.7], [3, -1, 3.3], [2.5, 2, -0.8]])
    _log.info("Inputs:\n%s", inputs)

    layer1 = LayerDense(5, 4)
    layer2 = LayerDense(2, 5)

    outputs1 = layer1.forward(inputs)
    _log.info("Output after first layer:\n%s", outputs1)
    outputs2 = layer2.forward(outputs1)
    _log.info("Output after second layer:\n%s", outputs2)


if __name__ == "__main__":
    np.random.seed(0)
    main()
