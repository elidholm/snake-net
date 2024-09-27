"""Activation functions for neural networks."""

import logging

import numpy as np
from rich.logging import RichHandler

logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="%Y-%m-%d %H:%M:%S", handlers=[RichHandler()])
_log = logging.getLogger(__name__)


class ActivationReLU:
    """Rectified Linear Unit activation function."""

    def __init__(
        self,
        outputs: np.ndarray = np.array([]),
    ) -> None:
        self.outputs = outputs

    def forward(self, inputs: np.ndarray):
        """Forward propagation
        Args:
            inputs (np.ndarray): Input data
        """
        self.outputs = np.maximum(0, inputs)


class ActivationSoftmax:
    """Softmax activation function."""

    def __init__(
        self,
        outputs: np.ndarray = np.array([]),
    ) -> None:
        self.outputs = outputs

    def forward(self, inputs: np.ndarray):
        """Forward propagation
        Args:
            inputs (np.ndarray): Input data
        """
        if np.array_equal(inputs, np.array([])):
            self.outputs = np.array([])
            return

        exp_values = np.exp(inputs - np.max(inputs, axis=-1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=-1, keepdims=True)
        self.outputs = probabilities


def main():
    """Main function."""
    inputs = np.array([[1, 2, 3, 4], [-2, 1, -2, 7]])
    logging.info("Inputs:\n%s", inputs)

    activation_relu = ActivationReLU()
    activation_relu.forward(inputs)
    _log.info("Outputs after ReLU activation function:\n%s", activation_relu.outputs)

    activation_softmax = ActivationSoftmax()
    activation_softmax.forward(inputs)
    _log.info("Outputs after Softmax activation function:\n%s", activation_softmax.outputs)


if __name__ == "__main__":
    np.random.seed(0)
    main()
