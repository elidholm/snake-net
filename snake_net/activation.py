"""Activation functions for neural networks."""

import logging
from abc import ABC, abstractmethod

import numpy as np
from rich.logging import RichHandler

logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="%Y-%m-%d %H:%M:%S", handlers=[RichHandler()])
_log = logging.getLogger(__name__)


class Activation(ABC):

    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward propagation
        Args:
            inputs (np.ndarray): Input data
        Returns:
            np.ndarray: Output data
        """
        return np.array([])


class ActivationReLU(Activation):
    """Rectified Linear Unit activation function."""

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return np.maximum(0, inputs)


class ActivationSoftmax(Activation):
    """Softmax activation function."""

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        exp_values = np.exp(inputs - np.max(inputs, axis=0, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=0, keepdims=True)
        return probabilities


def main():
    """Main function."""
    inputs = np.array([[1, 2, 3, 4], [-2, 1, -2, 7]])
    logging.info("Inputs:\n%s", inputs)

    activation_relu = ActivationReLU()
    relu_outputs = activation_relu.forward(inputs)
    _log.info("Outputs after ReLU activation function:\n%s", relu_outputs)

    activation_softmax = ActivationSoftmax()
    softmax_outputs = activation_softmax.forward(inputs)
    _log.info("Outputs after Softmax activation function:\n%s", softmax_outputs)


if __name__ == "__main__":
    np.random.seed(0)
    main()
