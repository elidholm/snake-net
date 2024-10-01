"""Helper functions for the project."""

import logging

import numpy as np
from rich.logging import RichHandler

logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="%Y-%m-%d %H:%M:%S", handlers=[RichHandler()])
_log = logging.getLogger(__name__)


def relu_derivative(inputs: np.ndarray) -> np.ndarray:
    """Derivative of the ReLU activation function."""
    return inputs > 0


def one_hot_encode(labels: np.ndarray) -> np.ndarray:
    """One-hot encoding of the labels."""
    one_hot_labels = np.zeros((labels.size, labels.max() + 1))
    one_hot_labels[np.arange(labels.size), labels] = 1
    one_hot_labels = one_hot_labels.T
    return one_hot_labels


def main():
    inputs = np.array([[-1, 2, -3], [4, -5, 6]])
    _log.info("Inputs:\n%s", inputs)

    relu_derivative_values = relu_derivative(inputs)
    _log.info("ReLU derivative values:\n%s", relu_derivative_values)

    labels = np.array([0, 1, 2, 1, 0])
    _log.info("Labels: %s", labels)

    one_hot_labels = one_hot_encode(labels)
    _log.info("One-hot labels:\n%s", one_hot_labels)


if __name__ == "__main__":
    main()
