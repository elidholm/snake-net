"""Activation functions for neural networks."""

import logging

import numpy as np
from rich.logging import RichHandler

logging.basicConfig(level=logging.DEBUG, format="%(message)s", datefmt="%Y-%m-%d %H:%M:%S", handlers=[RichHandler()])
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


def main():
    """Main function."""
    inputs = np.array([-1, 2, -3, 4])
    logging.info("Inputs:\n%s", inputs)

    activation = ActivationReLU()
    activation.forward(inputs)
    _log.info("Outputs after ReLU activation function:\n%s", activation.outputs)


if __name__ == "__main__":
    np.random.seed(0)
    main()
