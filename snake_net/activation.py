"""Activation functions for neural networks."""

import numpy as np
import rich


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
    inputs = np.array([[-1, 2, -3, 4]])
    rich.print(f"Inputs:\n{inputs}")


if __name__ == "__main__":
    np.random.seed(0)
    main()
