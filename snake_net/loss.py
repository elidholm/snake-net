"""Loss functions for neural networks."""

import logging
from abc import ABC, abstractmethod

import numpy as np
from rich.logging import RichHandler

logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="%Y-%m-%d %H:%M:%S", handlers=[RichHandler()])
_log = logging.getLogger(__name__)


class Loss(ABC):
    def calculate(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        sample_losses = self.forward(predictions, labels)
        data_loss = np.mean(sample_losses)
        return data_loss

    @abstractmethod
    def forward(self, predictions, labels):
        """Forward pass for the loss function."""
        return


class LossCrossentropy(Loss):
    def forward(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        samples = len(predictions)
        predictions_clipped = np.clip(predictions, 1e-7, 1 - 1e-7)

        if len(labels.shape) < 2:
            correct_confidences = predictions_clipped[range(samples), labels]
        else:
            correct_confidences = np.sum(predictions_clipped * labels, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)

        return negative_log_likelihoods


def main():
    points, labels = Generator().spiral_data(n_points=100, n_classes=3)

    dense1 = LayerDense(2, 3)
    activation1 = ActivationReLU()

    dense2 = LayerDense(3, 3)
    activation2 = ActivationSoftmax()

    dense1.forward(points)
    activation1.forward(dense1.outputs)

    dense2.forward(activation1.outputs)
    activation2.forward(dense2.outputs)

    _log.info("Output predictions:\n%s", activation2.outputs[:5])

    loss_function = LossCrossentropy()
    loss = loss_function.calculate(activation2.outputs, labels)

    _log.info("Loss: %s", loss)


if __name__ == "__main__":
    from activation import (  # pylint: disable=import-error
        ActivationReLU,
        ActivationSoftmax,
    )
    from generator import Generator  # pylint: disable=import-error
    from layer import LayerDense  # pylint: disable=import-error

    np.random.seed(0)

    main()
