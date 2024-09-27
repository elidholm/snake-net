"""Loss function test."""

import numpy as np
import pytest

from snake_net.loss import LossCrossentropy

np.random.seed(0)


class TestLossCrossentropy:

    @pytest.fixture(scope="function")
    def cross_entropy_loss(self) -> LossCrossentropy:
        return LossCrossentropy()

    @pytest.fixture(scope="function")
    def tolerance(self) -> float:
        return 2e-7

    def test_loss_of_correct_prediction_should_be_zero(self, cross_entropy_loss: LossCrossentropy, tolerance: float):
        """Check that loss of correct prediction is zero."""
        actual = cross_entropy_loss.calculate(np.array([[0, 1, 0]]), np.array([1]))
        assert actual < tolerance

    def test_loss_of_incorrect_prediction_should_be_greater_than_zero(
        self, cross_entropy_loss: LossCrossentropy, tolerance: float
    ):
        """Check that loss of incorrect prediction is greater than zero."""
        actual = cross_entropy_loss.calculate(np.array([[0, 1, 0]]), np.array([0]))
        assert actual > tolerance
