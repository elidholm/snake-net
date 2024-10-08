"""ReLU activation function test."""

import numpy as np
import pytest

from snake_net.activation import ActivationReLU


class TestActivationReLU:

    @pytest.fixture(scope="function")
    def relu(self) -> ActivationReLU:
        return ActivationReLU()

    @pytest.fixture(scope="function")
    def empty_array(self) -> np.ndarray:
        return np.array([])

    @pytest.fixture(scope="function")
    def positive_array(self) -> np.ndarray:
        return np.array([1, 2, 3, 4])

    @pytest.fixture(scope="function")
    def negative_array(self) -> np.ndarray:
        return np.array([-1, -2, -3, -4])

    @pytest.fixture(scope="function")
    def mixed_array(self) -> np.ndarray:
        return np.array([-1, 2, -3, 4])

    def test_initial_outputs_should_be_empty(self, relu: ActivationReLU, empty_array: np.ndarray):
        """Check that outputs are initiated to empty array."""
        actual = relu.outputs
        expected = empty_array
        assert np.array_equal(actual, expected)

    def test_output_of_empty_input_should_be_empty(self, relu: ActivationReLU, empty_array: np.ndarray):
        """Check that output is empty if input is empty."""
        relu.forward(empty_array)
        actual = relu.outputs
        expected = empty_array
        assert np.array_equal(actual, expected)

    def test_output_of_positive_input_should_be_itself(self, relu: ActivationReLU, positive_array: np.ndarray):
        """Check that output is itself if input is positive."""
        relu.forward(positive_array)
        actual = relu.outputs
        expected = positive_array
        assert np.array_equal(actual, expected)

    def test_output_of_negative_input_should_be_zeros(self, relu: ActivationReLU, negative_array: np.ndarray):
        """Check that output is zeros if input is negative."""
        relu.forward(negative_array)
        actual = relu.outputs
        expected = np.array([0, 0, 0, 0])
        assert np.array_equal(actual, expected)

    def test_output_of_mixed_input_should_be_correct(self, relu: ActivationReLU, mixed_array: np.ndarray):
        """Check that output is correct if input is mixed."""
        relu.forward(mixed_array)
        actual = relu.outputs
        expected = np.array([0, 2, 0, 4])
        assert np.array_equal(actual, expected)
