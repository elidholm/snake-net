"""Softmax activation function test."""

import numpy as np
import pytest

from snake_net.activation import ActivationSoftmax


class TestActivationSoftmax:

    @pytest.fixture(scope="function")
    def softmax(self) -> ActivationSoftmax:
        return ActivationSoftmax()

    @pytest.fixture(scope="function")
    def empty_array(self) -> np.ndarray:
        return np.array([])

    @pytest.fixture(scope="function")
    def one_dim_array(self) -> np.ndarray:
        return np.array([1, 2, 3, 4])

    @pytest.fixture(scope="function")
    def multi_dim_array(self) -> np.ndarray:
        return np.array([[1, 2, 3, 4], [-2, 1, -2, 7]])

    def test_initial_outputs_should_be_empty(self, softmax: ActivationSoftmax, empty_array: np.ndarray):
        """Check that outputs are initiated to empty array."""
        actual = softmax.outputs
        expected = empty_array
        assert np.array_equal(actual, expected)

    def test_output_of_empty_input_should_be_empty(self, softmax: ActivationSoftmax, empty_array: np.ndarray):
        """Check that output is empty if input is empty."""
        softmax.forward(empty_array)
        actual = softmax.outputs
        expected = empty_array
        assert np.array_equal(actual, expected)

    def test_output_of_one_dim_input_should_be_correct(self, softmax: ActivationSoftmax, one_dim_array: np.ndarray):
        """Check that output is correct if input is one-dimensional."""
        softmax.forward(one_dim_array)
        actual = softmax.outputs
        expected = np.array([0.0320586, 0.0871443, 0.2368828, 0.6439143])
        assert np.allclose(actual, expected, atol=1e-7)

    def test_output_of_multi_dim_input_should_be_correct(
        self, softmax: ActivationSoftmax, multi_dim_array: np.ndarray
    ):
        """Check that output is correct if input is multi-dimensional."""
        softmax.forward(multi_dim_array)
        actual = softmax.outputs
        expected = np.array(
            [
                [0.0320586, 0.0871443, 0.2368828, 0.6439143],
                [0.0001231, 0.0024720, 0.0001231, 0.9972818],
            ]
        )
        print(actual)
        print(expected)
        assert np.allclose(actual, expected, atol=1e-7)
