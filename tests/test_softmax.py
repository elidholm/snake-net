"""Softmax activation function test."""

import numpy as np
import pytest

from snake_net.activation import ActivationSoftmax


class TestActivationSoftmax:

    @pytest.fixture(scope="function")
    def softmax(self) -> ActivationSoftmax:
        return ActivationSoftmax()

    @pytest.fixture(scope="function")
    def one_dim_array(self) -> np.ndarray:
        return np.array([1, 2, 3, 4])

    @pytest.fixture(scope="function")
    def multi_dim_array(self) -> np.ndarray:
        return np.array([[1, 2, 3, 4], [-2, 1, -2, 7]])

    def test_output_of_one_dim_input_should_be_correct(self, softmax: ActivationSoftmax, one_dim_array: np.ndarray):
        """Check that output is correct if input is one-dimensional."""
        actual = softmax.forward(one_dim_array)
        expected = np.array([0.0321, 0.0871, 0.2369, 0.6439])
        assert np.allclose(actual, expected, atol=1e-4)

    def test_output_of_multi_dim_input_should_be_correct(
        self, softmax: ActivationSoftmax, multi_dim_array: np.ndarray
    ):
        """Check that output is correct if input is multi-dimensional."""
        actual = softmax.forward(multi_dim_array)
        expected = np.array(
            [
                [0.9526, 0.7311, 0.9933, 0.0474],
                [0.0474, 0.2689, 0.0067, 0.9526],
            ]
        )
        print(actual, np.sum(actual))
        print(expected, np.sum(expected))
        assert np.allclose(actual, expected, atol=1e-4)
