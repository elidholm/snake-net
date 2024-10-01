"""Helper functions test."""

import numpy as np
import pytest

from snake_net.helpers import one_hot_encode, relu_derivative


class TestHelperFunctions:

    @pytest.fixture(scope="function")
    def labels(self) -> np.ndarray:
        return np.array([0, 0, 0, 1, 2, 1, 0, 2, 2, 1, 2, 1, 1, 0])

    @pytest.fixture(scope="function")
    def inputs(self) -> np.ndarray:
        return np.array(
            [[1, -2, 3, 0, -4], [0, 0, 0, 0, 0], [1, 2, 3, 4, 5], [-1, 0, -1, 0, -1], [-1, -1, -1, -1, -1]]
        )

    def test_one_hot_encoding_shape(self, labels: np.ndarray):
        """Test one-hot encoding shape."""
        actual = one_hot_encode(labels)
        expected = (3, 14)
        assert actual.shape == expected

    def test_one_hot_encoding_values(self, labels: np.ndarray):
        """Test one-hot encoding values."""
        actual = one_hot_encode(labels)
        expected = np.array(
            [
                [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0],
                [0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0],
            ]
        )
        assert np.array_equal(actual, expected)

    def test_relu_derivative(self, inputs: np.ndarray):
        """Test ReLU derivative function."""
        actual = relu_derivative(inputs)
        expected = np.array([[1, 0, 1, 0, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
        assert np.array_equal(actual, expected)
