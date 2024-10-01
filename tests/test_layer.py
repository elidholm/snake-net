"""LayerDense test."""

import numpy as np
import pytest

from snake_net.layer import LayerDense

np.random.seed(0)


class TestLayerDense:

    @pytest.fixture(scope="function")
    def number_of_neurons(self) -> int:
        return 5

    @pytest.fixture(scope="function")
    def number_of_inputs(self) -> int:
        return 4

    @pytest.fixture(scope="function")
    def input_batch_size(self) -> int:
        return 3

    @pytest.fixture(scope="function")
    def layer(self, number_of_neurons: int, number_of_inputs: int) -> LayerDense:
        return LayerDense(number_of_neurons, number_of_inputs)

    @pytest.fixture(scope="function")
    def inputs(self, number_of_inputs: int, input_batch_size: int) -> np.ndarray:
        inputs = np.random.randint(-5, 5, size=(number_of_inputs, input_batch_size))
        return inputs

    def test_one_layer_outputs_shape(
        self, layer: LayerDense, inputs: np.ndarray, number_of_neurons: int, input_batch_size: int
    ):
        """Test one layer outputs shape."""

        outputs = layer.forward(inputs)
        actual = outputs.shape
        expected = (number_of_neurons, input_batch_size)
        assert actual == expected

    def test_one_layer_outputs_values(self, layer: LayerDense, inputs: np.ndarray):
        """Test one layer outputs values."""

        actual = layer.forward(inputs)

        print(actual)
        expected = np.array(
            [
                [1.1209, -4.0330, 1.1846],
                [0.7939, -4.0740, 1.5949],
                [-1.5367, -7.3221, -5.9008],
                [-1.9041, -2.7028, -4.6993],
                [1.6641, 0.8696, 3.1323],
            ]
        )
        assert np.allclose(actual, expected, atol=1e-4)
