"""LayerDense test."""

import numpy as np
import pytest

from snake_net.layer import LayerDense


class TestLayerDense:

    @pytest.fixture(scope="function")
    def first_layer(self) -> LayerDense:
        return LayerDense(4, 5)

    @pytest.fixture(scope="function")
    def second_layer(self) -> LayerDense:
        return LayerDense(5, 2)

    @pytest.fixture(scope="function")
    def inputs(self) -> np.ndarray:
        inputs = np.array([[1, 2, 3, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]])
        return inputs

    def test_one_layer_output_shape(self, first_layer: LayerDense, inputs: np.ndarray):
        """Test one layer output shape."""
        np.random.seed(0)

        first_layer.forward(inputs)
        actual = first_layer.output.shape
        expected = (3, 5)
        assert actual == expected

    def test_two_layers_output_shape(self, first_layer: LayerDense, second_layer: LayerDense, inputs: np.ndarray):
        """Test two layers output shape"""
        np.random.seed(0)

        first_layer.forward(inputs)
        second_layer.forward(first_layer.output)
        actual = second_layer.output.shape
        expected = (3, 2)
        assert actual == expected

    def test_one_layer_output_values(self, first_layer: LayerDense, inputs: np.ndarray):
        """Test one layer output values."""
        np.random.seed(0)

        first_layer.forward(inputs)
        actual = first_layer.output
        expected = np.array(
            [
                [0.10758131, 1.03983522, 0.24462411, 0.31821498, 0.18851053],
                [-0.08349796, 0.70846411, 0.00293357, 0.44701525, 0.36360538],
                [-0.50763245, 0.55688422, 0.07987797, -0.34889573, 0.04553042],
            ]
        )
        assert np.allclose(actual, expected)

    def test_two_layers_output_values(self, first_layer: LayerDense, second_layer: LayerDense, inputs: np.ndarray):
        """Test two layers output values."""
        np.random.seed(0)

        first_layer.forward(inputs)
        second_layer.forward(first_layer.output)
        actual = second_layer.output
        print(actual)
        expected = np.array(
            [
                [0.148296, -0.08397602],
                [0.14100315, -0.01340469],
                [0.20124979, -0.07290616],
            ]
        )
        assert np.allclose(actual, expected)
