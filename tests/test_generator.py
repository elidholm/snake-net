"""Sample data generator test."""

import pytest

from snake_net.generator import Generator


class TestSampleDataGenerator:

    @pytest.fixture(scope="function")
    def data_generator(self) -> Generator:
        return Generator()

    @pytest.fixture(scope="function")
    def n_points(self) -> int:
        return 100

    @pytest.fixture(scope="function")
    def n_classes(self) -> int:
        return 3

    def test_shape_of_output_points_should_be_correct(self, data_generator: Generator, n_points: int, n_classes: int):
        """Check that output points are the correct shape."""
        points, _ = data_generator.spiral_data(n_points, n_classes)
        actual = points.shape
        expected = (n_points * n_classes, 2)
        assert actual == expected

    def test_shape_of_output_labels_should_be_correct(self, data_generator: Generator, n_points: int, n_classes: int):
        """Check that outputs labels are the correct shape."""
        _, labels = data_generator.spiral_data(n_points, n_classes)
        actual = labels.shape
        expected = (n_points * n_classes,)
        assert actual == expected
