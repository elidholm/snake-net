"""Generate data for classification task"""

import logging
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
from rich.logging import RichHandler

logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="%Y-%m-%d %H:%M:%S", handlers=[RichHandler()])
_log = logging.getLogger(__name__)


class Generator:
    @lru_cache(maxsize=128)
    def spiral_data(self, n_points: int, n_classes: int):
        points = np.zeros((n_points * n_classes, 2))
        labels = np.zeros(n_points * n_classes, dtype="uint8")
        for class_number in range(n_classes):
            ix = range(n_points * class_number, n_points * (class_number + 1))
            r = np.linspace(0.0, 1, n_points)
            t = np.linspace(class_number * 4, (class_number + 1) * 4, n_points) + np.random.randn(n_points) * 0.2
            points[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
            labels[ix] = class_number
        return points, labels


def main():
    n_points, n_classes = 100, 3
    _log.info("Number of points: %d", n_points)
    _log.info("Number of classes: %d", n_classes)

    data_generator = Generator()
    points, labels = data_generator.spiral_data(n_points, n_classes)
    _log.debug("Generated points:\n%s", points)
    _log.debug("Labels:\n%s", labels)

    plt.scatter(points[:, 0], points[:, 1], c=labels, cmap="brg")
    plt.show()


if __name__ == "__main__":
    np.random.seed(0)
    main()
