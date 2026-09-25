import unittest

import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput,
    ChessInput,
    EllipseInput,
    MatrixInput,
    MatrixVectorBatchInput,
    OneHotInput,
    RandomMatrixInput,
    RectangleInput,
    TimeSeriesInput,
)

from numpy_tasks import (
    analyze_time_series,
    binarize,
    chess,
    draw_ellipse,
    draw_rectangle,
    matrix_statistics,
    one_hot,
    sum_prod,
    unique_columns,
    unique_rows,
)

class TestNumpyTasks(unittest.TestCase):
    def test_sum_prod(self):
        matrices = np.array([[[1, 0], [0, 1]], [[2, 0], [0, 2]]])
        vectors = np.array([[[1], [2]], [[3], [4]]])
        result = sum_prod(MatrixVectorBatchInput(matrices, vectors))
        np.testing.assert_array_equal(result, np.array([[7], [10]]))

        matrices2 = np.array([[[1, 2], [3, 4]]])
        vectors2 = np.array([[[1], [0]]])
        result2 = sum_prod(MatrixVectorBatchInput(matrices2, vectors2))
        np.testing.assert_array_equal(result2, np.array([[1], [3]]))

    def test_binarize(self):
        matrix = np.array([[0.2, 0.7], [0.5, 0.6]])
        result = binarize(BinarizeInput(matrix, 0.5))
        np.testing.assert_array_equal(result, np.array([[0, 1], [0, 1]]))

        result_default = binarize(BinarizeInput(np.array([[0.4, 0.6]])))
        np.testing.assert_array_equal(result_default, np.array([[0, 1]]))

    def test_unique_rows(self):
        matrix = np.array([[3, 1, 3], [2, 2, 1]])
        result = unique_rows(MatrixInput(matrix))
        self.assertEqual([list(r) for r in result], [[1, 3], [1, 2]])

    def test_unique_columns(self):
        matrix = np.array([[3, 1, 3], [2, 2, 1]])
        result = unique_columns(MatrixInput(matrix))
        self.assertEqual([list(c) for c in result], [[2, 3], [1, 2], [1, 3]])

    def test_matrix_statistics(self):
        data = RandomMatrixInput(rows=2, columns=3, mean=0.0, std=1.0, seed=42)
        stats = matrix_statistics(data)

        np.random.seed(42)
        expected_matrix = np.random.normal(0.0, 1.0, (2, 3))

        np.testing.assert_allclose(stats.matrix, expected_matrix)
        np.testing.assert_allclose(stats.row_means, np.mean(expected_matrix, axis=1))
        np.testing.assert_allclose(stats.column_means, np.mean(expected_matrix, axis=0))
        np.testing.assert_allclose(stats.row_variances, np.var(expected_matrix, axis=1))
        np.testing.assert_allclose(stats.column_variances, np.var(expected_matrix, axis=0))

    def test_chess(self):
        result = chess(ChessInput(rows=2, columns=2, first=1, second=0))
        np.testing.assert_array_equal(result, np.array([[1, 0], [0, 1]]))

        result2 = chess(ChessInput(rows=3, columns=3, first=1, second=0))
        np.testing.assert_array_equal(
            result2, np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
        )

    def test_draw_rectangle(self):
        data = RectangleInput(
            width=2,
            height=2,
            image_height=6,
            image_width=6,
            shape_color=(255, 0, 0),
            background_color=(0, 0, 0),
        )
        image = draw_rectangle(data)
        self.assertEqual(image.shape, (6, 6, 3))
        self.assertEqual(image.dtype, np.uint8)

        np.testing.assert_array_equal(image[0, 0], np.array([0, 0, 0]))

        np.testing.assert_array_equal(image[2, 2], np.array([255, 0, 0]))
        np.testing.assert_array_equal(image[3, 3], np.array([255, 0, 0]))
        np.testing.assert_array_equal(image[0, 3], np.array([0, 0, 0]))

    def test_draw_ellipse(self):
        data = EllipseInput(
            semi_axis_x=2,
            semi_axis_y=1,
            image_height=5,
            image_width=5,
            shape_color=(255, 255, 255),
            background_color=(0, 0, 0),
        )
        image = draw_ellipse(data)
        self.assertEqual(image.shape, (5, 5, 3))
        self.assertEqual(image.dtype, np.uint8)

        np.testing.assert_array_equal(image[2, 2], np.array([255, 255, 255]))

        np.testing.assert_array_equal(image[0, 0], np.array([0, 0, 0]))

    def test_analyze_time_series(self):
        data = TimeSeriesInput(values=[1, 3, 2, 4, 1], window=2)
        stats = analyze_time_series(data)
        self.assertAlmostEqual(stats.mean, 2.2)
        self.assertAlmostEqual(stats.variance, float(np.var([1, 3, 2, 4, 1])))
        self.assertAlmostEqual(stats.std, float(np.std([1, 3, 2, 4, 1])))
        self.assertEqual(list(stats.local_maxima_indices), [1, 3])
        self.assertEqual(list(stats.local_minima_indices), [2])

        np.testing.assert_allclose(stats.moving_average, [2.0, 2.5, 3.0, 2.5])

    def test_one_hot(self):
        result = one_hot(OneHotInput(labels=[0, 2, 1], class_count=3))
        np.testing.assert_array_equal(
            result, np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        )

        result2 = one_hot(OneHotInput(labels=[0, 1, 0]))
        np.testing.assert_array_equal(result2, np.array([[1, 0], [0, 1], [1, 0]]))


if __name__ == '__main__':
    unittest.main()
