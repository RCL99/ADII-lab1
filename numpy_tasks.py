"""Заготовки задач на NumPy."""

import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors
    return np.sum(np.matmul(matrices, vectors), axis=0)


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    return np.where(matrix > threshold, 1, 0)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    return [np.unique(row).tolist() for row in matrix]


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    return [np.unique(col).tolist() for col in np.transpose(matrix)]


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed

    np.random.seed(seed)

    matrix = np.random.normal(mean, std, (rows, columns))

    # средние и дисперсии по строкам axis=1 и столбцам axis=0
    row_means = np.mean(matrix, axis=1)
    column_means = np.mean(matrix, axis=0)

    row_variances = np.var(matrix, axis=1)
    column_variances = np.var(matrix, axis=0)

    return MatrixStatistics(
        matrix=matrix,
        row_means=row_means,
        column_means=column_means,
        row_variances=row_variances,
        column_variances=column_variances
    )


def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    # сумма координат % 2 == 0 ? first : second
    indices = np.indices((rows, columns)).sum(axis=0)
    return np.where(indices % 2 == 0, first, second)


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color

    # фон
    image = np.full(
        (image_height, image_width, 3),
        background_color,
        dtype=np.uint8
    )

    # центр
    x0 = image_width // 2
    y0 = image_height // 2

    # границы прямоугольника вокруг центра
    x1 = x0 - width // 2
    x2 = x1 + width
    y1 = y0 - height // 2
    y2 = y1 + height

    # закрашивание прямоугольника
    image[y1:y2, x1:x2] = shape_color

    return image



def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color

    # фон
    image = np.full(
        (image_height, image_width, 3),
        background_color,
        dtype=np.uint8
    )

    # центр картинки
    x0 = image_width // 2
    y0 = image_height // 2

    y, x = np.ogrid[:image_height, :image_width]
    mask = (
        ((x - x0) ** 2 / semi_axis_x ** 2) +
        ((y - y0) ** 2 / semi_axis_y ** 2)
        <= 1
    )

    # закрашивание эллипса
    image[mask] = shape_color

    return image


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = data.values, data.window

    mean_val = np.mean(values)  # мат. ожидание
    var_val = np.var(values)  # дисперсия
    std_val = np.std(values)  # среднеквадратичное отклонение

    # локальные min и max
    local_maxima = []
    local_minima = []

    for i in range(1, len(values) - 1):
        if values[i - 1] < values[i] > values[i + 1]:
            local_maxima.append(i)
        elif values[i - 1] > values[i] < values[i + 1]:
            local_minima.append(i)

    # скользящее среднее
    moving_average = []
    for i in range(len(values) - window + 1):
        moving_average.append(sum(values[i : i + window]) / window)

    return TimeSeriesStatistics(
        mean=mean_val,
        variance=var_val,
        std=std_val,
        local_maxima_indices=local_maxima,
        local_minima_indices=local_minima,
        moving_average=moving_average,
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = data.labels, data.class_count

    num_classes = class_count if class_count is not None else max(labels) + 1

    matrix = np.zeros((len(labels), num_classes), dtype=int)
    for i, label in enumerate(labels):
        matrix[i, label] = 1

    return matrix