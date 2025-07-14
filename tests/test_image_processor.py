import pytest
import numpy as np
from src.core.image_processor import ImageProcessor
from src.models.tire import BasePoint

def test_preprocess_image(sample_image):
    """Тест предварительной обработки изображения."""
    processor = ImageProcessor()
    processed_image = processor.preprocess_image(sample_image)
    
    # Ожидаем, что изображение стало одноканальным (в оттенках серого)
    assert len(processed_image.shape) == 2
    # Ожидаем, что размеры не изменились
    assert processed_image.shape == (sample_image.shape[0], sample_image.shape[1])

def test_detect_edges(sample_image):
    """Тест детектора границ."""
    processor = ImageProcessor()
    # Сначала нужна предобработка
    gray_image = processor.preprocess_image(sample_image)
    edges = processor.detect_edges(gray_image)
    
    # Ожидаем, что карта границ имеет те же размеры
    assert edges.shape == (sample_image.shape[0], sample_image.shape[1])
    # Ожидаем, что на изображении есть белые пиксели (границы)
    assert np.any(edges > 0)

def test_find_base_points_structure(sample_image):
    """
    Тест структуры возвращаемых данных при поиске базовых точек.
    Так как алгоритмы сложны, здесь мы проверяем не точность, а формат.
    """
    processor = ImageProcessor()
    base_points = processor.find_base_points(sample_image)

    # Ожидаем, что результат - словарь
    assert isinstance(base_points, dict)
    # Ожидаем наличие всех ключевых точек
    expected_keys = ["Ц", "П", "ЛР", "З", "ЦБ", "НБ", "ЛБ"]
    assert all(key in base_points for key in expected_keys)
    # Ожидаем, что все значения - это объекты BasePoint
    assert all(isinstance(point, BasePoint) for point in base_points.values())

def test_detect_layers_structure(simple_gray_image):
    """Тест структуры данных при определении слоев."""
    processor = ImageProcessor()
    
    num_layers = 3 # Тестируем с 3 слоями
    segmented_mask, layer_contours = processor.detect_layers(simple_gray_image, num_layers=num_layers)

    # Проверяем маску
    assert segmented_mask.shape == simple_gray_image.shape
    assert segmented_mask.dtype == np.uint8
    # Уникальных значений в маске должно быть не больше, чем мы просили кластеров
    assert len(np.unique(segmented_mask)) <= num_layers

    # Проверяем контуры
    assert isinstance(layer_contours, dict)
    # Ключей должно быть столько, сколько просили кластеров
    assert len(layer_contours) == num_layers
    # Значения должны быть списками
    assert all(isinstance(v, list) for v in layer_contours.values()) 