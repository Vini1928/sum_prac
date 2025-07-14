import pytest
import numpy as np
import cv2

@pytest.fixture(scope="module")
def sample_image():
    """
    Создает простое тестовое изображение.
    Черный фон (300x400) с белым прямоугольником в центре.
    Это предсказуемая структура для тестирования поиска контуров и точек.
    """
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    # Рисуем белый прямоугольник
    cv2.rectangle(img, (100, 100), (300, 200), (255, 255, 255), -1)
    return img

@pytest.fixture(scope="module")
def simple_gray_image():
    """
    Создает простое изображение в оттенках серого для тестирования.
    """
    img = np.zeros((100, 100), dtype=np.uint8)
    img[25:75, 25:75] = 150 # Серый квадрат в центре
    return img 