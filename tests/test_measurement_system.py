import pytest
from src.core.measurements import MeasurementSystem
from src.models.tire import BasePoint

@pytest.fixture
def measurement_system():
    """Фикстура для создания экземпляра MeasurementSystem."""
    return MeasurementSystem()

def test_initial_calibration(measurement_system):
    """Проверка, что начальный коэффициент калибровки равен 1.0."""
    assert measurement_system.image_processor.calibration_factor == 1.0

def test_set_calibration(measurement_system):
    """Тест установки калибровочного коэффициента."""
    measurement_system.set_calibration(real_distance=10.0, pixel_distance=20.0)
    assert measurement_system.image_processor.calibration_factor == 0.5

@pytest.mark.parametrize("p1_coords, p2_coords, cal_factor, expected_dist", [
    # Тест 1: Простой случай, без калибровки (Пифагорова тройка)
    ((0, 0), (3, 4), 1.0, 5.0),
    # Тест 2: Тот же случай, но с калибровкой (1 пиксель = 2 мм)
    ((0, 0), (3, 4), 2.0, 10.0),
    # Тест 3: Горизонтальная линия
    ((10, 20), (20, 20), 1.5, 15.0),
    # Тест 4: Вертикальная линия
    ((5, 5), (5, 10), 3.0, 15.0),
    # Тест 5: Нулевое расстояние
    ((1, 1), (1, 1), 10.0, 0.0),
])
def test_measure_direct_distance(measurement_system, p1_coords, p2_coords, cal_factor, expected_dist):
    """Тест измерения прямого расстояния с разными параметрами."""
    # Устанавливаем калибровку
    measurement_system.image_processor.calibration_factor = cal_factor
    
    # Создаем базовые точки
    p1 = BasePoint("P1", p1_coords[0], p1_coords[1], "")
    p2 = BasePoint("P2", p2_coords[0], p2_coords[1], "")
    
    distance = measurement_system.measure_direct_distance(p1, p2)
    
    # Проверяем, что вычисленное расстояние близко к ожидаемому
    assert distance == pytest.approx(expected_dist)

def test_measurement_history(measurement_system):
    """Тест сохранения измерений в историю."""
    assert len(measurement_system.get_measurement_history()) == 0
    
    p1 = BasePoint("M1", 0, 0, "")
    p2 = BasePoint("M2", 1, 1, "")
    
    measurement_system.add_measurement_to_history(
        measurement_type='distance',
        points=[p1, p2],
        result=1.414
    )
    
    history = measurement_system.get_measurement_history()
    assert len(history) == 1
    assert history[0]['type'] == 'distance'
    assert history[0]['result'] == 1.414 