from typing import Tuple, List, Dict
import numpy as np
from .image_processor import ImageProcessor
from ..models.tire import Tire, BasePoint

class MeasurementSystem:
    """Класс для выполнения измерений на срезе шины"""
    
    def __init__(self, image_processor: ImageProcessor):
        self.image_processor = image_processor
        self.measurement_history: List[Dict] = []
        
    def set_calibration(self, real_distance: float, pixel_distance: float) -> None:
        """Установка калибровки системы"""
        self.image_processor.set_calibration(real_distance, pixel_distance)
        
    def set_calibration_factor(self, factor: float) -> None:
        """Установка калибровочного коэффициента напрямую."""
        self.image_processor.calibration_factor = factor
        
    def measure_direct_distance(self, point1: BasePoint, point2: BasePoint) -> float:
        """Измерение прямого расстояния между двумя точками"""
        return self.image_processor.measure_distance(
            (point1.x, point1.y),
            (point2.x, point2.y)
        )
        
    def measure_radial_distance(self, center_point: BasePoint, 
                              target_point: BasePoint) -> float:
        """Измерение радиального расстояния"""
        # Вычисление угла между точками
        angle = np.arctan2(target_point.y - center_point.y,
                          target_point.x - center_point.x)
        # Измерение расстояния
        distance = self.measure_direct_distance(center_point, target_point)
        return distance
        
    def measure_perpendicular_distance(self, point: BasePoint, 
                                     line_start: BasePoint,
                                     line_end: BasePoint) -> float:
        """Измерение перпендикулярного расстояния от точки до линии"""
        # Вычисление расстояния от точки до прямой
        numerator = abs((line_end.y - line_start.y) * point.x -
                       (line_end.x - line_start.x) * point.y +
                       line_end.x * line_start.y -
                       line_end.y * line_start.x)
        denominator = np.sqrt((line_end.y - line_start.y)**2 +
                            (line_end.x - line_start.x)**2)
        return numerator / denominator * self.image_processor.calibration_factor
        
    def add_measurement_to_history(self, measurement_type: str,
                                 points: List[BasePoint],
                                 result: float) -> None:
        """Добавление измерения в историю"""
        self.measurement_history.append({
            'type': measurement_type,
            'points': points,
            'result': result,
            'timestamp': np.datetime64('now')
        })
        
    def get_measurement_history(self) -> List[Dict]:
        """Получение истории измерений"""
        return self.measurement_history
        
    def validate_measurement(self, value: float, expected_range: Tuple[float, float]) -> bool:
        """Проверка измерения на попадание в допустимый диапазон"""
        min_value, max_value = expected_range
        return min_value <= value <= max_value 