from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np

@dataclass
class BasePoint:
    """Класс для представления базовой точки на срезе шины"""
    name: str
    x: float
    y: float
    description: str

@dataclass
class TireLayer:
    """Класс для представления слоя шины"""
    name: str
    type: str
    thickness: float
    is_present: bool = True

class Tire:
    """Класс для представления шины и её характеристик"""
    
    def __init__(self, type: str, model: str, size: str):
        self.type = type  # ЛШ или ЦМК
        self.model = model
        self.size = size
        self.base_points: Dict[str, BasePoint] = {}
        self.layers: List[TireLayer] = []
        self.image = None
        self.processed_image = None
        
    def add_base_point(self, point: BasePoint) -> None:
        """Добавление базовой точки"""
        self.base_points[point.name] = point
        
    def add_layer(self, layer: TireLayer) -> None:
        """Добавление слоя шины"""
        self.layers.append(layer)
        
    def set_image(self, image: np.ndarray) -> None:
        """Установка изображения среза шины"""
        self.image = image
        
    def get_layer_by_name(self, name: str) -> Optional[TireLayer]:
        """Получение слоя по имени"""
        return next((layer for layer in self.layers if layer.name == name), None)
        
    def validate_structure(self) -> Dict[str, bool]:
        """Проверка наличия всех необходимых слоев"""
        return {layer.name: layer.is_present for layer in self.layers} 