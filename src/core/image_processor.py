import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from ..models.tire import BasePoint

class ImageProcessor:
    """Класс для обработки изображений срезов шин"""
    
    def __init__(self):
        self.calibration_factor: float = 1.0  # мм/пиксель
        
    def set_calibration(self, real_distance: float, pixel_distance: float) -> None:
        """Установка калибровочного коэффициента"""
        self.calibration_factor = real_distance / pixel_distance
        
    def measure_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Измерение расстояния между двумя точками в миллиметрах."""
        pixel_distance = np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
        return pixel_distance * self.calibration_factor
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Предварительная обработка изображения. 
        Работает как с цветными, так и с серыми изображениями.
        """
        # Преобразование в оттенки серого, если необходимо
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy() # Работаем с копией, чтобы не изменять оригинал
            
        # Применение фильтра для уменьшения шума
        denoised = cv2.GaussianBlur(gray, (5, 5), 0)
        # Повышение контраста
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        return enhanced
        
    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """Определение границ на изображении"""
        # Применяем оператор Кэнни для поиска границ
        edges = cv2.Canny(image, 50, 150)
        # Применяем морфологические операции для улучшения результата
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        return edges

    def find_central_line(self, image: np.ndarray) -> BasePoint:
        """Поиск центральной линии"""
        height, width = image.shape[:2]
        # Находим центр изображения по горизонтали
        center_x = width // 2
        # Ищем верхнюю точку профиля шины
        edges = self.detect_edges(image)
        for y in range(height):
            if edges[y, center_x] > 0:
                return BasePoint("Ц", float(center_x), float(y), "Центральная линия")
        return BasePoint("Ц", float(center_x), 0.0, "Центральная линия")

    def find_shoulder_point(self, image: np.ndarray, central_point: BasePoint) -> BasePoint:
        """
        Поиск точки плеча (П) на основе анализа кривизны сглаженного контура.
        """
        edges = self.detect_edges(image)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return BasePoint("П", 0.0, 0.0, "Точка плеча снаружи")

        tire_contour = max(contours, key=cv2.contourArea)

        # Сглаживаем контур, чтобы убрать шум от протектора
        epsilon = 0.001 * cv2.arcLength(tire_contour, True)
        approx_contour = cv2.approxPolyDP(tire_contour, epsilon, True)

        # Ищем точку в верхней правой части шины
        height, width = image.shape[:2]
        upper_y_limit = height * 0.4
        right_x_limit = central_point.x + (width - central_point.x) * 0.2

        candidate_points = [
            p[0] for p in approx_contour 
            if int(p[0][1]) < upper_y_limit and int(p[0][0]) > right_x_limit # type: ignore
        ]

        if not candidate_points:
            return BasePoint("П", 0.0, 0.0, "Точка плеча снаружи")

        # Ищем точку с максимальным изгибом (наиболее удаленную от линии,
        # соединяющей верхнюю центральную точку и боковую точку).
        # Возьмем самую верхнюю и самую правую точки из кандидатов для построения линии.
        top_point = min(candidate_points, key=lambda p: p[1])
        right_point = max(candidate_points, key=lambda p: p[0])
        
        line_vec = np.array(right_point) - np.array(top_point)
        if np.linalg.norm(line_vec) == 0: # Если точки совпадают
             return BasePoint("П", float(candidate_points[0][0]), float(candidate_points[0][1]), "Точка плеча снаружи")

        max_dist = -1
        shoulder_point = candidate_points[0]

        for point in candidate_points:
            point_vec = np.array(point) - np.array(top_point)
            # Находим перпендикулярное расстояние до линии
            cross_product = np.cross(line_vec, point_vec)
            dist = np.linalg.norm(cross_product) / np.linalg.norm(line_vec)
            if dist > max_dist:
                max_dist = dist
                shoulder_point = point

        return BasePoint("П", float(shoulder_point[0]), float(shoulder_point[1]), "Точка плеча снаружи")

    def find_mold_split_line(self, image: np.ndarray) -> BasePoint:
        """Поиск линии размыкания пресс-формы"""
        edges = self.detect_edges(image)
        height, width = edges.shape[:2]
        
        # Ищем характерную точку на боковой поверхности
        quarter_height = height // 4
        for x in range(width-1, width//2, -1):  # Идем справа налево
            for y in range(quarter_height, height-quarter_height):
                if edges[y, x] > 0:
                    # Проверяем характерный угол
                    if x > 0 and y > 0:
                        if edges[y-1, x-1] > 0 and edges[y+1, x-1] == 0:
                            return BasePoint("ЛР", x, y, "Линия размыкания пресс-формы")
        
        return BasePoint("ЛР", width-1, height//2, "Линия размыкания пресс-формы")

    def find_protective_rib(self, image: np.ndarray, shoulder_point: BasePoint) -> BasePoint:
        """
        Поиск вершины защитного ребра обода. Ищем точку с максимальной кривизной
        на боковине, ниже точки плеча.
        """
        edges = self.detect_edges(image)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return BasePoint("З", 0.0, 0.0, "Вершина защитного ребра обода")
            
        tire_contour = max(contours, key=cv2.contourArea)

        # Ищем точку на боковине (справа от центра и ниже плеча)
        mid_height = image.shape[0] / 2
        
        # Ищем наиболее выступающую вправо точку на боковине
        sidewall_points = [p[0] for p in tire_contour if p[0][1] > shoulder_point.y and p[0][1] < mid_height]
        if not sidewall_points:
             return BasePoint("З", shoulder_point.x, shoulder_point.y + 20, "Вершина защитного ребра обода")
        
        rib_point = max(sidewall_points, key=lambda p: p[0])
        
        return BasePoint("З", float(rib_point[0]), float(rib_point[1]), "Вершина защитного ребра обода")

    def find_bead_points(self, image: np.ndarray) -> Tuple[BasePoint, BasePoint, BasePoint]:
        """
        Финальный, единственно верный алгоритм поиска точек борта.
        Логика:
        1. Найти самую нижнюю точку всего контура (ЦБ).
        2. Разделить контур на "левую" и "правую" части относительно ЦБ.
        3. Найти самую левую точку в левой части (НБ) и самую правую в правой (ЛБ).
        """
        edges = self.detect_edges(image)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        default_nb = BasePoint("НБ", 0.0, 0.0, "Носок борта")
        default_cb = BasePoint("ЦБ", 0.0, 0.0, "Центр пятки борта")
        default_lb = BasePoint("ЛБ", 0.0, 0.0, "Линия борта")

        if not contours:
            return default_nb, default_cb, default_lb

        tire_contour = max(contours, key=cv2.contourArea)
        all_points = [p[0] for p in tire_contour]

        if not all_points:
            return default_nb, default_cb, default_lb
        
        # Шаг 1: Находим ЦБ - самую нижнюю точку всего контура
        cb_point = max(all_points, key=lambda p: p[1])
        cb = BasePoint("ЦБ", float(cb_point[0]), float(cb_point[1]), "Центр пятки борта")

        # Шаг 2: Разделяем контур на левую и правую части относительно ЦБ
        left_part = [p for p in all_points if p[0] < cb_point[0]]
        right_part = [p for p in all_points if p[0] > cb_point[0]]

        # Проверка, что обе части не пусты
        if not left_part or not right_part:
            return default_nb, cb, default_lb

        # Шаг 3: Ищем НБ в левой части и ЛБ в правой
        # НБ - самая левая точка в левой части
        nb_point = min(left_part, key=lambda p: p[0])
        nb = BasePoint("НБ", float(nb_point[0]), float(nb_point[1]), "Носок борта")

        # ЛБ - самая правая точка в правой части
        lb_point = max(right_part, key=lambda p: p[0])
        lb = BasePoint("ЛБ", float(lb_point[0]), float(lb_point[1]), "Линия борта")
            
        return nb, cb, lb

    def find_base_points(self, image: np.ndarray) -> Dict[str, BasePoint]:
        """Основной метод для поиска всех базовых точек."""
        points = {}

        # 1. Центральная линия (основа для других точек)
        central_point = self.find_central_line(image)
        points[central_point.name] = central_point

        # 2. Точка плеча (зависит от центра)
        shoulder_point = self.find_shoulder_point(image, central_point)
        points[shoulder_point.name] = shoulder_point

        # 3. Линия размыкания пресс-формы (не зависит от других)
        mold_line_point = self.find_mold_split_line(image)
        points[mold_line_point.name] = mold_line_point

        # 4. Вершина защитного ребра (зависит от плеча)
        protective_rib_point = self.find_protective_rib(image, shoulder_point)
        points[protective_rib_point.name] = protective_rib_point

        # 5. Точки борта (самые сложные, не зависят от других)
        nb, cb, lb = self.find_bead_points(image)
        points[nb.name] = nb
        points[cb.name] = cb
        points[lb.name] = lb

        return points
        
    def filter_artifacts(self, image: np.ndarray) -> np.ndarray:
        """Фильтрация артефактов и помех"""
        # Морфологические операции для удаления мелких артефактов
        kernel = np.ones((3,3), np.uint8)
        cleaned = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        return cleaned
        
    def detect_layers(self, image: np.ndarray, num_layers: int = 7) -> Tuple[np.ndarray, Dict[int, List[np.ndarray]]]:
        """
        Определение слоев шины с использованием K-Means кластеризации.
        Применяется только к области самой шины, игнорируя фон.
        
        :param image: Входное изображение (ожидается в оттенках серого).
        :param num_layers: Ожидаемое количество слоев (кластеров).
        :return: Кортеж (маска с метками кластеров, словарь {id_кластера: [контуры]}).
        """
        # 1. Создание маски шины для исключения фона
        # Используем инвертированный порог, так как фон на тестовом изображении светлый
        _, thresh = cv2.threshold(image, 240, 255, cv2.THRESH_BINARY_INV)
        
        # Находим контуры и берем самый большой (это должна быть шина)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros_like(image), {}
            
        tire_contour = max(contours, key=cv2.contourArea)
        
        # Создаем маску, заливая контур шины
        tire_mask = np.zeros_like(image)
        cv2.drawContours(tire_mask, [tire_contour], -1, 255, -1)

        # 2. Подготовка данных для K-Means только из области шины
        pixels_to_cluster = image[tire_mask == 255]
        pixels_to_cluster = pixels_to_cluster.reshape(-1, 1).astype(np.float32)

        # 3. Применение K-Means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
        _, labels, _ = cv2.kmeans(pixels_to_cluster, num_layers, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # 4. Восстановление изображения и создание масок
        segmented_mask = np.zeros_like(image)
        # Присваиваем метки кластеров обратно в область маски
        segmented_mask[tire_mask == 255] = labels.flatten()
        
        # 5. Поиск контуров для каждого слоя
        layer_contours = {}
        for i in range(num_layers):
            mask = np.uint8(segmented_mask == i)
            
            # Убираем шум
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            cluster_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Фильтруем слишком маленькие контуры
            min_area = 100 
            layer_contours[i] = [c for c in cluster_contours if cv2.contourArea(c) > min_area]

        return np.uint8(segmented_mask), layer_contours 

    def determine_layers(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Определение слоев шины с использованием K-Means кластеризации.
        Возвращает информацию о слоях (включая площадь и контуры).
        """
        # 1. Создание маски шины для исключения фона
        # Используем инвертированный порог, так как фон на изображении светлый
        _, thresh = cv2.threshold(image, 240, 255, cv2.THRESH_BINARY_INV)
        
        # Находим контуры и берем самый большой (это должна быть шина)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {"message": "Не удалось найти контур шины.", "layers": {}}
            
        tire_contour = max(contours, key=cv2.contourArea)
        
        # Создаем маску, заливая контур шины
        tire_mask = np.zeros_like(image)
        cv2.drawContours(tire_mask, [tire_contour], -1, 255, -1)

        # 2. Подготовка данных для K-Means только из области шины
        pixel_values = image[tire_mask == 255].reshape(-1, 1).astype(np.float32)

        # 3. Применение K-Means
        k = 8
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.95)
        _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        centers = np.uint8(centers)
        
        # 4. Восстановление сегментированного изображения только в области маски
        segmented_grayscale = np.zeros_like(image)
        segmented_grayscale[tire_mask == 255] = centers[labels.flatten()].flatten()

        # 5. Находим и анализируем слои
        layer_info = {}

        for i in range(k):
            # Создаем маску для каждого кластера
            mask = np.uint8(segmented_grayscale == centers[i])
            pixel_count = cv2.countNonZero(mask)
            
            # Пропускаем пустые слои
            if pixel_count == 0:
                continue

            # Расчет площади в мм^2
            area_sq_mm = pixel_count * (self.calibration_factor ** 2) if self.calibration_factor > 0 else 0
            
            # Находим контуры для этого слоя
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            layer_name = f"Слой {len(layer_info) + 1}"
            layer_info[layer_name] = {
                "pixel_count": pixel_count,
                "area_sq_mm": f"{area_sq_mm:.2f} мм²",
                "contours": [c for c in contours if cv2.contourArea(c) > 50] # Фильтруем мелкие
            }
        
        # Сортируем слои по размеру для удобства
        sorted_layers = sorted(layer_info.items(), key=lambda item: item[1]['pixel_count'], reverse=True)
        
        final_layer_info = {
            "message": f"Найдено {len(sorted_layers)} групп слоев.",
            "layers": dict(sorted_layers)
        }

        return final_layer_info 