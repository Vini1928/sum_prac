import sys
import json
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QHBoxLayout, QPushButton, QLabel, QFileDialog,
                            QStatusBar, QToolBar, QMessageBox, QListWidget)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QAction, QImage, QPixmap
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..core.image_processor import ImageProcessor
from ..core.measurements import MeasurementSystem
from ..models.tire import Tire, BasePoint
from .calibration_dialog import CalibrationDialog

class MainWindow(QMainWindow):
    """Главное окно приложения"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Анализ радиальных срезов шин")
        self.setGeometry(100, 100, 1200, 800)
        
        self.image_processor = ImageProcessor()
        self.measurement_system = MeasurementSystem(self.image_processor)
        self.current_tire = None
        self.base_points = {}

        self.is_measuring = False
        self.measurement_type = None
        self.measurement_points = []
        
        self.detected_layers = {}
        self.highlighted_layer_name = None
        self.layer_colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
            (0, 255, 255), (255, 0, 255), (128, 0, 128)
        ]

        self._create_ui()
        self._create_menu()
        self._create_toolbar()
        self.statusBar().showMessage("Готов к работе")
        
    def _create_ui(self):
        """Создание элементов интерфейса"""
        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Главный layout
        layout = QHBoxLayout(central_widget)
        
        # Область изображения
        self.image_label = QLabel()
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid black")
        self.image_label.mousePressEvent = self.image_label_clicked
        layout.addWidget(self.image_label)
        
        # Панель инструментов
        tools_panel = QWidget()
        tools_layout = QVBoxLayout(tools_panel)
        
        # Кнопки
        self.load_button = QPushButton("Загрузить изображение")
        self.load_button.clicked.connect(self.load_image)
        
        self.process_button = QPushButton("Обработать")
        self.process_button.clicked.connect(self.process_image)
        self.process_button.setEnabled(False)
        
        self.find_points_button = QPushButton("Найти базовые точки")
        self.find_points_button.clicked.connect(self.find_base_points)
        self.find_points_button.setEnabled(False)
        
        self.measure_dist_button = QPushButton("Измерить расстояние")
        self.measure_dist_button.clicked.connect(self.start_distance_measurement)
        self.measure_dist_button.setEnabled(False)

        self.detect_layers_button = QPushButton("Определить слои")
        self.detect_layers_button.clicked.connect(self.detect_layers)
        self.detect_layers_button.setEnabled(False)

        self.measure_button = QPushButton("Измерить")
        self.measure_button.clicked.connect(self.start_measurement)
        self.measure_button.setEnabled(False)
        
        self.save_report_button = QPushButton("Сохранить отчет")
        self.save_report_button.clicked.connect(self.save_report)
        self.save_report_button.setEnabled(False)
        
        tools_layout.addWidget(self.load_button)
        tools_layout.addWidget(self.process_button)
        tools_layout.addWidget(self.find_points_button)
        tools_layout.addWidget(self.measure_dist_button)
        tools_layout.addWidget(self.detect_layers_button)
        tools_layout.addWidget(self.save_report_button)
        tools_layout.addWidget(self.measure_button)
        tools_layout.addStretch()
        
        # Панель информации о точках
        points_info_panel = QWidget()
        points_info_layout = QVBoxLayout(points_info_panel)
        self.points_info_label = QLabel("Базовые точки:")
        points_info_layout.addWidget(self.points_info_label)
        tools_layout.addWidget(points_info_panel)

        # Панель результатов измерений
        self.results_list = QListWidget()
        tools_layout.addWidget(QLabel("Результаты измерений:"))
        tools_layout.addWidget(self.results_list)

        # Панель обнаруженных слоев
        self.layers_list = QListWidget()
        self.layers_list.currentItemChanged.connect(self.on_layer_selected)
        tools_layout.addWidget(QLabel("Обнаруженные слои:"))
        tools_layout.addWidget(self.layers_list)
        
        layout.addWidget(tools_panel)
        
    def _create_menu(self):
        """Создание главного меню"""
        menubar = self.menuBar()
        
        # Меню Файл
        file_menu = menubar.addMenu("Файл")
        
        open_action = QAction("Открыть", self)
        open_action.triggered.connect(self.load_image)
        file_menu.addAction(open_action)
        
        save_action = QAction("Сохранить отчет", self)
        save_action.triggered.connect(self.save_report)
        file_menu.addAction(save_action)

        file_menu.addSeparator()
        exit_action = QAction("Выход", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Меню Инструменты
        tools_menu = menubar.addMenu("Инструменты")
        
        calibrate_action = QAction("Калибровка", self)
        calibrate_action.triggered.connect(self.open_calibration_dialog)
        tools_menu.addAction(calibrate_action)
        
    def _create_toolbar(self):
        """Создание панели инструментов"""
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        
        toolbar.addAction(QAction("Открыть", self, triggered=self.load_image))
        toolbar.addAction(QAction("Обработать", self, triggered=self.process_image))
        toolbar.addAction(QAction("Калибровка", self, triggered=self.open_calibration_dialog))

    def on_layer_selected(self, current, previous):
        """Обработчик выбора элемента в списке слоев."""
        if current is not None:
            # Извлекаем имя слоя из текста элемента
            self.highlighted_layer_name = current.text().split(":")[0]
        else:
            self.highlighted_layer_name = None
        
        # Перерисовываем изображение для обновления подсветки
        if self.current_tire and self.current_tire.image is not None:
            self.display_image(self.current_tire.image)
        
    def display_image(self, image: np.ndarray):
        """Отображение изображения с аннотациями (рефакторинг для стабильности)."""
        img_to_show = image.copy()
        
        if len(img_to_show.shape) == 2:
            img_to_show = cv2.cvtColor(img_to_show, cv2.COLOR_GRAY2BGR)

        # 1. Отрисовка контуров слоев
        if self.detected_layers:
            layers_data = self.detected_layers.get("layers", {})
            for i, (name, data) in enumerate(layers_data.items()):
                contours = data.get("contours")
                if contours:
                    color = self.layer_colors[i % len(self.layer_colors)]
                    cv2.drawContours(img_to_show, contours, -1, color, 2)
        
            # 2. Отрисовка подсветки
            if self.highlighted_layer_name and self.highlighted_layer_name in layers_data:
                highlight_data = layers_data[self.highlighted_layer_name]
                contours = highlight_data.get("contours")
                if contours:
                    # Рисуем толстый белый контур поверх
                    cv2.drawContours(img_to_show, contours, -1, (255, 255, 255), 4)

        # Конвертация в Pillow для надежной отрисовки текста и фигур
        pil_img = Image.fromarray(cv2.cvtColor(img_to_show, cv2.COLOR_BGR2RGB)).convert("RGBA")
        draw = ImageDraw.Draw(pil_img)
        try:
            font = ImageFont.truetype("arial.ttf", size=20)
        except IOError:
            font = ImageFont.load_default(size=20)

        # 3. Отрисовка базовых точек с помощью Pillow
        if self.base_points:
            for name, point in self.base_points.items():
                x, y = int(point.x), int(point.y)
                # Рисуем красный круг
                draw.ellipse((x-6, y-6, x+6, y+6), fill=(255, 0, 0, 255))
                # Рисуем фон для текста
                text_bbox = draw.textbbox((x + 10, y - 15), name, font=font)
                plaque_bbox = (text_bbox[0]-5, text_bbox[1]-5, text_bbox[2]+5, text_bbox[3]+5)
                draw.rectangle(plaque_bbox, fill=(0, 0, 0, 128))
                # Рисуем текст
                draw.text((x + 10, y - 15), name, font=font, fill=(255, 255, 0))

        # 4. Отрисовка завершенных измерений с помощью Pillow
        history = self.measurement_system.get_measurement_history()
        for meas in history:
            if meas['type'] == 'distance':
                p1 = (int(meas['points'][0].x), int(meas['points'][0].y))
                p2 = (int(meas['points'][1].x), int(meas['points'][1].y))
                draw.line([p1, p2], fill=(0, 255, 0, 255), width=2)
                text_pos = (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2) - 10)
                draw.text(text_pos, f"{meas['result']:.2f}mm", font=font, fill=(0, 255, 0))

        # 5. Отрисовка временной точки измерения с помощью Pillow
        if self.is_measuring and len(self.measurement_points) == 1:
            p1 = self.measurement_points[0]
            draw.ellipse((p1[0]-5, p1[1]-5, p1[0]+5, p1[1]+5), fill=(0, 255, 255, 255))

        # 6. Конвертация итогового изображения в numpy-массив
        final_img_np = np.array(pil_img)

        # 7. Создание QImage с принудительным копированием данных в памяти
        height, width, _ = final_img_np.shape
        bytes_per_line = 4 * width
        q_image = QImage(final_img_np.data, width, height, bytes_per_line, QImage.Format.Format_RGBA8888).copy()
        
        # 8. Отображение
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio)
        self.image_label.setPixmap(scaled_pixmap)
        
    def update_points_info(self):
        """Обновление информации о базовых точках"""
        if self.base_points:
            info_text = "Базовые точки:\n"
            for name, point in self.base_points.items():
                info_text += f"{name}: ({point.x}, {point.y})\n"
            self.points_info_label.setText(info_text)
        else:
            self.points_info_label.setText("Базовые точки не найдены")
        
    def load_image(self):
        """Загрузка изображения"""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите изображение",
            "",
            "Images (*.png *.jpg *.jpeg)"
        )
        
        if file_name:
            image = cv2.imread(file_name)
            if image is not None:
                self.current_tire = Tire("ЛШ", "Unknown", "Unknown")
                self.current_tire.set_image(image)
                self.display_image(image)
                self.process_button.setEnabled(True)
                self.base_points = {}
                self.update_points_info()
                self.statusBar().showMessage("Изображение загружено")
            else:
                QMessageBox.critical(self, "Ошибка", "Не удалось загрузить изображение")
                
    def process_image(self):
        """Обработка изображения"""
        if self.current_tire and self.current_tire.image is not None:
            processed = self.image_processor.preprocess_image(self.current_tire.image)
            self.current_tire.processed_image = processed
            self.display_image(processed)
            self.find_points_button.setEnabled(True)
            self.statusBar().showMessage("Изображение обработано")
            
    def find_base_points(self):
        """Поиск базовых точек"""
        if self.current_tire and self.current_tire.image is not None:
            try:
                self.base_points = self.image_processor.find_base_points(self.current_tire.image)
                self.display_image(self.current_tire.image)
                self.update_points_info()
                self.measure_dist_button.setEnabled(True)
                self.detect_layers_button.setEnabled(True)
                self.save_report_button.setEnabled(True)
                self.statusBar().showMessage("Базовые точки найдены")
            except Exception as e:
                QMessageBox.warning(self, "Предупреждение",
                                  f"Не удалось найти все базовые точки: {str(e)}")
            
    def start_distance_measurement(self):
        """Начать измерение прямого расстояния"""
        self.is_measuring = True
        self.measurement_type = 'distance'
        self.measurement_points = []
        self.statusBar().showMessage("Режим измерения: выберите первую точку.")
        self.image_label.setCursor(Qt.CursorShape.CrossCursor)

    def image_label_clicked(self, event):
        """Обработка клика по изображению для измерения."""
        if not self.is_measuring or self.current_tire.image is None:
            return

        # --- ИСПРАВЛЕНИЕ: Используем размеры оригинального изображения ---
        img_h, img_w = self.current_tire.image.shape[:2]
        
        label_size = self.image_label.size()
        
        scale = min(label_size.width() / img_w, label_size.height() / img_h)
        
        scaled_w = img_w * scale
        scaled_h = img_h * scale
        offset_x = (label_size.width() - scaled_w) / 2
        offset_y = (label_size.height() - scaled_h) / 2
        # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

        # Проверяем, что клик был внутри масштабированного изображения
        if not (offset_x <= event.pos().x() < offset_x + scaled_w and
                offset_y <= event.pos().y() < offset_y + scaled_h):
            return

        img_x = (event.pos().x() - offset_x) / scale
        img_y = (event.pos().y() - offset_y) / scale
        
        self.measurement_points.append((int(img_x), int(img_y)))
        # Немедленно перерисовываем, чтобы показать выбранную точку
        self.display_image(self.current_tire.image)

        if len(self.measurement_points) == 1:
            self.statusBar().showMessage("Выберите вторую точку.")
        elif len(self.measurement_points) == 2:
            self.perform_distance_measurement()

    def perform_distance_measurement(self):
        """Выполнение измерения и отображение результата"""
        p1_coords, p2_coords = self.measurement_points
        
        p1 = BasePoint("M1", p1_coords[0], p1_coords[1], "M-Point 1")
        p2 = BasePoint("M2", p2_coords[0], p2_coords[1], "M-Point 2")

        distance = self.measurement_system.measure_direct_distance(p1, p2)
        
        self.measurement_system.add_measurement_to_history('distance', [p1, p2], distance)

        result_text = f"Расстояние: {distance:.2f} мм (между {p1_coords} и {p2_coords})"
        self.results_list.addItem(result_text)
        
        self.is_measuring = False
        self.measurement_points = []
        self.image_label.setCursor(Qt.CursorShape.ArrowCursor)
        self.statusBar().showMessage(f"Измерено: {distance:.2f} мм. Готов к следующей задаче.")
        
        self.display_image(self.current_tire.image)
        self.save_report_button.setEnabled(True)

    def detect_layers(self):
        """Запуск процесса определения слоев"""
        if self.current_tire is None or self.current_tire.image is None:
            QMessageBox.warning(self, "Внимание", "Сначала загрузите изображение.")
            return

        # Проверяем, что калибровка была выполнена
        if self.image_processor.calibration_factor == 1.0:
            QMessageBox.information(self, "Внимание", "Пожалуйста, выполните калибровку перед определением слоев.")
            return
            
        try:
            self.statusBar().showMessage("Идет определение слоев... Это может занять некоторое время.")
            QApplication.processEvents()

            gray_image = cv2.cvtColor(self.current_tire.image, cv2.COLOR_BGR2GRAY)
            processed_gray = self.image_processor.preprocess_image(gray_image)

            layer_info = self.image_processor.determine_layers(processed_gray)
            
            self.detected_layers = layer_info 
            
            self.update_layers_list()
            self.display_image(self.current_tire.image)
            self.statusBar().showMessage(layer_info.get("message", "Слои определены."))

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось определить слои: {e}")

    def update_layers_list(self):
        """Обновление списка найденных слоев в UI"""
        self.layers_list.clear()
        if not self.detected_layers:
            return
        
        # 'layers' - это теперь словарь, который нужно итерировать
        layers_data = self.detected_layers.get("layers", {})
        
        for layer_name, layer_data in layers_data.items():
            pixel_count = layer_data.get("pixel_count", 0)
            # Убедимся, что area_sq_mm это строка
            area_str = layer_data.get("area_sq_mm", "0.00 мм²")
            item_text = f"{layer_name}: {pixel_count} пикс., S ≈ {area_str}"
            self.layers_list.addItem(item_text)

    def start_measurement(self):
        """Начало процесса измерения"""
        QMessageBox.information(self, "Измерение", 
                                "Выберите тип измерения с помощью кнопок на панели.")
        
    def open_calibration_dialog(self):
        """Открытие диалогового окна калибровки"""
        if self.current_tire is None or self.current_tire.image is None:
            QMessageBox.warning(self, "Внимание", "Сначала загрузите изображение.")
            return

        # Получаем текущее отображаемое изображение
        pixmap = self.image_label.pixmap()
        if not pixmap:
            QMessageBox.warning(self, "Внимание", "Нет изображения для калибровки.")
            return

        dialog = CalibrationDialog(pixmap, self)
        dialog.calibration_done.connect(self.set_calibration_factor)
        dialog.exec()

    def set_calibration_factor(self, factor: float):
        """Установка коэффициента калибровки"""
        self.measurement_system.set_calibration_factor(factor)
        self.statusBar().showMessage(f"Калибровка установлена: {factor:.4f} мм/пиксель")
        # Активируем кнопку определения слоев после успешной калибровки
        self.detect_layers_button.setEnabled(True)
            
    def save_report(self):
        """Сохранение отчета в форматах JSON и PNG."""
        if self.current_tire is None:
            QMessageBox.warning(self, "Ошибка", "Нет данных для сохранения.")
            return

        # Предлагаем имя файла по умолчанию
        default_filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить отчет",
            default_filename,
            "JSON Files (*.json);;All Files (*)"
        )

        if not file_name:
            return # Пользователь отменил сохранение

        # 1. Подготовка данных для JSON
        base_points_data = {name: {"x": p.x, "y": p.y, "desc": p.description} 
                            for name, p in self.base_points.items()}
        
        measurements_data = []
        for meas in self.measurement_system.get_measurement_history():
            measurements_data.append({
                "type": meas['type'],
                "result_mm": meas['result'],
                "points": [{"name": p.name, "x": p.x, "y": p.y} for p in meas['points']]
            })

        report_data = {
            "report_date": datetime.now().isoformat(),
            "tire_info": {
                "type": self.current_tire.type,
                "model": self.current_tire.model,
                "size": self.current_tire.size
            },
            "calibration_factor": self.image_processor.calibration_factor,
            "base_points": base_points_data,
            "measurements": measurements_data
        }

        # 2. Сохранение JSON файла
        try:
            with open(file_name, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка сохранения", f"Не удалось сохранить JSON файл: {e}")
            return
            
        # 3. Сохранение аннотированного изображения
        image_path = file_name.replace('.json', '_annotated.png')
        pixmap = self.image_label.pixmap()
        if pixmap:
            if not pixmap.save(image_path, "PNG"):
                QMessageBox.warning(self, "Ошибка сохранения", f"Не удалось сохранить файл изображения: {image_path}")
        
        self.statusBar().showMessage(f"Отчет сохранен в {file_name}")
        QMessageBox.information(self, "Успех", "Отчет успешно сохранен.")

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec()) 