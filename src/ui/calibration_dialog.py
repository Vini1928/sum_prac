import sys
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QMessageBox)
from PyQt6.QtCore import Qt, pyqtSignal, QPoint
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor
import numpy as np

class CalibrationDialog(QDialog):
    """Диалоговое окно для калибровки измерений"""
    
    # Сигнал, который передает калибровочный коэффициент
    calibration_done = pyqtSignal(float)

    def __init__(self, pixmap: QPixmap, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Калибровка")
        self.setMinimumSize(800, 600)
        
        self.pixmap = pixmap
        self.points = []
        self.pixel_distance = 0.0
        
        self._create_ui()

    def _create_ui(self):
        layout = QVBoxLayout(self)
        
        # Инструкция
        instruction_label = QLabel(
            "1. Введите известное расстояние в миллиметрах.\n"
            "2. Кликните на две точки на изображении, чтобы измерить расстояние в пикселях."
        )
        layout.addWidget(instruction_label)
        
        # Поле для ввода реального расстояния
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Реальное расстояние (мм):"))
        self.real_distance_input = QLineEdit()
        self.real_distance_input.setPlaceholderText("например, 10.0")
        input_layout.addWidget(self.real_distance_input)
        layout.addLayout(input_layout)
        
        # Область для отображения изображения
        self.image_label = QLabel()
        self.image_label.setPixmap(self.pixmap)
        self.image_label.setCursor(Qt.CursorShape.CrossCursor)
        self.image_label.mousePressEvent = self.image_clicked
        layout.addWidget(self.image_label)
        
        # Отображение информации
        self.info_label = QLabel("Выберите первую точку.")
        layout.addWidget(self.info_label)
        
        # Кнопки
        buttons_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept_calibration)
        self.ok_button.setEnabled(False)
        
        self.cancel_button = QPushButton("Отмена")
        self.cancel_button.clicked.connect(self.reject)
        
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.ok_button)
        buttons_layout.addWidget(self.cancel_button)
        layout.addLayout(buttons_layout)

    def image_clicked(self, event):
        """Обработка клика по изображению"""
        if len(self.points) >= 2:
            self.points = [] # Сбрасываем точки, если уже выбраны две
        
        pos = event.pos()
        self.points.append(pos)
        
        if len(self.points) == 1:
            self.info_label.setText("Выберите вторую точку.")
        elif len(self.points) == 2:
            self.calculate_pixel_distance()
            self.info_label.setText(f"Расстояние в пикселях: {self.pixel_distance:.2f}. Нажмите 'OK' для подтверждения.")
            self.ok_button.setEnabled(True)
            
        self.update_pixmap()

    def update_pixmap(self):
        """Обновление изображения с отмеченными точками"""
        pixmap = self.pixmap.copy()
        painter = QPainter(pixmap)
        pen = QPen(QColor("red"), 3)
        painter.setPen(pen)
        
        for point in self.points:
            painter.drawPoint(point)
            
        if len(self.points) == 2:
            pen.setStyle(Qt.PenStyle.DashLine)
            pen.setColor(QColor("blue"))
            painter.setPen(pen)
            painter.drawLine(self.points[0], self.points[1])
            
        painter.end()
        self.image_label.setPixmap(pixmap)

    def calculate_pixel_distance(self):
        """Вычисление расстояния между точками в пикселях"""
        p1 = self.points[0]
        p2 = self.points[1]
        self.pixel_distance = np.sqrt((p2.x() - p1.x())**2 + (p2.y() - p1.y())**2)

    def accept_calibration(self):
        """Подтверждение калибровки"""
        try:
            real_distance = float(self.real_distance_input.text())
            if real_distance <= 0:
                raise ValueError("Расстояние должно быть положительным числом.")
            if self.pixel_distance == 0:
                raise ValueError("Расстояние в пикселях не может быть равно нулю.")
                
            calibration_factor = real_distance / self.pixel_distance
            self.calibration_done.emit(calibration_factor)
            self.accept()
            
        except ValueError as e:
            QMessageBox.warning(self, "Ошибка ввода", f"Неверное значение: {e}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Произошла ошибка: {str(e)}") 