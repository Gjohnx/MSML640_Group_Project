from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QGridLayout, QHBoxLayout, QSizePolicy
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
import numpy as np
import cv2

class ProcessedView(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(5, 5, 5, 5)

        # Previous Layout Title
        title = QLabel("Processed Feed")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;")
        self.main_layout.addWidget(title)

        # Default single view UI for other methods
        self.single_label = QLabel("No processed feed")
        self.single_label.setAlignment(Qt.AlignCenter)
        self.single_label.setStyleSheet("background-color: black; border: 1px solid gray;")
        self.single_label.setMinimumSize(320, 240)
        self.single_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.main_layout.addWidget(self.single_label)

        # 4-view processed feed UI for Processed Comparison Method
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout()
        self.grid_layout.setSpacing(5)

        def create_panel(title_text):
            container = QWidget()
            layout = QVBoxLayout()
            layout.setContentsMargins(2, 2, 2, 2)

            title = QLabel(title_text)
            title.setAlignment(Qt.AlignCenter)
            title.setStyleSheet("font-size: 12px; font-weight: bold; padding: 2px;")

            label = QLabel()
            label.setAlignment(Qt.AlignCenter)
            label.setFixedSize(300, 300)
            label.setStyleSheet("background-color: black; border: 1px solid gray;")

            layout.addWidget(title)
            layout.addWidget(label)
            container.setLayout(layout)
            return container, label

        self.cropped_container, self.cropped_label = create_panel("1. Cropped Cube Face")
        self.corner_container, self.corner_label = create_panel("2. Corner Detection")
        self.warp_container, self.warp_label = create_panel("3. Perspective Wrapped")
        self.final_container, self.final_label = create_panel("4. Processed Final Image")

        self.grid_layout.addWidget(self.cropped_container, 0, 0)
        self.grid_layout.addWidget(self.corner_container, 0, 1)
        self.grid_layout.addWidget(self.warp_container, 1, 0)
        self.grid_layout.addWidget(self.final_container, 1, 1)

        self.grid_widget.setLayout(self.grid_layout)
        grid_wrapper = QHBoxLayout()
        grid_wrapper.addStretch()
        grid_wrapper.addWidget(self.grid_widget)
        grid_wrapper.addStretch()

        self.main_layout.addLayout(grid_wrapper)
        self.grid_widget.hide()
        self.setLayout(self.main_layout)

    # single view display
    def display_frame(self, frame: np.ndarray):
        self.show_single_view()
        if frame is None:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, w * ch, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        pixmap = pixmap.scaled(self.single_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.single_label.setPixmap(pixmap)

    # 4-view display
    def display_processed_comparison(self, cropped, corner, warp, final):
        self.show_comparison_view()
        self._set_label_image(self.cropped_label, cropped)
    
    def _set_label_image(self, label, img):
        if img is None:
            label.clear()
            label.setText("None")
            return
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, w * ch, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        pix = pix.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pix)

    # This is the switch control
    def show_single_view(self):
        self.grid_widget.hide()
        self.single_label.show()
    def show_comparison_view(self):
        self.single_label.hide()
        self.grid_widget.show()
