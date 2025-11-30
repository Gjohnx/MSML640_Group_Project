from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QComboBox, 
    QGroupBox
)
from PySide6.QtCore import Qt, Signal
from models.configuration_model import ConfigurationModel


class ControlsView(QWidget):

    detection_method_selected = Signal(str)
    
    def __init__(self, configuration_model: ConfigurationModel, parent=None):
        super().__init__(parent)
        self._configuration_model = configuration_model
        self._init_ui()
        self._load_detection_methods()
    
    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Title
        title = QLabel("Configuration")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;")
        layout.addWidget(title)
        
        # Detection method selection group
        detection_method_group = QGroupBox("Select Detection Method")
        detection_method_layout = QVBoxLayout()
        
        self.detection_method_combo = QComboBox()
        self.detection_method_combo.addItem("None", "")
        self.detection_method_combo.currentTextChanged.connect(self._on_detection_method_changed)
        detection_method_layout.addWidget(QLabel("Detection Method:"))
        detection_method_layout.addWidget(self.detection_method_combo)
        
        detection_method_group.setLayout(detection_method_layout)
        layout.addWidget(detection_method_group)
        
        # Spacer
        layout.addStretch()
        
        self.setLayout(layout)
    
    def _load_detection_methods(self):
        """Load detection methods from the configuration model."""
        detection_method_names = self._configuration_model.available_detection_methods
        for name in detection_method_names:
            self.detection_method_combo.addItem(name, name)
    
    def _on_detection_method_changed(self, text: str):
        data = self.detection_method_combo.currentData()
        self.detection_method_selected.emit(data if data else text)
    
    def set_detection_methods(self, detection_method_names: list[str]):
        current_selection = self.detection_method_combo.currentData()
        self.detection_method_combo.clear()
        self.detection_method_combo.addItem("None", "")
        
        for name in detection_method_names:
            self.detection_method_combo.addItem(name, name)
        
        # Restore previous selection if it still exists
        if current_selection and current_selection in detection_method_names:
            index = self.detection_method_combo.findData(current_selection)
            if index >= 0:
                self.detection_method_combo.setCurrentIndex(index)
    
    def set_selected_detection_method(self, detection_method_name: str):
        index = self.detection_method_combo.findData(detection_method_name)
        if index >= 0:
            self.detection_method_combo.setCurrentIndex(index)
        elif detection_method_name == "":
            self.detection_method_combo.setCurrentIndex(0)

