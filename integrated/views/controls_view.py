from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QComboBox, 
    QGroupBox
)
from PySide6.QtCore import Qt, Signal
from models.configuration_model import ConfigurationModel


class ControlsView(QWidget):

    algorithm_selected = Signal(str)
    
    def __init__(self, configuration_model: ConfigurationModel, parent=None):
        super().__init__(parent)
        self._configuration_model = configuration_model
        self._init_ui()
        self._load_algorithms()
    
    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Title
        title = QLabel("Configuration")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;")
        layout.addWidget(title)
        
        # Algorithm selection group
        algorithm_group = QGroupBox("Select Algorithm")
        algorithm_layout = QVBoxLayout()
        
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItem("None", "")
        self.algorithm_combo.currentTextChanged.connect(self._on_algorithm_changed)
        algorithm_layout.addWidget(QLabel("Algorithm:"))
        algorithm_layout.addWidget(self.algorithm_combo)
        
        algorithm_group.setLayout(algorithm_layout)
        layout.addWidget(algorithm_group)
        
        # Spacer
        layout.addStretch()
        
        self.setLayout(layout)
    
    def _load_algorithms(self):
        """Load algorithms from the configuration model."""
        algorithm_names = self._configuration_model.available_algorithms
        for name in algorithm_names:
            self.algorithm_combo.addItem(name, name)
    
    def _on_algorithm_changed(self, text: str):
        data = self.algorithm_combo.currentData()
        self.algorithm_selected.emit(data if data else text)
    
    def set_algorithms(self, algorithm_names: list[str]):
        current_selection = self.algorithm_combo.currentData()
        self.algorithm_combo.clear()
        self.algorithm_combo.addItem("None", "")
        
        for name in algorithm_names:
            self.algorithm_combo.addItem(name, name)
        
        # Restore previous selection if it still exists
        if current_selection and current_selection in algorithm_names:
            index = self.algorithm_combo.findData(current_selection)
            if index >= 0:
                self.algorithm_combo.setCurrentIndex(index)
    
    def set_selected_algorithm(self, algorithm_name: str):
        index = self.algorithm_combo.findData(algorithm_name)
        if index >= 0:
            self.algorithm_combo.setCurrentIndex(index)
        elif algorithm_name == "":
            self.algorithm_combo.setCurrentIndex(0)

