from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
    QGroupBox, QPushButton, QMessageBox
)
from PySide6.QtCore import Qt, Signal


class ControlsView(QWidget):

    start_detection_clicked = Signal(str)
    reset_clicked = Signal()
    detect_clicked = Signal()
    resolve_clicked = Signal()  # New Signal for the Resolve action
    start_resolution_clicked = Signal(str)
    next_step_clicked = Signal()
    prev_step_clicked = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.faces = [] # Initialize faces list to prevent crashes if not set
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Title
        title = QLabel("Configuration")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;")
        layout.addWidget(title)

        # --- Detection Method Group ---
        detection_method_group = QGroupBox("Select Detection Method")
        detection_method_layout = QVBoxLayout()
        
        self.detection_method_combo = QComboBox()
        self.detection_method_combo.setEnabled(False)
        detection_method_layout.addWidget(QLabel("Detection Method:"))
        detection_method_layout.addWidget(self.detection_method_combo)
        
        detection_method_group.setLayout(detection_method_layout)
        layout.addWidget(detection_method_group)
        
        # Start Detection button
        self.start_detection_button = QPushButton("Start Detection")
        self.start_detection_button.setEnabled(False)
        self.start_detection_button.clicked.connect(self._on_start_detection_clicked)
        layout.addWidget(self.start_detection_button)

        # --- Button Row: Reset | Detect | Resolve ---
        detection_buttons_layout = QHBoxLayout()
        
        # 1. Reset
        self.reset_button = QPushButton("Reset")
        self.reset_button.setEnabled(False)
        self.reset_button.clicked.connect(self._on_reset_clicked)
        detection_buttons_layout.addWidget(self.reset_button)
        
        # 2. Detect
        self.detect_button = QPushButton("Detect")
        self.detect_button.setEnabled(False)
        self.detect_button.clicked.connect(self._on_detect_clicked)
        detection_buttons_layout.addWidget(self.detect_button)
        
        # 3. Resolve (The new button)
        self.resolve_button = QPushButton("Resolve")
        self.resolve_button.setEnabled(False)
        self.resolve_button.clicked.connect(self._on_resolve_clicked)
        detection_buttons_layout.addWidget(self.resolve_button)

        layout.addLayout(detection_buttons_layout)
        
        # --- Resolution Method Group ---
        resolution_method_group = QGroupBox("Select Resolution Method")
        resolution_method_layout = QVBoxLayout()
        
        self.resolution_method_combo = QComboBox()
        self.resolution_method_combo.setEnabled(False)
        resolution_method_layout.addWidget(QLabel("Resolution Method:"))
        resolution_method_layout.addWidget(self.resolution_method_combo)
        
        resolution_method_group.setLayout(resolution_method_layout)
        layout.addWidget(resolution_method_group)
        
        # Start Resolution button
        self.start_resolution_button = QPushButton("Start Resolution")
        self.start_resolution_button.setEnabled(False)
        self.start_resolution_button.clicked.connect(self._on_start_resolution_clicked)
        layout.addWidget(self.start_resolution_button)
        
        # --- Step Buttons ---
        step_buttons_layout = QHBoxLayout()
        
        self.prev_step_button = QPushButton("Prev Step")
        self.prev_step_button.setEnabled(False)
        self.prev_step_button.clicked.connect(self._on_prev_step_clicked)
        step_buttons_layout.addWidget(self.prev_step_button)
        
        self.next_step_button = QPushButton("Next Step")
        self.next_step_button.setEnabled(False)
        self.next_step_button.clicked.connect(self._on_next_step_clicked)
        step_buttons_layout.addWidget(self.next_step_button)
        
        layout.addLayout(step_buttons_layout)
        
        # Spacer
        layout.addStretch()
        
        self.setLayout(layout)
    
    # --- Methods ---

    def set_detection_methods(self, detection_method_names: list[str]):
        self.detection_method_combo.clear()
        for name in detection_method_names:
            self.detection_method_combo.addItem(name, name)
        self.detection_method_combo.setCurrentIndex(0)
        self.enable_detection_method()
        self.enable_start_detection()
    
    def _on_start_detection_clicked(self):
        selected_method = self.detection_method_combo.currentData()
        self.start_detection_clicked.emit(selected_method)
    
    def set_resolution_methods(self, resolution_method_names: list[str]):
        self.resolution_method_combo.clear()
        for name in resolution_method_names:
            self.resolution_method_combo.addItem(name, name)
        self.resolution_method_combo.setCurrentIndex(0)
    
    def _on_start_resolution_clicked(self):
        selected_method = self.resolution_method_combo.currentData()
        self.start_resolution_clicked.emit(selected_method)
    
    def _on_resolve_clicked(self):
        # Validate that we have exactly 6 faces
        if len(self.faces) == 6:
            # Validation passed: Emit signal to Parent to change status
            self.resolve_clicked.emit()
        else:
            # Validation failed: Show warning
            QMessageBox.warning(
                self, 
                "Incomplete Cube", 
                f"Only {len(self.faces)}/6 faces scanned. Please scan all 6 sides before resolving."
            )
    
    def _on_reset_clicked(self):
        self.reset_clicked.emit()
    
    def _on_detect_clicked(self):
        self.detect_clicked.emit()
    
    def _on_prev_step_clicked(self):
        self.prev_step_clicked.emit()
    
    def _on_next_step_clicked(self):
        self.next_step_clicked.emit()

    # --- Enable/Disable Helpers ---

    def enable_start_detection(self):
        self.start_detection_button.setEnabled(True)

    def disable_start_detection(self):
        self.start_detection_button.setEnabled(False)
    
    def disable_detection_method(self):
        self.detection_method_combo.setEnabled(False)
    
    def enable_detection_method(self):
        self.detection_method_combo.setEnabled(True)
    
    def enable_reset(self):
        self.reset_button.setEnabled(True)
    
    def disable_reset(self):
        self.reset_button.setEnabled(False)
    
    def enable_detect(self):
        self.detect_button.setEnabled(True)
    
    def disable_detect(self):
        self.detect_button.setEnabled(False)

    # Helper to control the new Resolve button
    def enable_resolve(self):
        self.resolve_button.setEnabled(True)

    def disable_resolve(self):
        self.resolve_button.setEnabled(False)
    
    def disable_resolution_method(self):
        self.resolution_method_combo.setEnabled(False)
    
    def enable_resolution_method(self):
        self.resolution_method_combo.setEnabled(True)
    
    def enable_start_resolution(self):
        self.start_resolution_button.setEnabled(True)
    
    def disable_start_resolution(self):
        self.start_resolution_button.setEnabled(False)
    
    def enable_prev_step(self):
        self.prev_step_button.setEnabled(True)
    
    def disable_prev_step(self):
        self.prev_step_button.setEnabled(False)
    
    def enable_next_step(self):
        self.next_step_button.setEnabled(True)

    def disable_next_step(self):
        self.next_step_button.setEnabled(False)