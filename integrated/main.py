import sys
from PySide6.QtWidgets import QApplication
from controllers.app_controller import AppController


def main():
    app = QApplication(sys.argv)
        
    # Create the application controller
    controller = AppController()
    controller.start()
    
    # Run the event loop
    exit_code = app.exec()
    
    try:
        controller.stop()
    except Exception as e:
        print(f"Error closing the application: {e}")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

