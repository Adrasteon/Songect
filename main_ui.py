import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QMenuBar, QStatusBar, QLabel, QVBoxLayout, QWidget
from PySide6.QtGui import QAction

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Songect Desktop Application")
        self.setGeometry(100, 100, 800, 600)  # x, y, width, height

        self._create_menu_bar()
        self._create_status_bar()
        self._create_central_widget()

    def _create_menu_bar(self):
        menu_bar = self.menuBar()

        # File Menu
        file_menu = menu_bar.addMenu("&File")

        open_action = QAction("&Open Audio File...", self)
        # open_action.triggered.connect(self.open_file_dialog) # Placeholder for future connection
        file_menu.addAction(open_action)

        exit_action = QAction("&Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Help Menu
        help_menu = menu_bar.addMenu("&Help")
        about_action = QAction("&About", self)
        # about_action.triggered.connect(self.show_about_dialog) # Placeholder for future connection
        help_menu.addAction(about_action)

    def _create_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def _create_central_widget(self):
        # Placeholder central widget
        main_layout = QVBoxLayout()
        placeholder_label = QLabel("Songect - Main Content Area (Placeholder)")
        placeholder_label.setStyleSheet("font-size: 16px; color: grey;")
        main_layout.addWidget(placeholder_label)
        
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    # Placeholder methods for actions (to be implemented later)
    # def open_file_dialog(self):
    #     self.status_bar.showMessage("Open file dialog triggered...")
    #     # Add file dialog logic here

    # def show_about_dialog(self):
    #     self.status_bar.showMessage("About dialog triggered...")
    #     # Add about dialog logic here

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
