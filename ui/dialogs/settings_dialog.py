from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QLineEdit, QFormLayout

class SettingsDialog(QDialog):
    def __init__(self, parent=None, config_manager=None):
        super().__init__(parent)
        self.setWindowTitle("Application Settings")
        self.config_manager = config_manager
        self.current_settings = self.config_manager.load_settings() if self.config_manager else {}

        self.layout = QVBoxLayout(self)

        self.form_layout = QFormLayout()

        # Example Setting: Model Path
        self.model_path_label = QLabel("Model Path:")
        self.model_path_input = QLineEdit()
        if self.current_settings.get("model_path"):
            self.model_path_input.setText(self.current_settings["model_path"])
        self.form_layout.addRow(self.model_path_label, self.model_path_input)

        # Example Setting: Default Device
        self.device_label = QLabel("Default Device (e.g., cpu, cuda):")
        self.device_input = QLineEdit()
        if self.current_settings.get("default_device"):
            self.device_input.setText(self.current_settings["default_device"])
        self.form_layout.addRow(self.device_label, self.device_input)

        self.layout.addLayout(self.form_layout)

        # Buttons
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_settings)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)

        self.button_layout = QVBoxLayout()
        self.button_layout.addWidget(self.save_button)
        self.button_layout.addWidget(self.cancel_button)
        self.layout.addLayout(self.button_layout)

        self.setLayout(self.layout)

    def save_settings(self):
        if self.config_manager:
            updated_settings = {
                "model_path": self.model_path_input.text(),
                "default_device": self.device_input.text()
                # Add other settings here
            }
            self.config_manager.save_settings(updated_settings)
        self.accept()

    def get_settings(self):
        # This method could be used if the dialog is meant to return settings
        # directly rather than saving them through a config_manager.
        return {
            "model_path": self.model_path_input.text(),
            "default_device": self.device_input.text()
        }

if __name__ == '__main__':
    # This is for testing the dialog independently
    from PySide6.QtWidgets import QApplication
    import sys

    # Dummy ConfigManager for testing
    class DummyConfigManager:
        def load_settings(self):
            print("DummyConfigManager: Loading settings")
            return {"model_path": "/path/to/models", "default_device": "cpu"}
        def save_settings(self, settings):
            print(f"DummyConfigManager: Saving settings: {settings}")

    app = QApplication(sys.argv)
    config_manager = DummyConfigManager()
    dialog = SettingsDialog(config_manager=config_manager)
    if dialog.exec():
        print("Settings dialog accepted")
        # settings = dialog.get_settings() # Or retrieve from config_manager
        # print("Saved Settings:", settings)
    else:
        print("Settings dialog canceled")
    sys.exit(app.exec())
