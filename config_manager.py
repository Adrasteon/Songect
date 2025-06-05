import json
import os

class ConfigManager:
    def __init__(self, config_file_name='app_config.json'):
        """
        Initializes the ConfigManager.

        Args:
            config_file_name (str): The name of the configuration file.
                                      This file will be stored in a user-specific config directory.
        """
        # Determine a user-specific directory for the config file
        # This is a common practice to avoid permission issues and keep user configs separate.
        if os.name == 'nt': # Windows
            app_data_dir = os.getenv('APPDATA')
            if not app_data_dir:
                app_data_dir = os.path.expanduser("~") # Fallback to home directory
            self.config_dir = os.path.join(app_data_dir, 'Songect')
        else: # macOS, Linux
            self.config_dir = os.path.join(os.path.expanduser("~"), '.config', 'Songect')

        if not os.path.exists(self.config_dir):
            try:
                os.makedirs(self.config_dir)
            except OSError as e:
                print(f"Error creating config directory {self.config_dir}: {e}")
                # Fallback to current directory if user-specific dir fails
                self.config_dir = os.getcwd()
        
        self.config_file_path = os.path.join(self.config_dir, config_file_name)
        self.default_settings = {
            "model_path": "",
            "default_device": "cpu",
            # Add other default settings here
        }

    def load_settings(self):
        """
        Loads settings from the configuration file.
        If the file doesn't exist or is invalid, returns default settings.
        """
        if not os.path.exists(self.config_file_path):
            print(f"Config file not found at {self.config_file_path}. Using default settings.")
            return self.default_settings.copy()
        try:
            with open(self.config_file_path, 'r') as f:
                settings = json.load(f)
                # Ensure all default keys are present
                for key, value in self.default_settings.items():
                    if key not in settings:
                        settings[key] = value
                return settings
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {self.config_file_path}. Using default settings.")
            return self.default_settings.copy()
        except Exception as e:
            print(f"Error loading settings from {self.config_file_path}: {e}. Using default settings.")
            return self.default_settings.copy()

    def save_settings(self, settings):
        """
        Saves the given settings to the configuration file.

        Args:
            settings (dict): The settings dictionary to save.
        """
        try:
            with open(self.config_file_path, 'w') as f:
                json.dump(settings, f, indent=4)
            print(f"Settings saved to {self.config_file_path}")
        except Exception as e:
            print(f"Error saving settings to {self.config_file_path}: {e}")

    def get_setting(self, key, default=None):
        """
        Retrieves a specific setting by key.

        Args:
            key (str): The key of the setting to retrieve.
            default: The value to return if the key is not found.

        Returns:
            The value of the setting, or the default value if not found.
        """
        settings = self.load_settings()
        return settings.get(key, default)

    def update_setting(self, key, value):
        """
        Updates a specific setting and saves the configuration.

        Args:
            key (str): The key of the setting to update.
            value: The new value for the setting.
        """
        settings = self.load_settings()
        settings[key] = value
        self.save_settings(settings)

if __name__ == '__main__':
    # Example Usage
    config_manager = ConfigManager(config_file_name='test_app_config.json')

    # Load settings (or defaults if file doesn't exist)
    current_settings = config_manager.load_settings()
    print(f"Initial/Loaded settings: {current_settings}")

    # Modify a setting
    current_settings['model_path'] = '/new/path/to/models'
    current_settings['new_setting'] = True 

    # Save settings
    config_manager.save_settings(current_settings)

    # Load them again to verify
    reloaded_settings = config_manager.load_settings()
    print(f"Reloaded settings: {reloaded_settings}")

    # Test get_setting
    print(f"Model path from get_setting: {config_manager.get_setting('model_path')}")
    print(f"Non-existent setting: {config_manager.get_setting('non_existent_key', 'default_val')}")

    # Test update_setting
    config_manager.update_setting('default_device', 'cuda:0')
    print(f"Device after update_setting: {config_manager.get_setting('default_device')}")

    # Clean up the test config file
    try:
        if os.path.exists(config_manager.config_file_path):
            os.remove(config_manager.config_file_path)
            print(f"Cleaned up test config file: {config_manager.config_file_path}")
        # Attempt to remove the directory if it was created for the test config and is empty
        if config_manager.config_file_name == 'test_app_config.json': # only if it's the test one
            if os.path.exists(config_manager.config_dir) and not os.listdir(config_manager.config_dir):
                os.rmdir(config_manager.config_dir)
                print(f"Cleaned up test config directory: {config_manager.config_dir}")
            elif os.path.exists(os.path.join(os.getcwd(), 'test_app_config.json')): # if it fell back to cwd
                 os.remove(os.path.join(os.getcwd(), 'test_app_config.json'))
                 print(f"Cleaned up test config file from CWD: {os.path.join(os.getcwd(), 'test_app_config.json')}")

    except Exception as e:
        print(f"Error during cleanup: {e}")
