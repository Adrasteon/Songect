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
            "model_path": "", # Base path for models if not found elsewhere
            "default_device": "cpu", # Default device for PyTorch operations ('cpu' or 'cuda')
            
            # Settings for AITabTranscription based on its __init__
            "n_channel": 2, # Default number of audio channels (e.g., 2 for stereo)
            "sources": ["vocals", "bass", "drums", "other"], # Default source separation stems
            "sample_rate": 44100, # Default sample rate for processing
            
            "separate": { # Configuration for source separation model (e.g., UNet wrapper)
                "model_name": "unet", 
                "model_path": "C:/Users/YourUser/Path/To/Your/Models/separation_model.pth", # Placeholder
                "model": { # Parameters for the UNets constructor (which then passes to UNet instances)
                    # 'sources' and 'n_channel' will be added dynamically in aitabs.py
                    "conv_n_filters": [16, 32, 64, 128, 256, 512], # Example filter configuration for UNet
                    "down_activation": "ELU",
                    "up_activation": "ELU",
                    "down_dropouts": None, # Or a list of dropout rates, e.g., [0.0, 0.0, ...]
                    "up_dropouts": None    # Or a list of dropout rates
                },
                "spec": { # Spectrogram settings specific to separation if different from global
                    "n_fft": 4096,
                    "hop_length": 1024,
                    "n_time": 256 # Example, might be specific to model input
                }
            },
            "lyrics": { # Configuration for lyrics transcription and alignment
                "model_name": "lyric_net", # Example
                "model_path": "path/to/your/lyric_model.pth", # Placeholder
                # Add other lyrics-specific params if needed
            },
            "beat": { # Configuration for beat and tempo detection
                "model_name": "beat_net", 
                "model_path": "path/to/your/beat_model.pth", # Placeholder
                "model": { # Parameters for BeatNet constructor
                    "source": 3, 
                    "n_classes": 3, 
                    "weights": [0.4, 0.3, 0.3], 
                    "n_freq": 2048
                }
            },
            "chord": { # Configuration for chord detection
                "model_name": "chord_net", 
                "model_path": "path/to/your/chord_model.pth", # Placeholder
                "model": { # Parameters for ChordNet constructor
                    "n_freq": 2048,
                    "n_classes": 122, # Example number of chord classes
                    "n_group": 32,
                    "f_layers": 5,
                    "t_layers": 5,
                    "d_model": 512,
                    "n_head": 8,
                    "dim_feedforward": 2048,
                    "dropout": 0.1,
                    "activation": "relu"
                }
            },
            "segment": { # Configuration for structure segmentation
                "model_name": "segment_net", 
                "model_path": "path/to/your/segment_model.pth", # Placeholder
                "model": { # Parameters for SegmentNet (likely embedding + transformer)
                    "n_channel": 2, # For SegmentEmbeddings
                    "n_hidden": 128, # For SegmentEmbeddings
                    "d_model": 2048, # For SegmentEmbeddings & Transformer
                    "dropout": 0.1, # For SegmentEmbeddings & Transformer
                    "n_head": 8, # For Transformer
                    "dim_feedforward": 2048, # For Transformer
                    "num_encoder_layers": 6 # For Transformer
                }
            },
            "pitch": { # Configuration for pitch (melody) detection
                "model_name": "pitch_net", 
                "model_path": "C:/Users/YourUser/Path/To/Your/Models/pitch_model.pth", # Placeholder
                "model": { # Parameters for PitchNet (likely embedding + transformer)
                    "n_channel": 1, # For PitchEmbedding (usually mono vocal)
                    "d_model": 512, # For PitchEmbedding & Transformer
                    "n_hidden": 32, # For PitchEmbedding
                    "dropout": 0.1, # For PitchEmbedding & Transformer
                    "n_head": 8, # For Transformer
                    "dim_feedforward": 2048, # For Transformer
                    "num_encoder_layers": 6, # For Transformer
                    "num_decoder_layers": 6, # For Transformer (if applicable)
                    "n_freq_out": 360 # Example output dimension for pitch bins
                }
            },
            "spec": { # Global/default spectrogram settings
                "n_fft": 4096,             # Standard FFT size
                "hop_length": 1024,        # Standard hop length for STFT
                "win_length": 4096,        # Window length for STFT
                "pad": True,               # Added: Default padding for STFT (True for center=True)
                "n_mels": 256,             # Number of Mel bands, if Mel spectrograms are used
                "f_min": 0.0,              # Minimum frequency for Mel filters
                "f_max": None             # Maximum frequency for Mel filters (None uses sr/2)
            },
            "tempo": { # Configuration for tempo detection (e.g., using librosa.beat.beat_track)
                "hop_length": 512,         # Hop length for librosa's beat tracker
                # Add other tempo-specific params if needed
            }
            # Add other default settings here as needed by other parts of AITabTranscription or your app
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
        # Check against the actual config_file_path being managed by this instance
        if config_manager.config_file_path.endswith('test_app_config.json'): # only if it's the test one
            if os.path.exists(config_manager.config_dir) and not os.listdir(config_manager.config_dir):
                os.rmdir(config_manager.config_dir)
                print(f"Cleaned up test config directory: {config_manager.config_dir}")
            elif os.path.exists(os.path.join(os.getcwd(), 'test_app_config.json')): # if it fell back to cwd
                 os.remove(os.path.join(os.getcwd(), 'test_app_config.json'))
                 print(f"Cleaned up test config file from CWD: {os.path.join(os.getcwd(), 'test_app_config.json')}")

    except Exception as e:
        print(f"Error during cleanup: {e}")
