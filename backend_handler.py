from python.aitabs import AITabTranscription
# Placeholder for other necessary imports, e.g., for configuration

class BackendHandler:
    def __init__(self, config=None):
        """
        Initializes the BackendHandler.
        Loads and initializes AITabTranscription with a configuration.
        """
        self.config = config
        self.transcriber = None
        self._initialize_transcriber()

    def _initialize_transcriber(self):
        """
        Loads and initializes the AITabTranscription model.
        This might involve loading model weights, setting up devices, etc.
        based on the self.config.
        """
        # Placeholder: Actual initialization logic will depend on AITabTranscription
        # and the structure of your configuration.
        try:
            # Example: self.transcriber = AITabTranscription(config=self.config)
            # For now, let's assume AITabTranscription can be instantiated without a config
            # or with a default one if config is None.
            # Pass the stored config to AITabTranscription
            if self.config is None:
                # This case should ideally be handled by ensuring ConfigManager always returns a valid default config,
                # or AITabTranscription should have its own internal defaults if config is None.
                # For now, raising an error or logging a warning might be appropriate if config is essential.
                print("Warning: BackendHandler has no config; AITabTranscription might not initialize correctly.")
                # Depending on AITabTranscription's design, it might raise an error here or use defaults.
                # If AITabTranscription strictly requires a config, this will fail, which is what we're seeing.
            self.transcriber = AITabTranscription(config=self.config)
            print("AITabTranscription initialized successfully.")
        except Exception as e:
            print(f"Error initializing AITabTranscription: {e}")
            self.transcriber = None

    def transcribe_audio(self, wav_fp, device=None):
        """
        Triggers the transcription process for the given audio file.

        Args:
            wav_fp (str): The file path to the WAV audio file.
            device (str, optional): The device to use for transcription (e.g., 'cpu', 'cuda').
                                    Defaults to None, which might mean AITabTranscription
                                    uses a default device.

        Returns:
            Transcription results, or None if an error occurs or transcriber is not initialized.
        """
        if not self.transcriber:
            print("Error: Transcriber not initialized.")
            return None

        try:
            # Determine the device to use for transcription
            effective_device = device
            if effective_device is None:
                if self.config and 'default_device' in self.config:
                    effective_device = self.config['default_device']
                else:
                    effective_device = 'cpu' # Fallback if not in config
            
            print(f"Starting transcription for {wav_fp} on device: {effective_device}")
            # Actual call to the transcribe method of AITabTranscription
            results = self.transcriber.transcribe(wav_fp, device=effective_device)
            print(f"Transcription process completed for {wav_fp}.")
            return results
        except Exception as e:
            print(f"Error during transcription: {e}")
            return None

if __name__ == '__main__':
    # Example usage (for testing purposes)
    # Create a dummy config if needed by AITabTranscription
    dummy_config = {} # Replace with actual configuration if required

    handler = BackendHandler(config=dummy_config)

    if handler.transcriber:
        # Create a dummy wav file path for testing
        # In a real scenario, this would be a path to an actual audio file.
        dummy_wav_file = "path/to/your/test_audio.wav"
        
        # Test the transcribe_audio method
        # You might need to ensure a dummy audio file exists at dummy_wav_file
        # or mock the AITabTranscription().transcribe() call for this test.
        print(f"Attempting to transcribe: {dummy_wav_file}")
        # transcription_output = handler.transcribe_audio(dummy_wav_file, device='cpu')
        # if transcription_output:
        #     print(f"Transcription Output: {transcription_output}")
        # else:
        #     print("Transcription failed or no output.")
        print("Example usage: To run, provide a real .wav file and uncomment calls.")
    else:
        print("BackendHandler could not be initialized with a transcriber.")

