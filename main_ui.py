import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QMenuBar, QStatusBar, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QProgressBar, QPushButton, QFileDialog, QMessageBox, QTabWidget, QTextEdit, QFormLayout, QListWidget, QGridLayout
from PySide6.QtGui import QAction, QTextCursor # Added QTextCursor
from PySide6.QtCore import Qt, QUrl
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput # Added QMediaPlayer, QAudioOutput

# Assuming these files are in the same directory or accessible via PYTHONPATH
from backend_handler import BackendHandler
from config_manager import ConfigManager
from worker import TranscriptionWorker
from ui.dialogs.settings_dialog import SettingsDialog

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Songect Desktop Application")
        self.setGeometry(100, 100, 800, 600)  # x, y, width, height

        # Initialize ConfigManager and BackendHandler
        self.config_manager = ConfigManager()
        self.backend_handler = BackendHandler(config=self.config_manager.load_settings()) # Pass loaded config

        self.worker_thread = None # To hold the reference to the worker thread
        self.current_audio_file_path = None # To store the path of the loaded audio file
        self.current_detailed_lyrics = [] # To store detailed lyrics with timings
        self.current_highlighted_word_index = -1 # To track the currently highlighted word

        self._init_media_player() # Initialize media player
        self._create_menu_bar()
        self._create_status_bar()
        self._create_central_widget()

    def _init_media_player(self):
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput() # Required for QMediaPlayer in Qt6
        self.media_player.setAudioOutput(self.audio_output)
        # Connect signals for state changes, errors, etc.
        self.media_player.playbackStateChanged.connect(self.update_playback_button_states)
        self.media_player.errorOccurred.connect(self.handle_media_player_error)
        self.media_player.positionChanged.connect(self._update_lyrics_highlighting) # Connect positionChanged

    def _create_menu_bar(self):
        menu_bar = self.menuBar()

        # File Menu
        file_menu = menu_bar.addMenu("&File")

        open_action = QAction("&Open Audio File...", self)
        open_action.triggered.connect(self.open_file_dialog_and_start_transcription) # Connect to actual function
        file_menu.addAction(open_action)

        # Settings Action
        settings_action = QAction("&Settings...", self)
        settings_action.triggered.connect(self.open_settings_dialog)
        file_menu.addAction(settings_action)

        file_menu.addSeparator()

        exit_action = QAction("&Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Help Menu
        help_menu = menu_bar.addMenu("&Help")
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about_dialog) # Placeholder for future connection
        help_menu.addAction(about_action)

    def _create_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.progress_bar = QProgressBar(self.status_bar)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False) # Initially hidden
        self.status_bar.addPermanentWidget(self.progress_bar) # Add to the right side
        self.status_bar.showMessage("Ready")

    def _create_central_widget(self):
        # Main layout
        main_layout = QVBoxLayout()

        # Control Panel
        control_panel_widget = QWidget()
        control_panel_layout = QHBoxLayout(control_panel_widget)
        control_panel_layout.setContentsMargins(0, 0, 0, 0) # Remove margins if desired

        self.open_file_button = QPushButton("Open Audio File...")
        self.open_file_button.clicked.connect(self.open_file_dialog_and_start_transcription)
        control_panel_layout.addWidget(self.open_file_button)

        self.settings_button = QPushButton("Settings...")
        self.settings_button.clicked.connect(self.open_settings_dialog)
        control_panel_layout.addWidget(self.settings_button)

        self.stop_button = QPushButton("Stop Transcription")
        self.stop_button.clicked.connect(self.stop_transcription_task)
        self.stop_button.setEnabled(False) # Initially disabled
        control_panel_layout.addWidget(self.stop_button)

        control_panel_layout.addSpacing(20) # Add some space

        # Playback Buttons
        self.play_button = QPushButton("Play Audio")
        self.play_button.clicked.connect(self.play_audio)
        self.play_button.setEnabled(False)
        control_panel_layout.addWidget(self.play_button)

        self.pause_button = QPushButton("Pause Audio")
        self.pause_button.clicked.connect(self.pause_audio)
        self.pause_button.setEnabled(False)
        control_panel_layout.addWidget(self.pause_button)

        self.stop_audio_button = QPushButton("Stop Audio")
        self.stop_audio_button.clicked.connect(self.stop_audio)
        self.stop_audio_button.setEnabled(False)
        control_panel_layout.addWidget(self.stop_audio_button)

        control_panel_layout.addStretch() # Add stretch to push buttons to the left

        main_layout.addWidget(control_panel_widget)
        
        # Tab Widget for results
        self.results_tab_widget = QTabWidget()
        main_layout.addWidget(self.results_tab_widget)

        # Summary Tab
        self.tab_summary = QWidget()
        self.results_tab_widget.addTab(self.tab_summary, "Summary")
        summary_layout = QVBoxLayout(self.tab_summary)
        self.summary_label = QLabel("Analysis results will be summarized here.") # Made it an instance member
        summary_layout.addWidget(self.summary_label)
        summary_layout.setAlignment(Qt.AlignTop)

        # Chords & Key Tab
        self.tab_chords_key = QWidget()
        self.results_tab_widget.addTab(self.tab_chords_key, "Chords & Key")
        chords_key_layout = QVBoxLayout(self.tab_chords_key)
        chords_key_layout.setAlignment(Qt.AlignTop)

        # Key Display
        key_form_layout = QFormLayout()
        self.key_label_title = QLabel("Detected Song Key:")
        self.key_display_label = QLabel("N/A") # To display the actual key
        key_form_layout.addRow(self.key_label_title, self.key_display_label)
        chords_key_layout.addLayout(key_form_layout)

        # Chords Display
        chords_display_label_title = QLabel("Detected Chords:")
        chords_key_layout.addWidget(chords_display_label_title)
        self.chords_display_area = QTextEdit()
        self.chords_display_area.setReadOnly(True)
        self.chords_display_area.setPlaceholderText("Chord sequence will appear here...")
        chords_key_layout.addWidget(self.chords_display_area)

        # Beats & Tempo Tab
        self.tab_beats_tempo = QWidget()
        self.results_tab_widget.addTab(self.tab_beats_tempo, "Beats & Tempo")
        beats_tempo_layout = QVBoxLayout(self.tab_beats_tempo)
        beats_tempo_layout.setAlignment(Qt.AlignTop)

        # BPM Display
        bpm_form_layout = QFormLayout()
        self.bpm_label_title = QLabel("Detected BPM:")
        self.bpm_display_label = QLabel("N/A") # To display the actual BPM
        bpm_form_layout.addRow(self.bpm_label_title, self.bpm_display_label)
        beats_tempo_layout.addLayout(bpm_form_layout)

        # Beats Display
        beats_display_label_title = QLabel("Detected Beats (timestamps):")
        beats_tempo_layout.addWidget(beats_display_label_title)
        self.beats_display_area = QTextEdit()
        self.beats_display_area.setReadOnly(True)
        self.beats_display_area.setPlaceholderText("Beat timestamps will appear here...")
        beats_tempo_layout.addWidget(self.beats_display_area)
        # Placeholder for beat visualization - to be added later
        beats_tempo_layout.addWidget(QLabel("Beat visualization will be here."))

        # Structure Tab
        self.tab_structure = QWidget()
        self.results_tab_widget.addTab(self.tab_structure, "Structure")
        structure_layout = QVBoxLayout(self.tab_structure)
        structure_layout.setAlignment(Qt.AlignTop)

        structure_display_label_title = QLabel("Song Structure (Segments):")
        structure_layout.addWidget(structure_display_label_title)
        self.structure_display_list = QListWidget()
        # self.structure_display_list.setAlternatingRowColors(True) # Optional: for better readability
        structure_layout.addWidget(self.structure_display_list)
        # Placeholder for structure visualization - to be added later
        structure_layout.addWidget(QLabel("Structure visualization will be here."))

        # Melody Tab
        self.tab_melody = QWidget()
        self.results_tab_widget.addTab(self.tab_melody, "Melody")
        melody_layout = QVBoxLayout(self.tab_melody)
        melody_layout.setAlignment(Qt.AlignTop)
        self.melody_plot_placeholder_label = QLabel("Melody pitch contour plot will be displayed here.")
        melody_layout.addWidget(self.melody_plot_placeholder_label)
        # Actual plotting widget will replace or augment this label later

        # Lyrics Tab
        self.tab_lyrics = QWidget()
        self.results_tab_widget.addTab(self.tab_lyrics, "Lyrics")
        lyrics_layout = QVBoxLayout(self.tab_lyrics)
        lyrics_layout.setAlignment(Qt.AlignTop)
        lyrics_display_label_title = QLabel("Transcribed Lyrics (Time-Aligned):") # Updated title
        lyrics_layout.addWidget(lyrics_display_label_title)
        self.lyrics_display_area = QTextEdit()
        self.lyrics_display_area.setReadOnly(True)
        self.lyrics_display_area.setPlaceholderText("Transcribed and time-aligned lyrics will appear here...")
        lyrics_layout.addWidget(self.lyrics_display_area)

        # Stems Tab
        self.tab_stems = QWidget()
        self.results_tab_widget.addTab(self.tab_stems, "Source Separation")
        stems_layout = QVBoxLayout(self.tab_stems) # Main layout for the tab
        stems_layout.setAlignment(Qt.AlignTop)

        stems_title_label = QLabel("Separated Audio Stems:")
        stems_layout.addWidget(stems_title_label)

        # Using QGridLayout for a more organized button layout
        stems_grid_layout = QGridLayout()
        stems_layout.addLayout(stems_grid_layout)

        stem_names = ["Vocals", "Bass", "Drums", "Other"]
        self.stem_buttons = {} # To store references to buttons for easy access

        for i, stem_name in enumerate(stem_names):
            stem_label = QLabel(f"{stem_name}:")
            stems_grid_layout.addWidget(stem_label, i, 0)

            play_button = QPushButton(f"Play {stem_name}")
            play_button.setEnabled(False)
            play_button.clicked.connect(lambda checked=False, s=stem_name: self.play_stem_placeholder(s))
            stems_grid_layout.addWidget(play_button, i, 1)
            self.stem_buttons[f"play_{stem_name.lower()}"] = play_button

            save_button = QPushButton(f"Save {stem_name}...")
            save_button.setEnabled(False)
            save_button.clicked.connect(lambda checked=False, s=stem_name: self.save_stem_placeholder(s))
            stems_grid_layout.addWidget(save_button, i, 2)
            self.stem_buttons[f"save_{stem_name.lower()}"] = save_button
        
        stems_layout.addStretch() # Add a stretch to push controls to the top

        # Tablature Tab (New for Phase 6)
        self.tab_tablature = QWidget()
        self.results_tab_widget.addTab(self.tab_tablature, "Tablature")
        tablature_layout = QVBoxLayout(self.tab_tablature)
        tablature_layout.setAlignment(Qt.AlignTop)
        tablature_display_label_title = QLabel("Generated Tablature/Chord Chart:")
        tablature_layout.addWidget(tablature_display_label_title)
        self.tablature_display_area = QTextEdit()
        self.tablature_display_area.setReadOnly(True) # Initially read-only
        self.tablature_display_area.setPlaceholderText("Generated tablature or chord chart will appear here...")
        self.tablature_display_area.setFontFamily("Courier New") # Monospaced font is good for tabs
        tablature_layout.addWidget(self.tablature_display_area)

        # self.placeholder_label = QLabel("Songect - Main Content Area") # Removed placeholder
        # self.placeholder_label.setStyleSheet("font-size: 16px; color: grey;")
        # self.placeholder_label.setAlignment(Qt.AlignCenter)
        # main_layout.addWidget(self.placeholder_label) # Removed placeholder

        # main_layout.addStretch() # Removed to allow tab widget to fill space
        
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def open_settings_dialog(self):
        dialog = SettingsDialog(self, self.config_manager)
        if dialog.exec():
            self.status_bar.showMessage("Settings saved.")
            # Re-initialize backend_handler if settings that affect it were changed
            # This is a simple way; a more robust way might involve specific signals/slots
            # or checking which settings changed.
            self.backend_handler = BackendHandler(config=self.config_manager.load_settings())
            print("Settings updated. BackendHandler re-initialized.")
        else:
            self.status_bar.showMessage("Settings dialog canceled.")

    def open_file_dialog_and_start_transcription(self):
        self.status_bar.showMessage("Opening file dialog...")
        file_filter = "WAV files (*.wav);;MP3 files (*.mp3);;All files (*.*)" # Added MP3
        audio_fp, _ = QFileDialog.getOpenFileName(self, "Open Audio File", "", file_filter)

        if audio_fp:
            self.current_audio_file_path = audio_fp # Store the path
            self.status_bar.showMessage(f"Selected file: {self.current_audio_file_path}")
            self.load_audio_for_playback(self.current_audio_file_path)
            # Decide if transcription should start automatically or via a separate button
            # For now, let's assume we still want to start it after loading for playback.
            if self.current_audio_file_path.lower().endswith('.wav'): # Assuming transcribe_audio needs WAV
                self.start_transcription_task(self.current_audio_file_path)
            else:
                QMessageBox.information(self, "Transcription Info", "Transcription currently only supports WAV files. Playback is available.")
                self.summary_label.setText(f"Loaded for playback: {self.current_audio_file_path}\\nTranscription only available for .wav files.")
                self.clear_analysis_results() # Clear previous results
        else:
            self.status_bar.showMessage("File selection canceled.")

    def load_audio_for_playback(self, file_path):
        if self.media_player.playbackState() != QMediaPlayer.PlaybackState.StoppedState:
            self.media_player.stop()
        
        self.media_player.setSource(QUrl.fromLocalFile(file_path))
        self.status_bar.showMessage(f"Loaded for playback: {file_path}", 3000)
        self.play_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.stop_audio_button.setEnabled(True)
        self.clear_analysis_results() # Clear results when a new file is loaded

    def clear_analysis_results(self):
        """Clears all analysis display areas."""
        self.summary_label.setText("Analysis results will be summarized here.")
        self.key_display_label.setText("N/A")
        self.chords_display_area.setText("")
        self.bpm_display_label.setText("N/A")
        self.beats_display_area.setText("")
        self.structure_display_list.clear()
        self.melody_plot_placeholder_label.setText("Melody pitch contour plot will be displayed here.")
        self.lyrics_display_area.setText("")
        self.lyrics_display_area.setPlaceholderText("Transcribed and time-aligned lyrics will appear here...")
        self.current_detailed_lyrics = [] 
        self.current_highlighted_word_index = -1
        self.tablature_display_area.setText("") # Clear tablature display
        self.tablature_display_area.setPlaceholderText("Generated tablature or chord chart will appear here...")
        # Clear/disable stem buttons
        for stem_name in ["Vocals", "Bass", "Drums", "Other"]:
            if f"play_{stem_name.lower()}" in self.stem_buttons:
                self.stem_buttons[f"play_{stem_name.lower()}"].setEnabled(False)
            if f"save_{stem_name.lower()}" in self.stem_buttons:
                self.stem_buttons[f"save_{stem_name.lower()}"].setEnabled(False)
        print("Analysis results display cleared.")

    def _update_lyrics_highlighting(self, position_ms):
        """Updates lyrics highlighting based on media player position."""
        if not self.current_detailed_lyrics or self.media_player.playbackState() != QMediaPlayer.PlaybackState.PlayingState:
            # If no detailed lyrics or not playing, ensure no highlighting or reset if needed
            # If lyrics were previously highlighted, might want to reset to plain text here
            # For now, do nothing if not playing or no lyrics
            return

        current_time_s = position_ms / 1000.0
        new_highlighted_word_index = -1

        # Find the current word
        for i, lyric_data in enumerate(self.current_detailed_lyrics):
            if lyric_data['start_time'] <= current_time_s < lyric_data['end_time']:
                new_highlighted_word_index = i
                break
        
        if new_highlighted_word_index != self.current_highlighted_word_index:
            self.current_highlighted_word_index = new_highlighted_word_index
            html_lyrics = ""
            for i, lyric_data in enumerate(self.current_detailed_lyrics):
                word = lyric_data['word']
                # Escape HTML special characters in the word itself
                escaped_word = word.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                if i == self.current_highlighted_word_index:
                    html_lyrics += f'<span style="background-color: yellow; font-weight: bold;">{escaped_word}</span> '
                else:
                    html_lyrics += f'{escaped_word} '
            
            # Preserve scroll position as much as possible
            # This is tricky with HTML re-rendering. A more robust solution might involve
            # custom text document manipulation or a QListView/QListWidget with custom delegates.
            # For QTextEdit, one common approach is to move cursor to highlighted word.
            current_cursor = self.lyrics_display_area.textCursor()
            self.lyrics_display_area.setHtml(html_lyrics.strip())

            if self.current_highlighted_word_index != -1:
                # Attempt to scroll to the highlighted word
                # This is a simplified scroll attempt. It might not be perfect.
                doc = self.lyrics_display_area.document()
                cursor = QTextCursor(doc)
                char_pos = 0
                for i in range(self.current_highlighted_word_index + 1):
                    if i < len(self.current_detailed_lyrics):
                        # Add length of word and a space, except for the last word
                        char_pos += len(self.current_detailed_lyrics[i]['word']) + (1 if i < len(self.current_detailed_lyrics) -1 else 0)
                
                # Move cursor to the approximate start of the highlighted word
                # This needs refinement to be accurate with HTML tags
                # A simpler way for now, just ensure cursor is visible if it was moved by setHtml
                # cursor.setPosition(char_pos - len(self.current_detailed_lyrics[self.current_highlighted_word_index]['word']))
                # self.lyrics_display_area.setTextCursor(cursor)
                # self.lyrics_display_area.ensureCursorVisible() 
                pass # Scrolling logic needs more robust implementation

    def play_audio(self):
        if not self.media_player.source().isEmpty():
            self.media_player.play()
            # Initial call to highlighting when play starts, if lyrics are loaded
            if self.current_detailed_lyrics:
                 self._update_lyrics_highlighting(self.media_player.position())
        else:
            QMessageBox.warning(self, "Playback Error", "No audio file loaded.")

    def pause_audio(self):
        self.media_player.pause()

    def stop_audio(self):
        self.media_player.stop()
        # self.play_button.setEnabled(True) # Re-enable play after stop
        # self.pause_button.setEnabled(False)
        # self.stop_audio_button.setEnabled(False) # Stop can be pressed again if loaded

    def update_playback_button_states(self, state):
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.play_button.setEnabled(False)
            self.pause_button.setEnabled(True)
            self.stop_audio_button.setEnabled(True)
        elif state == QMediaPlayer.PlaybackState.PausedState:
            self.play_button.setEnabled(True)
            self.pause_button.setEnabled(False)
            self.stop_audio_button.setEnabled(True)
        elif state == QMediaPlayer.PlaybackState.StoppedState:
            self.play_button.setEnabled(True if not self.media_player.source().isEmpty() else False)
            self.pause_button.setEnabled(False)
            self.stop_audio_button.setEnabled(False if self.media_player.source().isEmpty() else True)

    def handle_media_player_error(self, error, error_string=""):
        # The error argument is a QMediaPlayer.Error enum, error_string is its string representation
        QMessageBox.critical(self, "Media Player Error", f"Error: {error_string} (Code: {error})")
        self.play_button.setEnabled(False)
        self.pause_button.setEnabled(False)
        self.stop_audio_button.setEnabled(False)

    def start_transcription_task(self, wav_fp):
        if self.worker_thread and self.worker_thread.isRunning():
            QMessageBox.warning(self, "Busy", "A transcription task is already in progress.")
            return

        self.status_bar.showMessage(f"Starting transcription for: {wav_fp}...")
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.stop_button.setEnabled(True) # Enable stop button

        # Get device from config or use a default
        current_config = self.config_manager.load_settings()
        device = current_config.get("default_device", "cpu") # Default to 'cpu' if not set

        self.worker_thread = TranscriptionWorker(self.backend_handler, wav_fp, device)
        
        self.worker_thread.success.connect(self.handle_transcription_success)
        self.worker_thread.error.connect(self.handle_transcription_error)
        self.worker_thread.progress.connect(self.update_progress_bar)
        
        self.worker_thread.finished.connect(self.on_worker_finished) # Clean up thread object

        self.worker_thread.start()

    def stop_transcription_task(self):
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.stop() # Call the worker's stop method
            self.status_bar.showMessage("Attempting to stop transcription...", 3000)
            # The stop_button will be disabled in on_worker_finished or error/success handlers
        else:
            self.status_bar.showMessage("No active transcription to stop.", 3000)

    def handle_transcription_success(self, results):
        self.status_bar.showMessage("Transcription successful!", 5000) # Show for 5 seconds
        self.progress_bar.setVisible(False)
        self.stop_button.setEnabled(False)
        print("Transcription Results:", results)
        QMessageBox.information(self, "Success", f"Transcription successful! Results received.")

        # Update Summary Tab
        self.summary_label.setText(f"Transcription complete for: {self.current_audio_file_path}\\nResults (preview): {str(results)[:200]}...")

        # Update Chords & Key Tab
        # Assuming results is a dictionary and might contain 'key' and 'chords'
        # Adjust these keys based on your actual backend_handler output structure
        detected_key = results.get('key', 'N/A')
        self.key_display_label.setText(detected_key)

        detected_chords = results.get('chords', []) # Assuming chords is a list of strings
        if isinstance(detected_chords, list):
            self.chords_display_area.setText("\n".join(detected_chords) if detected_chords else "No chords detected.")
        elif isinstance(detected_chords, str): # If it's already a formatted string
            self.chords_display_area.setText(detected_chords if detected_chords else "No chords detected.")
        else:
            self.chords_display_area.setText("Chord data format not recognized.")

        # Update Beats & Tempo Tab
        # Assuming results might contain 'bpm' and 'beats' (e.g., list of timestamps)
        detected_bpm = results.get('bpm', 'N/A')
        self.bpm_display_label.setText(str(detected_bpm)) # Ensure it's a string

        detected_beats = results.get('beats', []) # Assuming beats is a list of timestamps
        if isinstance(detected_beats, list):
            # Format beats for display, e.g., one timestamp per line
            beats_text = "\n".join(map(str, detected_beats)) if detected_beats else "No beat data detected."
            self.beats_display_area.setText(beats_text)
        elif isinstance(detected_beats, str): # If it's already a formatted string
            self.beats_display_area.setText(detected_beats if detected_beats else "No beat data detected.")
        else:
            self.beats_display_area.setText("Beat data format not recognized.")

        # Update Structure Tab
        # Assuming results might contain 'structure' or 'segments'
        # (e.g., a list of dicts: [{'label': 'intro', 'start': 0.0, 'end': 10.5}, ...])
        self.structure_display_list.clear()
        detected_segments = results.get('segments', []) 
        if isinstance(detected_segments, list) and detected_segments:
            for segment in detected_segments:
                if isinstance(segment, dict):
                    label = segment.get('label', 'N/A')
                    start_time = segment.get('start_time', 'N/A') # Or 'start' depending on backend
                    end_time = segment.get('end_time', 'N/A')     # Or 'end' depending on backend
                    item_text = f"{label}: {start_time}s - {end_time}s"
                    self.structure_display_list.addItem(item_text)
                else:
                    self.structure_display_list.addItem(str(segment)) # Fallback for simple list of strings
            if not detected_segments:
                 self.structure_display_list.addItem("No segment data detected.")
        elif not detected_segments: # If the list was empty from the start
            self.structure_display_list.addItem("No segment data detected.")
        else:
            self.structure_display_list.addItem("Segment data format not recognized.")

        # Update Melody Tab (Placeholder for now)
        # Actual plotting will require a plotting widget and data extraction
        self.melody_plot_placeholder_label.setText("Melody data received. Plotting to be implemented.")
        # Example: if 'melody_data' in results:
        # self.melody_plot_widget.plot(results['melody_data'])

        # Update Lyrics Tab
        # Assuming results might contain 'lyrics' as a string or a list of lines
        detected_lyrics = results.get('lyrics', '')
        if isinstance(detected_lyrics, list):
            self.lyrics_display_area.setText("\n".join(detected_lyrics) if detected_lyrics else "No lyrics detected.")
        elif isinstance(detected_lyrics, str):
            self.lyrics_display_area.setText(detected_lyrics if detected_lyrics else "No lyrics detected.")
        else:
            self.lyrics_display_area.setText("Lyrics data format not recognized.")

        # Update Stems Tab (Enable buttons if corresponding data is present)
        # This is a placeholder logic. You'll need to check for actual stem data in results.
        # For example, if results contain a dictionary like: results['stems'] = {'vocals_path': 'path/to/vocals.wav'}
        available_stems = results.get('stems', {}) # Assuming stems are in a dict within results
        for stem_name_lower in ["vocals", "bass", "drums", "other"]:
            stem_name_title = stem_name_lower.capitalize()
            # Check if data/path for this stem exists in the results
            # This condition needs to be adapted to how your backend provides stem info
            stem_data_exists = (stem_name_lower + "_path" in available_stems) or (stem_name_lower in available_stems) 
            
            play_button = self.stem_buttons.get(f"play_{stem_name_lower}")
            if play_button:
                play_button.setEnabled(stem_data_exists)
            
            save_button = self.stem_buttons.get(f"save_{stem_name_lower}")
            if save_button:
                save_button.setEnabled(stem_data_exists)

        # Update Tablature Tab (Placeholder for now)
        # TODO: Implement logic to generate basic tablature/chord chart from results
        # For example, using results['chords'], results['segments'], results['detailed_lyrics']
        self.tablature_display_area.setText("Tablature generation based on analysis results to be implemented here.\n"
                                            f"Received Chords: {results.get('chords', 'N/A')}\n"
                                            f"Received Key: {results.get('key', 'N/A')}")

        # TODO: Populate other tabs with their respective results (if any remaining)

    def handle_transcription_error(self, error_message):
        self.status_bar.showMessage(f"Transcription error: {error_message}", 5000)
        self.progress_bar.setVisible(False)
        self.stop_button.setEnabled(False)
        QMessageBox.critical(self, "Transcription Error", error_message)
        self.summary_label.setText(f"Transcription failed for: {self.current_audio_file_path}\\nError: {error_message}")
        self.clear_analysis_results() # Clear results on error

    def update_progress_bar(self, percentage):
        self.progress_bar.setValue(percentage)

    def on_worker_finished(self):
        self.status_bar.showMessage("Transcription task finished.", 3000)
        self.progress_bar.setVisible(False) # Ensure it's hidden
        self.stop_button.setEnabled(False) # Disable stop button
        # self.worker_thread.deleteLater() # Schedule the thread object for deletion
        self.worker_thread = None # Clear the reference

    def show_about_dialog(self):
        self.status_bar.showMessage("About dialog triggered...")
        QMessageBox.about(self, "About Songect",
                          "Songect Desktop Application\\nVersion 0.1 (Dev)\\n\\n"
                          "Music analysis and tablature generation.")
        # Add about dialog logic here

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
