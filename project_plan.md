\
<!-- filepath: c:\\Users\\marti\\MusicTranscribe\\Songect\\project_plan.md -->
# Songect Desktop Application - Project Plan

**Objective:** Create a cross-platform desktop application using PySide (Qt for Python) that leverages the existing Songect Python backend to provide users with music analysis features, tablature generation, and editing capabilities.

**Date:** June 4, 2025
**Last Updated:** June 5, 2025

---

## Phase 1: Project Setup & Foundation (1-2 Weeks) - [COMPLETED]

1.  **Development Environment Setup:** - [COMPLETED]
    *   **Task:** Ensure Python environment is set up with PySide6 (recommended for latest Qt features) and other necessary project dependencies.
    *   **Code Requirements:**
        *   Update/create `requirements.txt` or use a virtual environment. - [COMPLETED] (`requirements.txt` includes PySide6)
        *   `pip install PySide6` - [COMPLETED]
    *   **Notes:** Consider using a virtual environment (e.g., `venv`, `conda`). - [COMPLETED] (`.venv` is used)

2.  **Basic Application Window:** - [COMPLETED]
    *   **Task:** Create the main application window with a basic layout (menu bar, status bar, central area).
    *   **Code Requirements (Python - PySide):**
        *   Main application class inheriting from `QMainWindow`. - [COMPLETED] (`main_ui.py:MainWindow`)
        *   Setup of menu bar (`QMenuBar`), status bar (`QStatusBar`). - [COMPLETED]
        *   Placeholder for the central widget area. - [COMPLETED] (Initially a QLabel, later replaced by QTabWidget)
    *   **Files:** `main_ui.py` (or similar) - [COMPLETED]

3.  **Project Structure for UI:** - [COMPLETED]
    *   **Task:** Organize UI-related Python files.
    *   **Code Requirements:** Create directories if needed (e.g., `ui/widgets/`, `ui/dialogs/`). - [COMPLETED] (`ui/dialogs` created, `ui/widgets` exists)

4.  **Initial Packaging Test:** - [COMPLETED]
    *   **Task:** Perform an early test of packaging a basic "Hello World" PySide app using PyInstaller or cx_Freeze.
    *   **Code Requirements (Shell - pwsh.exe):**
        ```powershell
        # Example with PyInstaller
        pyinstaller --onefile --windowed --name SongectApp main_ui.py
        ```
    *   **Notes:** This helps identify packaging issues early. - [COMPLETED] (Indicated by `SongectApp.spec` and `build/` directory)

---

## Phase 2: Core Backend Integration (2-3 Weeks) - [COMPLETED]

1.  **Backend Abstraction Layer:** - [COMPLETED]
    *   **Task:** Create a Python class/module that acts as an interface between the UI and the `AITabTranscription` class from `python/aitabs.py`.
    *   **Code Requirements (Python):**
        *   A class (`BackendHandler`) that can:
            *   Load and initialize `AITabTranscription` with a configuration. - [COMPLETED] (Basic structure in `backend_handler.py`)
            *   Have a method to trigger `transcribe(wav_fp, device)`. - [COMPLETED] (Basic structure in `backend_handler.py`)
    *   **Files:** `backend_handler.py` - [COMPLETED]
    *   **Actuals:** `backend_handler.py` created with `BackendHandler` class. Assumes `AITabTranscription` can be initialized and `transcribe` called; specific internal logic of these methods within `AITabTranscription` is pre-existing or to be filled by the user.

2.  **Configuration Management for UI:** - [COMPLETED]
    *   **Task:** Implement a way for the UI to manage configurations needed by the backend (e.g., model paths if not bundled, default processing options).
    *   **Code Requirements (Python - PySide):**
        *   Dialog for settings (`QDialog`). - [COMPLETED] (`settings_dialog.py:SettingsDialog`)
        *   Logic to save/load settings (e.g., using `QSettings` or a JSON file). - [COMPLETED] (`config_manager.py:ConfigManager` using JSON)
    *   **Files:** `settings_dialog.py`, `config_manager.py` - [COMPLETED]
    *   **Actuals:** `ConfigManager` saves to `app_config.json` in a user-specific directory. `SettingsDialog` provides UI for example settings.

3.  **Asynchronous Task Execution:** - [COMPLETED]
    *   **Task:** Implement running the `transcribe` method in a worker thread to keep the UI responsive.
    *   **Code Requirements (Python - PySide):**
        *   Use `QThread` and signals/slots for communication. - [COMPLETED]
        *   A worker class that takes the audio file path and emits signals for progress and results (or errors). - [COMPLETED] (`worker.py:TranscriptionWorker`)
    *   **Files:** `worker.py` (within `backend_handler.py` or separate) - [COMPLETED] (Created as `worker.py`)
    *   **Actuals:** `TranscriptionWorker` emits `success`, `error`, and `progress` signals.

4.  **Basic Progress Display:** - [COMPLETED]
    *   **Task:** Show progress to the user during analysis.
    *   **Code Requirements (Python - PySide):**
        *   `QProgressBar` in the UI. - [COMPLETED]
        *   Update progress bar based on signals from the worker thread. - [COMPLETED]
    *   **Actuals:** `QProgressBar` added to the status bar in `main_ui.py` and connected to `TranscriptionWorker`'s `progress` signal.

---

## Phase 3: UI for File Handling & Main Controls (1-2 Weeks) - [COMPLETED]

1.  **Audio File Input:** - [COMPLETED]
    *   **Task:** Allow users to select an audio file.
    *   **Code Requirements (Python - PySide):**
        *   Menu action ("Open File") or button. - [COMPLETED] (Both menu and button in `main_ui.py`)
        *   `QFileDialog.getOpenFileName()` to select WAV or other supported audio files. - [COMPLETED] (Supports WAV, MP3, All files)

2.  **Main Control Panel:** - [COMPLETED]
    *   **Task:** Create a panel with buttons to start analysis, potentially stop, and access settings.
    *   **Code Requirements (Python - PySide):**
        *   `QPushButton` widgets. - [COMPLETED]
        *   Layouts (`QVBoxLayout`, `QHBoxLayout`). - [COMPLETED]
    *   **Actuals:** Control panel added to `main_ui.py` with "Open Audio File...", "Settings...", and "Stop Transcription" buttons.

3.  **Audio Playback (Optional - Basic):** - [COMPLETED]
    *   **Task:** Basic playback of the loaded audio file.
    *   **Code Requirements (Python - PySide):**
        *   `QMediaPlayer` (from `QtMultimedia`) for playback. - [COMPLETED]
        *   Basic play/pause/stop buttons. - [COMPLETED]
    *   **Notes:** This can be enhanced later.
    *   **Actuals:** `QMediaPlayer` and `QAudioOutput` integrated into `main_ui.py`. "Play Audio", "Pause Audio", "Stop Audio" buttons added to the control panel and connected.

---

## Phase 4: UI for Displaying Analysis Results (3-4 Weeks) - [COMPLETED - Initial Display Logic]

1.  **Tabbed Interface for Results:** - [COMPLETED]
    *   **Task:** Use a `QTabWidget` to organize different analysis outputs.
    *   **Code Requirements (Python - PySide):**
        *   `QTabWidget` as the central display area. - [COMPLETED]
    *   **Actuals:** `QTabWidget` (`self.results_tab_widget`) implemented in `main_ui.py`, replacing the initial placeholder. Placeholder tabs for Summary, Chords & Key, Beats & Tempo, Structure, Melody, Lyrics, Source Separation, and Tablature created.

2.  **Display Chords & Key:** - [COMPLETED]
    *   **Task:** Create a tab to display detected chords (e.g., in a sequence) and the overall song key.
    *   **Code Requirements (Python - PySide):**
        *   `QTextEdit` or a custom widget to display chord sequences. - [COMPLETED] (`QTextEdit` `self.chords_display_area`)
        *   Labels for song key. - [COMPLETED] (`QLabel` `self.key_display_label`)
    *   **Actuals:** "Chords & Key" tab populated with a `QFormLayout` for the key and `QTextEdit` for chords. `handle_transcription_success` in `main_ui.py` updated to populate these based on assumed `results` keys (`'key'`, `'chords'`). `clear_analysis_results` also updated.

3.  **Display Beats & Tempo:** - [COMPLETED (Data display only)]
    *   **Task:** Create a tab to display BPM and visualize beat/downbeat markers (e.g., on a timeline or alongside lyrics/chords).
    *   **Code Requirements (Python - PySide):**
        *   Labels for BPM. - [COMPLETED] (`QLabel` `self.bpm_display_label`)
        *   Potentially a custom drawing widget (`QWidget` with `paintEvent`) or a `QGraphicsView` for beat visualization. - [PENDING - Visualization part]
    *   **Actuals:** "Beats & Tempo" tab populated with a `QFormLayout` for BPM and `QTextEdit` (`self.beats_display_area`) for beat timestamps. `handle_transcription_success` updated based on assumed `results` keys (`'bpm'`, `'beats'`). `clear_analysis_results` updated. Beat visualization is a placeholder `QLabel`.

4.  **Display Music Structure (Segments):** - [COMPLETED (Data display only)]
    *   **Task:** Create a tab to show segment labels (intro, verse, chorus) and their timings.
    *   **Code Requirements (Python - PySide):**
        *   `QListWidget` or `QTableWidget` to list segments and times. - [COMPLETED] (`QListWidget` `self.structure_display_list`)
        *   Visualization on a timeline. - [PENDING - Visualization part]
    *   **Actuals:** "Structure" tab populated with a `QListWidget`. `handle_transcription_success` updated based on assumed `results` key (`'segments'`). `clear_analysis_results` updated. Structure visualization is a placeholder `QLabel`.

5.  **Display Melody (Pitch Contour):** - [PARTIALLY COMPLETED - Placeholder UI only]
    *   **Task:** Create a tab to visualize the detected pitch contour of the melody.
    *   **Code Requirements (Python - PySide):**
        *   Custom drawing widget or integration with a plotting library (e.g., `Matplotlib` via Qt integration, or `pyqtgraph`). - [PENDING - Plotting library integration]
    *   **Actuals:** "Melody" tab has a placeholder `QLabel` (`self.melody_plot_placeholder_label`). `clear_analysis_results` and `handle_transcription_success` update this label's text. Actual plotting requires choosing and integrating a library. `matplotlib` and `pyqtgraph` are in `requirements.txt`.

6.  **Display Lyrics (Initial Placeholder & Future Integration):** - [COMPLETED - Initial Placeholder for raw text]
    *   **Task:** Create a tab for lyrics. Initially, it might just show the raw text if alignment is not yet complete.
    *   **Code Requirements (Python - PySide):**
        *   `QTextEdit` or `QLabel` for lyrics. - [COMPLETED] (`QTextEdit` `self.lyrics_display_area`)
        *   Later, this will be enhanced for time-aligned display. - [PARTIALLY COMPLETED - UI groundwork laid, see Phase 5, Task 2]
    *   **Actuals:** "Lyrics" tab has a `QTextEdit`. `handle_transcription_success` populates it with basic string/list lyrics data (assumed key `'lyrics'` or `'text'` from `get_lyrics` output). `clear_analysis_results` updated.

7.  **Source Separation Interface:** - [COMPLETED - Placeholder Interface]
    *   **Task:** Allow users to listen to or save the separated audio stems (vocals, bass, drums, other).
    *   **Code Requirements (Python - PySide):**
        *   Buttons for each stem to play (using `QMediaPlayer`). - [PENDING - Actual playback logic]
        *   Buttons to trigger "Save As..." dialogs for each stem. - [PENDING - Actual save logic]
    *   **Actuals:** "Source Separation" tab has placeholder Play/Save buttons for Vocals, Bass, Drums, Other, using `QGridLayout`. Placeholder methods `play_stem_placeholder` and `save_stem_placeholder` created. `handle_transcription_success` enables these buttons based on assumed `results` key (`'stems'`). `clear_analysis_results` updated.

---

## Phase 5: Lyrics Module Implementation & UI Integration (3-4 Weeks) - [PARTIALLY COMPLETED]

1.  **Implement `get_lyrics` in `python/utils.py`:** - [PARTIALLY COMPLETED - Whisper ASR with word timestamps implemented. Wav2Vec2 alignment structure with placeholder helpers added.]
    *   **Task:** Integrate Whisper for transcription and a Wav2Vec2 model for word-level alignment as discussed previously.
    *   **Code Requirements (Python - in `python/utils.py`):**
        *   Use `openai-whisper` library for transcription. - [COMPLETED]
        *   Use `transformers` library for Wav2Vec2 alignment model. - [PARTIALLY COMPLETED - Model loading and processing structure in place, core alignment algorithms in helper functions are placeholders.]
        *   Process alignment output into a usable format (list of words with start/end times, and the `lyrics_matrix` for `PitchNet`). - [PARTIALLY COMPLETED - Whisper provides word timestamps; Wav2Vec2 refinement for more precise timings is pending full implementation of helper functions.]
    *   **Dependencies:** `openai-whisper`, `transformers`, `ffmpeg` (system dependency). - [CONFIRMED in `requirements.txt` and `ffmpeg` setup discussed]
    *   **Actuals:** `python/utils.py` updated. `get_lyrics` now uses `openai-whisper` with `word_timestamps=True` to return a dictionary: `{"text": ..., "detailed_lyrics": [...]}`. The `detailed_lyrics` currently uses Whisper's timestamps. The structure for Wav2Vec2 alignment (including model loading, resampling, and calls to placeholder helper functions for `get_trellis`, `backtrack`, `merge_repeats_and_blanks`, `merge_tokens_to_words_from_alignment`) is in place. **Full implementation of these helper functions is a significant TODO.**

2.  **Time-Aligned Lyrics Display:** - [PARTIALLY COMPLETED - UI groundwork laid]
    *   **Task:** Update the Lyrics UI tab to display lyrics synchronized with audio playback, highlighting the current word.
    *   **Code Requirements (Python - PySide):**
        *   Custom widget or enhanced `QTextEdit` that can highlight text based on `QMediaPlayer`'s `positionChanged` signal and the word timings. - [IMPLEMENTED using `QTextEdit` and HTML for highlighting]
    *   **Actuals:** `main_ui.py` updated: `self.current_detailed_lyrics` stores timed lyrics. `QMediaPlayer.positionChanged` connected to `_update_lyrics_highlighting`. This method re-renders lyrics in `lyrics_display_area` using HTML to highlight the current word. Depends on backend providing `detailed_lyrics` in the results (currently from Whisper timestamps).

---

## Phase 6: Tablature Display & Editing (4-6 Weeks - Iterative) - [PARTIALLY COMPLETED - Placeholder UI only]

1.  **Basic Tablature Rendering:** - [PARTIALLY COMPLETED - Placeholder UI only]
    *   **Task:** Develop a component to render a basic version of the AI-generated tabs (chord charts, simplified staff lines).
    *   **Code Requirements (Python - PySide):**
        *   This is complex. Could start with a rich text display (`QTextEdit` with HTML formatting) or a `QGraphicsView` for custom drawing. - [QTextEdit added as placeholder]
        *   Logic to convert backend output (chords, beats, structure) into a visual tab representation. - [PENDING]
    *   **Actuals:** "Tablature" tab added to `main_ui.py` with a `QTextEdit` (`self.tablature_display_area`) using a monospaced font. `clear_analysis_results` and `handle_transcription_success` include placeholders for this tab. **Actual rendering logic is a major TODO.**

2.  **Tablature Data Model:** - [PENDING]
    *   **Task:** Define a data structure to hold the tablature information in a way that supports editing.
    *   **Code Requirements (Python):
        *   Custom Python classes representing measures, beats, notes, chords on the tab.

3.  **Chord Editing UI:** - [PENDING]
    *   **Task:** Allow users to click on a chord in the tab and change it (e.g., via a dropdown or text input).
    *   **Code Requirements (Python - PySide):
        *   Event handling on the tab display.
        *   Widgets for chord selection.

4.  **Rhythm Editing UI (Simplified):** - [PENDING]
    *   **Task:** Allow basic rhythm adjustments (e.g., changing note durations if applicable, or beat associations).
    *   **Code Requirements (Python - PySide):
        *   Depends heavily on the tab representation.

5.  **Lyrics Editing on Tab:** - [PENDING]
    *   **Task:** Allow users to edit the lyrics associated with specific parts of the tab.
    *   **Code Requirements (Python - PySide):
        *   Text input fields linked to tab sections.

---

## Phase 7: Additional Features & Polish (2-3 Weeks) - [PARTIALLY ADDRESSED]

1.  **Speed and Pitch Adjustment UI:** - [PENDING]
    *   **Task:** Implement UI controls (sliders, input fields) for audio speed and pitch adjustment.
    *   **Code Requirements (Python - PySide & Backend):
        *   UI elements.
        *   Backend functions for these adjustments (likely need to be added or exposed from `python/audio.py` or similar).

2.  **Comprehensive Error Handling & User Feedback:** - [PARTIALLY ADDRESSED]
    *   **Task:** Implement user-friendly error messages and feedback for all operations.
    *   **Code Requirements (Python - PySide):
        *   Use `QMessageBox` for errors and warnings. - [IMPLEMENTED for some cases]
        *   Update status bar messages. - [IMPLEMENTED for many actions]
    *   **Actuals:** `QMessageBox` used for transcription errors, media player errors, and some warnings. Status bar updated for various states. Further review needed for comprehensiveness.

3.  **UI Styling and Theming (Optional):** - [PENDING]
    *   **Task:** Apply custom styles or themes using Qt Style Sheets (QSS) or by customizing widget palettes.
    *   **Code Requirements (QSS or Python - PySide):
        *   `.qss` files or programmatic styling.

4.  **Help/About Section:** - [COMPLETED - Basic]
    *   **Task:** Create "Help" and "About" dialogs.
    *   **Code Requirements (Python - PySide):
        *   `QDialog` with static information.
    *   **Actuals:** Basic "About" dialog implemented using `QMessageBox.about()`. A more detailed "Help" dialog is pending.

---

## Phase 8: Testing & Packaging (2-3 Weeks) - [PENDING]

1.  **User Acceptance Testing (UAT):** - [PENDING]
    *   **Task:** Thorough testing of all features by target users (or internal team).
    *   **Notes:** Collect feedback and iterate.

2.  **Bug Fixing and Performance Optimization:** - [PENDING]
    *   **Task:** Address issues found during testing. Optimize slow parts of the UI or backend interaction.

3.  **Final Packaging for Distribution:** - [PENDING - Initial test done]
    *   **Task:** Create installers/packages for Windows, and potentially macOS/Linux.
    *   **Code Requirements (Shell - pwsh.exe, and other tools):
        *   Refine PyInstaller spec file (e.g., to include assets, icons, version info).
        *   Consider tools like Inno Setup (Windows) or `dmgbuild` (macOS) for creating installers.
    *   **Notes:** Ensure all dependencies (Python runtime, Qt libraries, AI models, ffmpeg if not expecting system install) are correctly bundled or handled. Initial test in Phase 1.4 provides a starting point.

---

**General Code Requirements & Best Practices:**

*   **Modular Design:** Keep UI components, backend logic, and utility functions in separate, well-defined modules. - [ADHERED TO]
*   **MVC/MVP Pattern:** Consider applying a design pattern like Model-View-Controller or Model-View-Presenter to separate concerns within the UI code. - [NOT EXPLICITLY APPLIED YET, but UI and backend logic are largely separate]
*   **Comments & Documentation:** Write clear comments and docstrings for all new UI code. - [PARTIALLY ADDRESSED - Basic comments and docstrings in generated code]
*   **Version Control:** Use Git consistently. - [ASSUMED - User is using Git based on previous prompts]
*   **Cross-Platform Testing:** Regularly test on target platforms (Windows, macOS, Linux) throughout development if cross-platform is a hard requirement. - [PENDING - Current development on Windows]

This plan is a high-level overview and timelines are estimates. Each phase, especially tablature editing, can be broken down further into smaller tasks.