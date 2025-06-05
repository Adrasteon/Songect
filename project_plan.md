# Songect Desktop Application - Project Plan

**Objective:** Create a cross-platform desktop application using PySide (Qt for Python) that leverages the existing Songect Python backend to provide users with music analysis features, tablature generation, and editing capabilities.

**Date:** June 4, 2025

---

## Phase 1: Project Setup & Foundation (1-2 Weeks)

1.  **Development Environment Setup:**
    *   **Task:** Ensure Python environment is set up with PySide6 (recommended for latest Qt features) and other necessary project dependencies.
    *   **Code Requirements:**
        *   Update/create `requirements.txt` or use a virtual environment.
        *   `pip install PySide6`
    *   **Notes:** Consider using a virtual environment (e.g., `venv`, `conda`).

2.  **Basic Application Window:**
    *   **Task:** Create the main application window with a basic layout (menu bar, status bar, central area).
    *   **Code Requirements (Python - PySide):**
        *   Main application class inheriting from `QMainWindow`.
        *   Setup of menu bar (`QMenuBar`), status bar (`QStatusBar`).
        *   Placeholder for the central widget area.
    *   **Files:** `main_ui.py` (or similar)

3.  **Project Structure for UI:**
    *   **Task:** Organize UI-related Python files.
    *   **Code Requirements:** Create directories if needed (e.g., `ui/widgets/`, `ui/dialogs/`).

4.  **Initial Packaging Test:**
    *   **Task:** Perform an early test of packaging a basic "Hello World" PySide app using PyInstaller or cx_Freeze.
    *   **Code Requirements (Shell - pwsh.exe):**
        ```powershell
        # Example with PyInstaller
        pyinstaller --onefile --windowed --name SongectApp main_ui.py
        ```
    *   **Notes:** This helps identify packaging issues early.

---

## Phase 2: Core Backend Integration (2-3 Weeks)

1.  **Backend Abstraction Layer:**
    *   **Task:** Create a Python class/module that acts as an interface between the UI and the `AITabTranscription` class from `python/aitabs.py`.
    *   **Code Requirements (Python):**
        *   A class (`BackendHandler`) that can:
            *   Load and initialize `AITabTranscription` with a configuration.
            *   Have a method to trigger `transcribe(wav_fp, device)`.
    *   **Files:** `backend_handler.py`

2.  **Configuration Management for UI:**
    *   **Task:** Implement a way for the UI to manage configurations needed by the backend (e.g., model paths if not bundled, default processing options).
    *   **Code Requirements (Python - PySide):**
        *   Dialog for settings (`QDialog`).
        *   Logic to save/load settings (e.g., using `QSettings` or a JSON file).
    *   **Files:** `settings_dialog.py`, `config_manager.py`

3.  **Asynchronous Task Execution:**
    *   **Task:** Implement running the `transcribe` method in a worker thread to keep the UI responsive.
    *   **Code Requirements (Python - PySide):**
        *   Use `QThread` and signals/slots for communication.
        *   A worker class that takes the audio file path and emits signals for progress and results (or errors).
    *   **Files:** `worker.py` (within `backend_handler.py` or separate)

4.  **Basic Progress Display:**
    *   **Task:** Show progress to the user during analysis.
    *   **Code Requirements (Python - PySide):**
        *   `QProgressBar` in the UI.
        *   Update progress bar based on signals from the worker thread.

---

## Phase 3: UI for File Handling & Main Controls (1-2 Weeks)

1.  **Audio File Input:**
    *   **Task:** Allow users to select an audio file.
    *   **Code Requirements (Python - PySide):**
        *   Menu action ("Open File") or button.
        *   `QFileDialog.getOpenFileName()` to select WAV or other supported audio files.

2.  **Main Control Panel:**
    *   **Task:** Create a panel with buttons to start analysis, potentially stop, and access settings.
    *   **Code Requirements (Python - PySide):**
        *   `QPushButton` widgets.
        *   Layouts (`QVBoxLayout`, `QHBoxLayout`).

3.  **Audio Playback (Optional - Basic):**
    *   **Task:** Basic playback of the loaded audio file.
    *   **Code Requirements (Python - PySide):**
        *   `QMediaPlayer` (from `QtMultimedia`) for playback.
        *   Basic play/pause/stop buttons.
    *   **Notes:** This can be enhanced later.

---

## Phase 4: UI for Displaying Analysis Results (3-4 Weeks)

1.  **Tabbed Interface for Results:**
    *   **Task:** Use a `QTabWidget` to organize different analysis outputs.
    *   **Code Requirements (Python - PySide):**
        *   `QTabWidget` as the central display area.

2.  **Display Chords & Key:**
    *   **Task:** Create a tab to display detected chords (e.g., in a sequence) and the overall song key.
    *   **Code Requirements (Python - PySide):**
        *   `QTextEdit` or a custom widget to display chord sequences.
        *   Labels for song key.

3.  **Display Beats & Tempo:**
    *   **Task:** Create a tab to display BPM and visualize beat/downbeat markers (e.g., on a timeline or alongside lyrics/chords).
    *   **Code Requirements (Python - PySide):**
        *   Labels for BPM.
        *   Potentially a custom drawing widget (`QWidget` with `paintEvent`) or a `QGraphicsView` for beat visualization.

4.  **Display Music Structure (Segments):**
    *   **Task:** Create a tab to show segment labels (intro, verse, chorus) and their timings.
    *   **Code Requirements (Python - PySide):**
        *   `QListWidget` or `QTableWidget` to list segments and times.
        *   Visualization on a timeline.

5.  **Display Melody (Pitch Contour):**
    *   **Task:** Create a tab to visualize the detected pitch contour of the melody.
    *   **Code Requirements (Python - PySide):**
        *   Custom drawing widget or integration with a plotting library (e.g., `Matplotlib` via Qt integration, or `pyqtgraph`).

6.  **Display Lyrics (Initial Placeholder & Future Integration):**
    *   **Task:** Create a tab for lyrics. Initially, it might just show the raw text if alignment is not yet complete.
    *   **Code Requirements (Python - PySide):**
        *   `QTextEdit` or `QLabel` for lyrics.
        *   Later, this will be enhanced for time-aligned display.

7.  **Source Separation Interface:**
    *   **Task:** Allow users to listen to or save the separated audio stems (vocals, bass, drums, other).
    *   **Code Requirements (Python - PySide):**
        *   Buttons for each stem to play (using `QMediaPlayer`).
        *   Buttons to trigger "Save As..." dialogs for each stem.

---

## Phase 5: Lyrics Module Implementation & UI Integration (3-4 Weeks)

1.  **Implement `get_lyrics` in `python/utils.py`:**
    *   **Task:** Integrate Whisper for transcription and a Wav2Vec2 model for word-level alignment as discussed previously.
    *   **Code Requirements (Python - in `python/utils.py`):**
        *   Use `openai-whisper` library for transcription.
        *   Use `transformers` library for Wav2Vec2 alignment model.
        *   Process alignment output into a usable format (list of words with start/end times, and the `lyrics_matrix` for `PitchNet`).
    *   **Dependencies:** `openai-whisper`, `transformers`, `ffmpeg` (system dependency).

2.  **Time-Aligned Lyrics Display:**
    *   **Task:** Update the Lyrics UI tab to display lyrics synchronized with audio playback, highlighting the current word.
    *   **Code Requirements (Python - PySide):**
        *   Custom widget or enhanced `QTextEdit` that can highlight text based on `QMediaPlayer`'s positionChanged signal and the word timings.

---

## Phase 6: Tablature Display & Editing (4-6 Weeks - Iterative)

1.  **Basic Tablature Rendering:**
    *   **Task:** Develop a component to render a basic version of the AI-generated tabs (chord charts, simplified staff lines).
    *   **Code Requirements (Python - PySide):**
        *   This is complex. Could start with a rich text display (`QTextEdit` with HTML formatting) or a `QGraphicsView` for custom drawing.
        *   Logic to convert backend output (chords, beats, structure) into a visual tab representation.

2.  **Tablature Data Model:**
    *   **Task:** Define a data structure to hold the tablature information in a way that supports editing.
    *   **Code Requirements (Python):**
        *   Custom Python classes representing measures, beats, notes, chords on the tab.

3.  **Chord Editing UI:**
    *   **Task:** Allow users to click on a chord in the tab and change it (e.g., via a dropdown or text input).
    *   **Code Requirements (Python - PySide):**
        *   Event handling on the tab display.
        *   Widgets for chord selection.

4.  **Rhythm Editing UI (Simplified):**
    *   **Task:** Allow basic rhythm adjustments (e.g., changing note durations if applicable, or beat associations).
    *   **Code Requirements (Python - PySide):**
        *   Depends heavily on the tab representation.

5.  **Lyrics Editing on Tab:**
    *   **Task:** Allow users to edit the lyrics associated with specific parts of the tab.
    *   **Code Requirements (Python - PySide):**
        *   Text input fields linked to tab sections.

---

## Phase 7: Additional Features & Polish (2-3 Weeks)

1.  **Speed and Pitch Adjustment UI:**
    *   **Task:** Implement UI controls (sliders, input fields) for audio speed and pitch adjustment.
    *   **Code Requirements (Python - PySide & Backend):**
        *   UI elements.
        *   Backend functions for these adjustments (likely need to be added or exposed from `python/audio.py` or similar).

2.  **Comprehensive Error Handling & User Feedback:**
    *   **Task:** Implement user-friendly error messages and feedback for all operations.
    *   **Code Requirements (Python - PySide):**
        *   Use `QMessageBox` for errors and warnings.
        *   Update status bar messages.

3.  **UI Styling and Theming (Optional):**
    *   **Task:** Apply custom styles or themes using Qt Style Sheets (QSS) or by customizing widget palettes.
    *   **Code Requirements (QSS or Python - PySide):**
        *   `.qss` files or programmatic styling.

4.  **Help/About Section:**
    *   **Task:** Create "Help" and "About" dialogs.
    *   **Code Requirements (Python - PySide):**
        *   `QDialog` with static information.

---

## Phase 8: Testing & Packaging (2-3 Weeks)

1.  **User Acceptance Testing (UAT):**
    *   **Task:** Thorough testing of all features by target users (or internal team).
    *   **Notes:** Collect feedback and iterate.

2.  **Bug Fixing and Performance Optimization:**
    *   **Task:** Address issues found during testing. Optimize slow parts of the UI or backend interaction.

3.  **Final Packaging for Distribution:**
    *   **Task:** Create installers/packages for Windows, and potentially macOS/Linux.
    *   **Code Requirements (Shell - pwsh.exe, and other tools):**
        *   Refine PyInstaller spec file (e.g., to include assets, icons, version info).
        *   Consider tools like Inno Setup (Windows) or `dmgbuild` (macOS) for creating installers.
    *   **Notes:** Ensure all dependencies (Python runtime, Qt libraries, AI models, ffmpeg if not expecting system install) are correctly bundled or handled.

---

**General Code Requirements & Best Practices:**

*   **Modular Design:** Keep UI components, backend logic, and utility functions in separate, well-defined modules.
*   **MVC/MVP Pattern:** Consider applying a design pattern like Model-View-Controller or Model-View-Presenter to separate concerns within the UI code.
*   **Comments & Documentation:** Write clear comments and docstrings for all new UI code.
*   **Version Control:** Use Git consistently.
*   **Cross-Platform Testing:** Regularly test on target platforms (Windows, macOS, Linux) throughout development if cross-platform is a hard requirement.

This plan is a high-level overview and timelines are estimates. Each phase, especially tablature editing, can be broken down further into smaller tasks.