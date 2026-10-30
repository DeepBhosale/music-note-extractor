# Music Note Extractor

> A Python-based application for extracting musical notes from audio files using Librosa and other libraries.

## 📌 Overview
The Music Note Extractor is a Python application that analyzes audio files to extract and display musical notes. It utilizes Librosa for audio processing, including loading, analyzing, and visualizing audio data. The application provides a user-friendly interface using Gradio to upload audio files and view the extracted notes and visualizations.

## ✨ Key Features
- **Audio Note Extraction**: Converts audio signals into musical notes.
- **Waveform Visualization**: Displays the waveform of the audio file.
- **Spectrogram Visualization**: Shows the frequency spectrum of the audio.
- **Real-time Note Display**: Updates the displayed notes in real-time as the audio plays.

## 🛠️ Tech Stack & Dependencies
- **Python**: The primary programming language.
- **Librosa**: For audio processing and feature extraction.
- **NumPy**: For numerical operations.
- **Matplotlib**: For data visualization.
- **Mido**: For MIDI file handling.
- **Soundfile**: For reading and writing audio files.

## 📂 Project Architecture
```
music-note-extractor/
├── app.py
└── requirements.txt
```

- **app.py**: Contains the main application logic, including note extraction, visualization, and user interface.
- **requirements.txt**: Lists the project dependencies.

## 🚀 Getting Started
### Prerequisites
- Python 3.9+
- Librosa, NumPy, Matplotlib, Mido, and Soundfile libraries (install via `pip install -r requirements.txt`)

### Installation
```bash
# Clone the repository
git clone https://github.com/DeepBhosale/music-note-extractor.git
cd music-note-extractor

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
# Execute the real entrypoint script
python app.py
```

## 🛡️ Autonomous Quality & Verification
Codebase audited and verified by **NOIR Autonomous Quality Engine**.