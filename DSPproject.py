import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from mido import Message, MidiFile, MidiTrack
import tempfile

# Title
st.title("🎵 Music Note Extractor & MIDI Generator")

# Upload audio file
uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file is not None:
    # Load audio
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        filename = tmp_file.name

    y, sr = librosa.load(filename)

    # Plot waveform
    st.subheader("Waveform")
    plt.figure(figsize=(10, 3))
    plt.plot(y)
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    st.pyplot(plt)

    # STFT & Spectrogram
    D = np.abs(librosa.stft(y))
    frequencies = librosa.fft_frequencies(sr=sr)
    times = librosa.frames_to_time(np.arange(D.shape[1]), sr=sr)

    st.subheader("Spectrogram")
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
                             sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    st.pyplot(plt)

    # Map frequencies to notes
    def freq_to_note(freq):
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        if freq == 0:
            return None
        midi_note = int(round(69 + 12 * np.log2(freq / 440.0)))
        octave = midi_note // 12 - 1
        note_index = midi_note % 12
        return f"{note_names[note_index]}{octave}"

    notes = []
    for i in range(D.shape[1]):
        index = np.argmax(D[:, i])
        freq = frequencies[index]
        note = freq_to_note(freq)
        if note is not None:
            notes.append(note)

    st.subheader("Detected Notes (first 50 frames)")
    st.write(notes[:50])

    # Create MIDI file
    note_to_midi = {
        'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
        'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
    }

    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    duration = 480

    for note in notes:
        name = note[:-1]
        octave = int(note[-1])
        midi_number = note_to_midi[name] + (octave + 1) * 12
        track.append(Message('note_on', note=midi_number, velocity=64, time=0))
        track.append(Message('note_off', note=midi_number, velocity=64, time=duration))

    midi_file_path = "extracted_song.mid"
    mid.save(midi_file_path)

    # Download link
    st.subheader("Download MIDI file")
    with open(midi_file_path, "rb") as f:
        st.download_button("Download MIDI", f, file_name="extracted_song.mid", mime="audio/midi")

    # Optional: download notes as text
    st.subheader("Download Notes as Text")
    with open("all_notes.txt", "w") as f:
        for note in notes:
            f.write(note + "\n")
    with open("all_notes.txt", "rb") as f:
        st.download_button("Download Notes", f, file_name="all_notes.txt", mime="text/plain")
