# app.py
import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from mido import Message, MidiFile, MidiTrack
import tempfile
import traceback
import os

st.set_page_config(page_title="Music Note Extractor", layout="wide")
st.title("🎵 Music Note Extractor & MIDI Generator")

uploaded_file = st.file_uploader("Upload a WAV file (try a short clip first)", type=["wav", "mp3"])

if uploaded_file is None:
    st.info("Upload a WAV or MP3 file to begin.")
    st.stop()

# Save upload to a temporary file
with tempfile.NamedTemporaryFile(suffix=os.path.splitext(uploaded_file.name)[1], delete=False) as tmp_file:
    tmp_file.write(uploaded_file.read())
    tmp_path = tmp_file.name

st.write(f"Saved uploaded file to temporary path: `{tmp_path}`")

# Helper to show exceptions both in-app and in logs
def show_exception(e: Exception):
    st.error(f"Error: {e}")
    tb = traceback.format_exc()
    print(tb)          # appears in Streamlit Cloud logs
    st.text_area("Traceback (for debugging)", tb, height=300)

# Load audio
try:
    # sr=22050 to downsample (less memory). If you want original use sr=None
    y, sr = librosa.load(tmp_path, sr=22050, mono=True)
    st.success(f"Loaded audio. Duration: {len(y)/sr:.2f} seconds, Sample rate: {sr} Hz")
except Exception as e:
    show_exception(e)
    st.stop()

# Plot waveform
try:
    st.subheader("Waveform")
    fig_wav, ax_wav = plt.subplots(figsize=(10, 3))
    ax_wav.plot(y)
    ax_wav.set_xlabel("Samples")
    ax_wav.set_ylabel("Amplitude")
    st.pyplot(fig_wav)
except Exception as e:
    show_exception(e)

# Compute STFT / Spectrogram
try:
    st.subheader("Spectrogram")
    D = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    frequencies = librosa.fft_frequencies(sr=sr)
    times = librosa.frames_to_time(np.arange(D.shape[1]), sr=sr, hop_length=512, n_fft=2048)

    fig_spec, ax_spec = plt.subplots(figsize=(10, 4))
    S_db = librosa.amplitude_to_db(D, ref=np.max)
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=ax_spec)
    fig_spec.colorbar(img, ax=ax_spec, format="%+2.0f dB")
    st.pyplot(fig_spec)
except Exception as e:
    show_exception(e)

# Map frequency -> note
def freq_to_note(freq):
    if freq is None or freq <= 0:
        return None
    try:
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F',
                      'F#', 'G', 'G#', 'A', 'A#', 'B']
        midi_note = int(round(69 + 12 * np.log2(freq / 440.0)))
        octave = midi_note // 12 - 1
        note_index = midi_note % 12
        return f"{note_names[note_index]}{octave}"
    except Exception:
        return None

# Extract all notes (with safe skipping)
notes = []
try:
    for i in range(D.shape[1]):
        col = D[:, i]
        if np.all(col == 0):
            notes.append(None)
            continue
        index = int(np.argmax(col))
        freq = float(frequencies[index]) if index < len(frequencies) else None
        notes.append(freq_to_note(freq))
except Exception as e:
    show_exception(e)
    st.stop()

# Show first 100 detected notes (timeline style with wrapping)
st.subheader("Detected Notes (first 100 frames)")

display_notes = [n for n in notes[:100] if n is not None]

if display_notes:
    lines = []
    for i in range(0, len(display_notes), 20):  # wrap every 20 notes
        chunk = display_notes[i:i+20]
        line = " → ".join([f"🎵 {n}" for n in chunk])
        lines.append(line)
    timeline_preview = "\n".join(lines)
    st.markdown(f"```\n{timeline_preview}\n```")
else:
    st.write("No notes detected in first 100 frames.")



# Save notes to a text file for download (timeline style with wrapping)
try:
    notes_txt_path = "all_notes.txt"
    display_notes = [n for n in notes if n is not None]

    if display_notes:
        # Wrap after 20 notes per line
        lines = []
        for i in range(0, len(display_notes), 20):
            chunk = display_notes[i:i+20]
            line = " → ".join([f"🎵 {n}" for n in chunk])
            lines.append(line)
        timeline_text = "\n".join(lines)
    else:
        timeline_text = "No notes detected."

    with open(notes_txt_path, "w", encoding="utf-8") as f:
        f.write(timeline_text)

    with open(notes_txt_path, "rb") as f:
        st.download_button("Download all_notes.txt", f, file_name="all_notes.txt", mime="text/plain")
except Exception as e:
    show_exception(e)

# Create MIDI file
try:
    note_to_midi = {
        'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
        'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
    }
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    duration = 240  # smaller duration for less overlap (you can change this)

    for note in notes:
        if note is None:
            continue
        name = note[:-1]
        octave = int(note[-1])
        midi_number = note_to_midi[name] + (octave + 1) * 12
        track.append(Message('note_on', note=midi_number, velocity=64, time=0))
        track.append(Message('note_off', note=midi_number, velocity=64, time=duration))

    midi_file_path = "extracted_song.mid"
    mid.save(midi_file_path)
    with open(midi_file_path, "rb") as f:
        st.download_button("Download MIDI file", f, file_name="extracted_song.mid", mime="audio/midi")
    st.success("MIDI file created.")
except Exception as e:
    show_exception(e)

# Clean up temporary file (optional)
try:
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
except Exception:
    pass
