# app.py
import gradio as gr
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from mido import Message, MidiFile, MidiTrack
import tempfile
import os
import soundfile as sf

# --- Helper functions (same logic as your Streamlit app) ---
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

def to_timeline_text(notes_list, wrap=20):
    display_notes = [n for n in notes_list if n is not None]
    if not display_notes:
        return "No notes detected."
    lines = []
    for i in range(0, len(display_notes), wrap):
        chunk = display_notes[i:i+wrap]
        line = " → ".join([f"🎵 {n}" for n in chunk])
        lines.append(line)
    return "\n".join(lines)

def analyze_file(audio_file):
    """
    audio_file: path to the uploaded file (gradio saves uploaded file to a temp path)
    Returns: dict containing images and downloadable bytes
    """
    # Ensure filename ext consistent
    tmp_path = audio_file.name if hasattr(audio_file, "name") else audio_file
    # Load with soundfile/librosa
    try:
        # librosa.load accepts path or numpy array. We read via soundfile to preserve format.
        y, sr = librosa.load(tmp_path, sr=22050, mono=True)
    except Exception as e:
        return {"error": f"Could not load audio: {e}"}

    duration = len(y)/sr

    # Waveform plot
    fig_wav, ax_wav = plt.subplots(figsize=(10, 2.5))
    ax_wav.plot(y, linewidth=0.5)
    ax_wav.set_xlabel("Samples")
    ax_wav.set_ylabel("Amplitude")
    ax_wav.set_title(f"Waveform (duration: {duration:.2f}s, sr: {sr} Hz)")
    fig_wav.tight_layout()

    # STFT / Spectrogram
    D = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    frequencies = librosa.fft_frequencies(sr=sr)
    times = librosa.frames_to_time(np.arange(D.shape[1]), sr=sr, hop_length=512, n_fft=2048)

    fig_spec, ax_spec = plt.subplots(figsize=(10, 3))
    S_db = librosa.amplitude_to_db(D, ref=np.max)
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=ax_spec)
    fig_spec.colorbar(img, ax=ax_spec, format="%+2.0f dB")
    ax_spec.set_title("Spectrogram")

    # Extract all notes (per frame)
    notes = []
    for i in range(D.shape[1]):
        col = D[:, i]
        if np.all(col == 0):
            notes.append(None)
            continue
        index = int(np.argmax(col))
        freq = float(frequencies[index]) if index < len(frequencies) else None
        notes.append(freq_to_note(freq))

    # Prepare timeline preview (first 100 frames wrapped)
    preview_text = to_timeline_text(notes[:100], wrap=20)

    # Prepare full notes text file (wrapped lines)
    notes_txt = to_timeline_text(notes, wrap=20)
    notes_bytes = notes_txt.encode("utf-8")

    # Create MIDI
    note_to_midi = {
        'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
        'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
    }
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    duration_ticks = 240

    for note in notes:
        if note is None:
            continue
        # handle names like C#10 safely
        name = note[:-1]
        octave = int(note[-1])
        midi_number = note_to_midi.get(name, 0) + (octave + 1) * 12
        track.append(Message('note_on', note=midi_number, velocity=64, time=0))
        track.append(Message('note_off', note=midi_number, velocity=64, time=duration_ticks))

    # Save midi to bytes
    midi_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mid")
    mid.save(midi_tmp.name)
    midi_tmp.flush()
    with open(midi_tmp.name, "rb") as f:
        midi_bytes = f.read()
    os.unlink(midi_tmp.name)

    # Save images to temporary files for Gradio to return
    wav_png = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    spec_png = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig_wav.savefig(wav_png.name, bbox_inches='tight')
    fig_spec.savefig(spec_png.name, bbox_inches='tight')
    plt.close(fig_wav)
    plt.close(fig_spec)

    return {
        "waveform_image": wav_png.name,
        "spectrogram_image": spec_png.name,
        "preview_text": preview_text,
        "notes_bytes": notes_bytes,
        "midi_bytes": midi_bytes,
        "duration": f"{duration:.2f} s",
        "sr": sr
    }

# --- Gradio UI ---
with gr.Blocks(title="Music Note Extractor & MIDI Generator") as demo:
    gr.Markdown("## 🎵 Music Note Extractor & MIDI Generator")
    gr.Markdown("Upload a WAV or MP3 file and get extracted notes & a MIDI file.")
    with gr.Row():
        audio_input = gr.File(label="Upload WAV/MP3", file_types=["audio"])
        analyze_btn = gr.Button("Analyze")
    with gr.Row():
        with gr.Column(scale=2):
            waveform_out = gr.Image(label="Waveform")
            spectrogram_out = gr.Image(label="Spectrogram")
        with gr.Column(scale=1):
            info_txt = gr.Markdown("")
            preview_box = gr.Textbox(label="Detected Notes (preview, wrapped)", lines=8)
            download_notes = gr.File(label="Download Notes (timeline style)")
            download_midi = gr.File(label="Download MIDI")

    def on_analyze(file):
        if file is None:
            return gr.update(), gr.update(), "No file", "", None, None
        result = analyze_file(file)
        if "error" in result:
            return gr.update(), gr.update(), f"Error: {result['error']}", "", None, None
        info = f"Duration: {result['duration']}, Sample rate: {result['sr']} Hz"
        # prepare downloadable files as tuple (name, bytes)
        notes_file = ("all_notes.txt", result["notes_bytes"])
        midi_file = ("extracted_song.mid", result["midi_bytes"])
        return result["waveform_image"], result["spectrogram_image"], info, result["preview_text"], notes_file, midi_file

    analyze_btn.click(fn=on_analyze, inputs=[audio_input], outputs=[waveform_out, spectrogram_out, info_txt, preview_box, download_notes, download_midi])

if __name__ == "__main__":
    demo.launch()
