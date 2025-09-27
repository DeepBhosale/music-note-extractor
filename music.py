import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from mido import Message, MidiFile, MidiTrack

# ---------------- Step 1: Load audio and plot waveform ----------------
filename = 'O Rangrez Bhaag Milkha Bhaag 128 Kbps.wav'  # your file
y, sr = librosa.load(filename)

plt.figure(figsize=(12, 4))
plt.plot(y)
plt.title("Waveform of the Song")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.show()

# ---------------- Step 2: Short-Time Fourier Transform (STFT) ----------------
D = np.abs(librosa.stft(y))  # STFT magnitude
frequencies = librosa.fft_frequencies(sr=sr)
times = librosa.frames_to_time(np.arange(D.shape[1]), sr=sr)

plt.figure(figsize=(12, 6))
librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
                         sr=sr, x_axis='time', y_axis='hz')
plt.colorbar(format='%+2.0f dB')
plt.title('Frequency Spectrum (Spectrogram)')
plt.show()

# ---------------- Step 3: Map frequencies to musical notes ----------------
def freq_to_note(freq):
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    if freq == 0:
        return None
    midi_note = int(round(69 + 12 * np.log2(freq / 440.0)))
    octave = midi_note // 12 - 1
    note_index = midi_note % 12
    return f"{note_names[note_index]}{octave}"

# Extract main frequency at each time frame (all notes)
notes = []
for i in range(D.shape[1]):
    index = np.argmax(D[:, i])
    freq = frequencies[index]
    note = freq_to_note(freq)
    notes.append(note)

# Print all detected notes (first 50 frames as sample)
print("All Detected Notes (first 50 frames):")
print(notes[:50])

# ---------------- Step 4: Save all notes to a text file ----------------
with open("all_notes.txt", "w") as f:
    for note in notes:
        if note is not None:  # skip empty frames
            f.write(note + "\n")
print("\nAll notes saved to all_notes.txt")

# ---------------- Step 5: Create MIDI file ----------------
note_to_midi = {
    'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
    'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
}

mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)

duration = 480  # duration of each note in MIDI ticks

for note in notes:
    if note is None:
        continue  # skip frames with no note
    name = note[:-1]
    octave = int(note[-1])
    midi_number = note_to_midi[name] + (octave + 1) * 12
    track.append(Message('note_on', note=midi_number, velocity=64, time=0))
    track.append(Message('note_off', note=midi_number, velocity=64, time=duration))

mid.save('all_notes_song.mid')
print("MIDI file saved as all_notes_song.mid")
