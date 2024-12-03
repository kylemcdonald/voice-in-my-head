import os
import ffmpeg
import numpy as np
import librosa

DEVNULL = open(os.devnull, "w")

# threshold is the audio level threshold
# width is the width of the window to consider as a chunk (in seconds), this adds a kind of padding
# resolution is the resolution of the rms, lower value is more temporally accurate, but slower
def get_chunks(y, sr, threshold=0.2, width=0.5, resolution=0.1):
    hop_length = int(sr * resolution)
    rms = librosa.feature.rms(y=y, frame_length=hop_length, hop_length=hop_length)[0]
    rms /= rms.std() * 3
    chunks = []
    start = None

    # we can also do this by using a bigger frame_length, but this way is faster (but less accurate)
    rms = rms > threshold
    dilation = int(width / resolution)
    rms = np.convolve(rms, np.ones(dilation), mode="same") > 0

    for i in range(len(rms)):
        if rms[i]:
            if start == None:
                start = i
        else:
            if start is not None:
                a = start + 0.5
                b = i + 0.5
                chunks.append((int(hop_length * a), int(hop_length * b)))
                start = None
    return chunks


import subprocess as sp


def write_elevenlabs_mp3(fn, audio, sr):
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "panic",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-f",
        "s16le",
        "-i",
        "pipe:",
        "-ab",
        "192k",
        "-t",
        "00:07:00",
        fn,
    ]
    p = sp.Popen(command, stdin=sp.PIPE, stdout=None, stderr=None)

    audio = audio.astype(np.float32) * (32767 / np.abs(audio).max())
    audio = np.int16(audio)
    p.communicate(audio.tobytes())


def auread(filename, sr=44100, normalize=True):
    print("about to read", filename)
    y, file_sr = librosa.load(filename, sr=sr, mono=True)
    print("read file", filename, "with sr", file_sr)
    print("type", y.dtype)
    if normalize:
        y /= np.abs(y).max()
    return y, sr


# single function that takes input_fn and output_fn, and loads and chunks and saves to mp3
def remove_silence(input_fn, output_fn):
    print("reading", input_fn)
    y, sr = auread(input_fn)
    print("chunking")
    chunks = np.hstack([y[a:b] for a, b in get_chunks(y, sr)])
    print("writing", output_fn)
    write_elevenlabs_mp3(output_fn, chunks, sr)

if __name__ == "__main__":
    import sys
    input_fn = sys.argv[1]
    output_fn = sys.argv[2]
    remove_silence(input_fn, output_fn)