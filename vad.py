import onnxruntime as ort
import numpy as np
import librosa
import torch
import queue
from time import time
from helpers import log

ort.set_default_logger_severity(3)
torch.set_num_threads(1)


# streaming voice activity detector with arbitrary samplerate
class VAD:
    def __init__(self, orig_sr=44100):
        self.orig_sr = orig_sr
        self.model, self.utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            onnx=True,
            verbose=False,
        )
        (
            self.get_speech_timestamps,
            self.save_audio,
            self.read_audio,
            self.VADIterator,
            self.collect_chunks,
        ) = self.utils
        self.vad_iterator = self.VADIterator(self.model)
        self.output_queue = queue.Queue()
        self.reset()

    def reset(self):
        self.vad_iterator.reset_states()
        self.output_queue.queue.clear()
        self.speaking = False
        self.last_speaking_time = None
        self.running = np.array([], dtype=np.float32)

    def update(self, y_buffer):
        y = np.frombuffer(y_buffer, dtype=np.int16).astype(np.float32) / 32768
        y_resampled = librosa.resample(y, orig_sr=self.orig_sr, target_sr=16000)
        self.running = np.hstack((self.running, y_resampled))
        window_size = 1536
        while len(self.running) > window_size:
            chunk = self.running[:window_size]
            speech_dict = self.vad_iterator(chunk, return_seconds=True)
            if speech_dict:
                self.output_queue.put(speech_dict)
                if 'start' in speech_dict:
                    log(f'started speaking')
                    self.speaking = True
                if 'end' in speech_dict:
                    log(f'stopped speaking')
                    self.speaking = False
            self.running = self.running[window_size:]
        if self.speaking:
            self.last_speaking_time = time()
