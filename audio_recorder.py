import threading
from helpers import log
from time import sleep
from vad import VAD

class AudioRecorder:
    def __init__(self):
        self.vad = VAD()
        self.shutdown_flag = threading.Event()
        self.recording_flag = threading.Event()
        self.vad_flag = threading.Event()
        self.thread = None
        self.recording = b""
        
    def start(self, speaker_device, frame_count):
        self.speaker_device = speaker_device
        self.frame_count = frame_count
        self.thread = threading.Thread(target=self.loop)
        self.thread.start()

    def loop(self):
        while True:
            audio_content = self.speaker_device.read_frames(self.frame_count)
            if len(audio_content) == 0:
                sleep(0.01)
                continue

            if self.shutdown_flag.is_set():
                break
            if self.recording_flag.is_set():
                self.recording += audio_content
            if self.vad_flag.is_set():
                self.vad.update(audio_content)

    def start_recording(self):
        self.recording_flag.set()
        self.recording = b""

    def stop_recording(self):
        self.recording_flag.clear()
        
    def start_vad(self):
        self.vad_flag.set()
        self.vad.reset()
        
    def stop_vad(self):
        self.vad_flag.clear()

    def stop_and_join(self):
        log("begin audio recorder shutdown")
        self.shutdown_flag.set()
        if self.thread:
            self.thread.join()
        log("end audio recorder shutdown")
