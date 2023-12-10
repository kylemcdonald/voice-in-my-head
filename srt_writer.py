import os
import time
import math

def seconds_to_srt_time(seconds: float) -> str:
    dec, whole = math.modf(seconds)
    dec = round(dec * 1000)
    m, s = divmod(int(whole), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{dec:03d}"

class SrtWriter:
    def __init__(self, id):
        self.index = 0
        self.start = None
        self.fn = f"transcripts/{id}.srt"

    def write(self, begin, end, text):
        if self.start is None:
            self.start = begin
        begin_stamp = seconds_to_srt_time(begin - self.start)
        end_stamp = seconds_to_srt_time(end - self.start)
        os.makedirs(os.path.dirname(self.fn), exist_ok=True)
        with open(self.fn, "a") as f:
            f.write(f"{self.index}\n")
            f.write(f"{begin_stamp} --> {end_stamp}\n")
            f.write(f"{text}\n")
            f.write("\n")
            self.index += 1