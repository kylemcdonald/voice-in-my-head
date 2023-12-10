import os
from concurrent.futures import ThreadPoolExecutor
import io
from datetime import datetime
from urllib.parse import urlparse
import json
import inspect


def iso_time():
    return datetime.now().isoformat(timespec="milliseconds")


class Logger:
    def __init__(self):
        self.id = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.fn = f"logs/{self.id}.jsonl"
        os.makedirs(os.path.dirname(self.fn), exist_ok=True)

    def __call__(self, entry):
        with open(self.fn, "a") as f:
            f.write(json.dumps(entry) + "\n")


singleton_logger = Logger()


def log(*data):
    if len(data) == 1:
        data = data[0]

    time = iso_time()

    current_frame = inspect.currentframe()
    caller_frame = current_frame.f_back
    caller = caller_frame.f_code
    fn = os.path.basename(caller.co_filename)
    line_no = caller.co_firstlineno
    func = caller.co_name
    caller_str = f"{fn}:{line_no} {func}"

    try:
        print(caller_str, json.dumps(data, indent=2), flush=True)
    except:
        data = str(data)
        print(caller_str, data, flush=True)

    entry = {"time": time, "fn": fn, "func": func, "no": line_no, "data": data}

    singleton_logger(entry)


def convert_to_unix_time(iso_time):
    dt = datetime.strptime(iso_time, "%Y-%m-%dT%H:%M:%S.%fZ")
    # dt = dt.replace(tzinfo=pytz.UTC)
    unix_time = dt.timestamp()
    return unix_time


def extract_room_name(url):
    parsed_url = urlparse(url)
    path_parts = parsed_url.path.split("/")
    return path_parts[-1]


class GeneratorBytesIO(io.IOBase):
    def __init__(self, generator_func):
        self.generator = generator_func

    def read(self, size=-1):
        try:
            return next(self.generator)
        except StopIteration:
            return b""

    def readable(self):
        return True


class ThreadedJob:
    def __init__(self, func, *args, **kwargs):
        self.executor = ThreadPoolExecutor()
        self.future = self.executor.submit(func, *args, **kwargs)

    def wait(self):
        result = self.future.result()
        self.executor.shutdown()
        return result


def chunker(iterable, chunk_size):
    chunk = bytearray()
    for item in iterable:
        chunk.extend(item)
        while len(chunk) >= chunk_size:
            yield bytes(chunk[:chunk_size])
            chunk = chunk[chunk_size:]
    if chunk:
        yield bytes(chunk)


def write_file_from_generator(filename, generator):
    accumulated = b""
    for chunk in generator:
        if chunk:
            accumulated += chunk
            yield chunk
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        f.write(accumulated)


def read_file_to_generator(filename, byte_count):
    with open(filename, "rb") as f:
        while True:
            chunk = f.read(byte_count)
            if chunk:
                yield chunk
            else:
                break
