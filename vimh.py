from helpers import log

log("importing")

import traceback
from daily import Daily, CallClient, EventHandler
from time import time, sleep
import hashlib
import argparse
import threading
import queue
from srt_writer import SrtWriter
from script_reader import ScriptReader
from elevenlabs_helpers import clone_voice, remove_old_voices
from chatgpt import ChatGPT
from daily_helpers import get_meeting_token
from helpers import (
    chunker,
    write_file_from_generator,
    read_file_to_generator,
    ThreadedJob,
    GeneratorBytesIO,
    extract_room_name,
    convert_to_unix_time,
)
from audio_recorder import AudioRecorder
import wave
import json
from streamp3 import MP3Decoder
import os
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()


elevenlabs_client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])


sample_rate = 44100
chunk_count = 10
frame_count = chunk_count * sample_rate // 100
max_total_time = int(os.getenv("TOTAL_TIME_MINUTES")) * 60
turn_time = int(os.getenv("TURN_TIME_SECONDS"))


class VoiceInMyHead(EventHandler, ScriptReader, ChatGPT):
    def __init__(self, script_fn, default_voice, language):
        ScriptReader.__init__(self, script_fn, language)
        ChatGPT.__init__(self)

        # conditions for starting the main loop
        self.inputs_updated = False
        self.joined = False
        self.transcription_started = False

        self.speech_queue = queue.Queue()

        self.thread = None
        self.shutdown_flag = threading.Event()
        self.messages = []

        log("begin audio recorder creation")
        self.audio_recorder = AudioRecorder()
        log("end audio recorder creation")

        log("begin virtual mic and speaker creation")
        self.mic_device = Daily.create_microphone_device(
            "my-mic", sample_rate=sample_rate, channels=1
        )
        self.speaker_device = Daily.create_speaker_device(
            "my-speaker", sample_rate=sample_rate, channels=1
        )
        Daily.select_speaker_device("my-speaker")
        log("end virtual mic and speaker creation")

        sleep(2)

        self.client = CallClient(self)
        self.client.set_user_name("voice")
        self.client.update_inputs(
            {
                "camera": False,
                "microphone": {"isEnabled": True, "settings": {"deviceId": "my-mic"}},
            },
            completion=self.on_inputs_updated,
        )

        self.client.update_subscription_profiles(
            {"base": {"camera": "unsubscribed", "microphone": "subscribed"}}
        )

        self.voice_list = remove_old_voices(elevenlabs_client)
        self.name_to_voice = {e.name: e.voice_id for e in self.voice_list}
        self.default_voice = default_voice

        log("done with init")

    def on_inputs_updated(self, inputs, error=None):
        if error:
            log(f"unable to update inputs: {error}")
            return
        log(f"inputs updated", inputs)
        self.inputs_updated = True

    def start(self):
        if self.inputs_updated and self.joined and self.transcription_started:
            self.main_loop_start.set()
        else:
            log("not ready to start main loop!")

    def on_transcription_started(self, status):
        log("on_transcription_started")
        log(status)

    def on_transcription_message(self, message):
        # messages from this app have no user_name
        if "user_name" not in message:
            return
        if "text" not in message:
            log("no text in message")
            return
        text = message["text"]
        log({"transcription": text})
        self.speech_queue.put(text)

        listen_end = convert_to_unix_time(message["timestamp"])
        listen_start = listen_end - message["duration_seconds"]
        self.srt_writer.write(listen_start, listen_end, "(User)\n" + text)
        self.client.send_app_message(
            {"room": "main-room", "event": "chat-msg", "message": text, "user": "user"}
        )

    # conditions for speech to be finished:
    # 1. they have said something
    # 2. they have been silent for a minimum amount of time

    # key note: we only want to quit if we have received a message from deepgram after detecting speech

    # max_duration: wait until a timeout is complete (nonzero), or until the first complete sentence (None)
    # mid_sentence_silence: wait at least this long since the last mid-sentence pause
    # end_sentence_silence: wait at least this long since the last finished sentence
    def listen(self, max_duration=None, mid_sentence_silence=6, end_sentence_silence=2):
        if max_duration is None:
            self.play_sound("listen-begin")
            self.audio_recorder.start_vad()
            # make sure we don't have any leftovers in the speech queue
            # this only applies when we are looking for the most recent response
            # and not when people are talking over the voice
            self.speech_queue.queue.clear()

        start_time = time()
        transcripts = []
        last_utterance_time = None
        while not self.shutdown_flag.is_set():
            # done listening if we've heard something
            # and there's been enough time since the last vad
            # and the last utterance is after the last vad (i.e. it's been transcribed)
            if max_duration is None:
                if len(transcripts) > 0:
                    last_vad_time = self.audio_recorder.vad.last_speaking_time
                    min_silence = mid_sentence_silence
                    # only use the quick turnaround if we are confident about having transcribed everything
                    if (
                        transcripts[-1][-1] in [".", "?", "!"]
                        and last_vad_time is not None
                        and last_utterance_time > last_vad_time
                    ):
                        min_silence = end_sentence_silence
                    # check to see if enough time has passed
                    if (
                        last_vad_time is not None
                        and time() - last_vad_time > min_silence
                    ):
                        log({"min_silence": min_silence})
                        break
            else:
                # done listening if we've heard enough (used during main experience)
                if time() - start_time > max_duration:
                    break

            try:
                transcript = self.speech_queue.get(timeout=0.1)
                transcripts.append(transcript)
                last_utterance_time = time()
            except queue.Empty:
                pass

        if max_duration is None:
            self.audio_recorder.stop_vad()
            self.play_sound("listen-end")

        transcript = " ".join(transcripts)

        self.messages.append({"role": "user", "content": transcript})
        log({"transcript": transcript})
        return transcript

    # can be a voice name or a voice id
    def set_voice(self, voice):
        self.default_voice = voice

    # could move this function into an elevenlabs helper class
    def speak(self, text, use_cache=True, accurate=False):
        text = text.replace("’", "'")  # causes some weird problems for elevenlabs?

        log({"speak": text})
        self.client.send_app_message(
            {"room": "main-room", "event": "chat-msg", "message": text, "user": "voice"}
        )

        voice = self.default_voice
        print("voice", voice)
        voice_id = voice
        if voice in self.name_to_voice:
            voice_id = self.name_to_voice[voice]
            print("voice in self.name_to_voice, voice_id: ", voice_id)

        hash = hashlib.sha256((voice + text).encode("utf-8")).hexdigest()
        output_filename = f"cache/{voice}/{hash}.mp3"

        byte_count = 2 * frame_count

        if use_cache and os.path.exists(output_filename):
            generator = read_file_to_generator(output_filename, byte_count)
        else:
            generator = elevenlabs_client.text_to_speech.convert_as_stream(
                text=text,
                optimize_streaming_latency=(0 if accurate else 3),
                voice_id=voice_id,
                model_id="eleven_turbo_v2_5"
            )
        if use_cache and not os.path.exists(output_filename):
            generator = write_file_from_generator(output_filename, generator)

        generator = GeneratorBytesIO(generator)
        decoder = MP3Decoder(generator)
        speak_start = time()
        for chunk in chunker(decoder, byte_count):
            self.mic_device.write_frames(chunk)
        speak_end = time()
        self.srt_writer.write(speak_start, speak_end, "(Voice)\n" + text)
        self.messages.append({"role": "assistant", "content": text})
        log({"speak-event": "done", "filename": output_filename, "start": speak_start, "end": speak_end})

        log("speak: done")

    def play_sound(self, fn):
        with wave.open(f"sound/{fn}.wav") as f:
            audio = f.readframes(-1)
        self.mic_device.write_frames(audio)

    def on_app_message(self, message, sender):
        log({"message": message})
        if "message" not in message:
            return
        text = message["message"]
        if text == "start":
            self.start()
        if text == "end":
            self.end()

    def run(self, meeting_url):
        self.main_loop_start = threading.Event()
        self.thread = threading.Thread(target=self.main_loop)
        self.thread.start()
        meeting_token = get_meeting_token(os.environ["DAILY_API_KEY"])
        self.client.join(
            meeting_url, meeting_token=meeting_token, completion=self.on_joined
        )
        sleep(2)

        self.srt_writer = SrtWriter(extract_room_name(meeting_url))        
        self.client.start_transcription(
            {
                "language": "en",
                "model": "2-general",
                "tier": "nova",
                "profanity_filter": False,
                "redact": False,
            },
            completion=self.on_start_transcription,
        )
        
        self.thread.join()
        log("main_loop joined")

    def on_joined(self, data, error):
        if error:
            log(f"unable to join meeting: {error}")
            return
        log("meeting joined")
        self.joined = True

    def on_start_transcription(self, data, error):
        if error:
            log(f"unable to start transcription: {error}")
            return
        log(f"transcription started")
        self.transcription_started = True

    def start_recording(self):
        self.audio_recorder.start_recording()

    def stop_recording(self):
        self.audio_recorder.stop_recording()

    def clone_voice(self):
        # remove old voices, clone current voice, save cloned voice id for later
        voice = clone_voice(elevenlabs_client, self.audio_recorder.recording, sample_rate)
        self.cloned_voice_id = voice.voice_id
        log({"voice_name": voice.name, "voice_id": voice.voice_id})

    def start_cloning(self):
        self.clone_job = ThreadedJob(self.clone_voice)

    def stop_cloning(self):
        self.clone_job.wait()
        return self.cloned_voice_id

    def start_collecting_messages(self):
        self.messages = []

    def get_collected_messages(self):
        return self.messages

    def experience_loop(self, goals_prompt):
        self.goals_prompt = goals_prompt
        entire_transcript = []
        while (time() - self.start_time) < max_total_time:
            try:
                log("starting new turn")
                overheard = self.listen(max_duration=turn_time)
                entire_transcript.append(overheard)

                # breakout if we need to shutdown
                if self.shutdown_flag.is_set():
                    log("escaping experience_loop")
                    break

                log("turn timeout")
                response = self.respond_to_overheard(overheard)
                self.speak(response, accurate=True)
            except Exception as e:
                log("error: " + str(e))

        entire_transcript = " ".join(entire_transcript)
        return entire_transcript

    def wait(self, seconds):
        sleep(seconds)

    def main_loop(self):
        self.main_loop_start.wait()
        
        self.start_time = time()

        log("begin audio recorder start")
        self.audio_recorder.start(self.speaker_device, frame_count)
        log("end audio recorder start")

        for row in self.rows:
            if self.shutdown_flag.is_set():
                log("escaping main_loop")
                break
            try:
                self.run_row(row)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                log("main_loop error: " + str(e))
                
        log("ending main_loop")

    def stop_and_join(self):
        log("begin main loop shutdown")
        self.shutdown_flag.set()
        if self.thread:
            self.thread.join()
        log("end main loop shutdown")

    def end(self):
        log("init: leave")
        # self.client.stop_transcription()
        self.stop_and_join()
        if self.audio_recorder:
            self.audio_recorder.stop_and_join()
        self.client.leave()  # tends to cause a segmentation fault?
        log("finished: leave")


def main():
    log("done importing")

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--meeting", required=True, help="Meeting URL")
    parser.add_argument("--id", required=False, help="Device ID")
    parser.add_argument(
        "-s", "--script", required=False, help="Script file", default="script.csv"
    )
    parser.add_argument("-l", "--language", required=False, help="Language code", default="en")
    parser.add_argument("-v", "--voice", required=False, help="Voice", default="Mimi")
    args, _ = parser.parse_known_args()

    log(args)

    Daily.init()

    app = VoiceInMyHead(f"scripts/" + args.script, args.voice, args.language)
    try:
        app.run(args.meeting)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        log(str(e))
        traceback.print_exc()
    finally:
        try:
            app.end()
        except Exception as e:
            log(str(e))


if __name__ == "__main__":
    main()
