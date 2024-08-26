import datetime
import subprocess
import os
from helpers import log
from remove_silence import remove_silence
from elevenlabs.core.api_error import ApiError
from elevenlabs import VoiceSettings

def pcm_to_mp3(mp3_filename, pcm, samplerate):
    os.makedirs(os.path.dirname(mp3_filename), exist_ok=True)

    wav_filename = mp3_filename[:-4] + ".wav"
    # first pass saves the raw audio
    p = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "panic",
            "-f",
            "s16le",
            "-ar",
            str(samplerate),
            "-ac",
            "1",
            "-i",
            "-",
            wav_filename,
        ],
        stdin=subprocess.PIPE,
    )
    p.communicate(pcm)
    log(f"saved {wav_filename}")
    
    # second pass normalizes the audio and removes silences
    remove_silence(wav_filename, mp3_filename)
    log(f"saved {mp3_filename}")


def clone_voice(client, pcm, samplerate):
    log(f"cloning voice from {len(pcm)} bytes...")
    dt = datetime.datetime.now().isoformat()
    filename = f"references/{dt.replace(':','-')}.mp3"
    pcm_to_mp3(filename, pcm, samplerate)
    log(f"uploading mp3 to elevenlabs...")
    voice = client.clone(
        name=dt,
        files=[
            filename,
        ],
    )
    log(f"cloned {filename} into name={voice.name} id={voice.voice_id}")
    voice_settings = VoiceSettings(
        stability=0.4, similarity_boost=1.0, style=0.4, use_speaker_boost=True
    )
    voice.edit(voice_settings=voice_settings)
    log(f"set voice settings for clone name={voice.name} id={voice.voice_id} settings={voice_settings}")
    return voice


def remove_old_voices(client, max_voice_count=15):
    voices = client.voices.get_all(show_legacy=True).voices
    cloned = [e for e in voices if e.category == "cloned"]
    vimh_voices = []
    for voice in cloned:
        try:
            date = voice.name
            if date[-1] == "Z":
                date = date[:-1]
            dt = datetime.datetime.fromisoformat(date)
            vimh_voices.append([dt, voice])
        except ValueError:
            pass
    vimh_voices.sort()
    deleted_voices = []
    for dt, voice in vimh_voices[:-max_voice_count]:
        try:
            voice.delete()
            log(f"deleted {voice.name}")
        except ApiError:
            log(f"already deleted {voice.name}")
            pass
        deleted_voices.append(voice.name)
    leftover_voices = [e for e in voices if e.name not in deleted_voices]
    print("voice:", leftover_voices)
    return leftover_voices
