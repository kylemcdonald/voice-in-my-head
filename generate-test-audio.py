import csv
import os
from elevenlabs.client import ElevenLabs
from elevenlabs.core import ApiError
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ["ELEVENLABS_API_KEY"]
client = ElevenLabs(api_key=api_key)

def generate_audio_files():
    # Get the voice ID for "Liam"
    voices = client.voices.get_all()
    voice_id = next((voice.voice_id for voice in voices.voices if voice.name == "Liam"), None)
    
    if voice_id is None:
        raise ValueError("Voice 'Liam' not found")
    
    with open('scripts/script.csv', 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for index, row in enumerate(reader):
            response_nl = row.get('response-nl', '').strip()
            if response_nl:
                try:
                    audio = client.text_to_speech.convert(
                        text=response_nl,
                        voice_id=voice_id,
                        model_id="eleven_turbo_v2_5",
                        language_code="nl"
                    )
                    os.makedirs("test_audio", exist_ok=True)
                    with open(f"test_audio/response_{index}.mp3", "wb") as f:
                        for chunk in audio:
                            f.write(chunk)
                    print(f"Generated audio for response {index}")
                except ApiError as e:
                    print(f"Error generating audio for response {index}: {str(e)}")

if __name__ == "__main__":
    generate_audio_files()
