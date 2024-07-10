from api_configs.configs import *

import requests

tts_config = get_tts_config()

tts_api = tts_config["default_api"]

if tts_api ==  "deepgram":

  def stream_audio_from_text(text, api_key, model_name):
    """
    Streams audio from the given text using the specified API.

    Args:
        text (str): The text to convert to speech.
        api_key (str): API key for authentication.
        model_name (str): Model name to be used for TTS generation.

    Returns:
        Generator: Yields chunks of audio data.
    """
    DEEPGRAM_URL = f"https://api.deepgram.com/v1/speak?model={model_name}&performance=some&encoding=linear16&sample_rate=24000"
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json"
    }
    payload = {"text": text}
    
    with requests.post(DEEPGRAM_URL, stream=True, headers=headers, json=payload) as r:
        for chunk in r.iter_content(chunk_size=1024):
            yield chunk

