
from api_configs.configs import *
import asyncio
import requests

# Import Deepgram components
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
)

asr_config = get_asr_config()

asr_api = asr_config["default_api"]

asr_model = asr_config["apis"][asr_api]["model"]
asr_sample_rate =  asr_config["apis"][asr_api]["sample_rate"]
asr_language =  asr_config["apis"][asr_api]["language"]


# Define TranscriptCollector class
class TranscriptCollector:
    def __init__(self):
        self.reset()

    def reset(self):
        # Reset transcript parts
        self.transcript_parts = []

    def add_part(self, part):
        # Add a part to the transcript
        self.transcript_parts.append(part)

    def get_full_transcript(self):
        # Get the full transcript
        return ' '.join(self.transcript_parts)

# Create a global transcript collector instance
transcript_collector = TranscriptCollector()


if asr_api ==  "deepgram":

  async def get_transcript(callback):
    transcription_complete = asyncio.Event()  # Event to signal transcription completion
    
    try:
        
        
        # Set up Deepgram client
        config = DeepgramClientOptions(options={"keepalive": "true"})
        deepgram: DeepgramClient = DeepgramClient("", config)

        dg_connection = deepgram.listen.asynclive.v("1")
        print("Listening...")

        async def on_message(self, result, **kwargs):
            sentence = result.channel.alternatives[0].transcript
            
            if not result.speech_final:
                transcript_collector.add_part(sentence)
            else:
                # This is the final part of the current sentence
                transcript_collector.add_part(sentence)
                full_sentence = transcript_collector.get_full_transcript()
                if len(full_sentence.strip()) > 0:
                    full_sentence = full_sentence.strip()
                    print(f"Human: {full_sentence}")
                    callback(full_sentence)  # Call the callback with the full_sentence
                    transcript_collector.reset()
                    transcription_complete.set()  # Signal to stop transcription and exit

        # Set up Deepgram connection event handler
        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

        # Set up Deepgram live options
        options = LiveOptions(
            model=asr_model,
            punctuate=True,
            language=asr_language,  #de
            encoding="linear16",
            channels=1,
            sample_rate=asr_sample_rate,
            endpointing=300,
            smart_format=True
            
        )

        # Start Deepgram connection
        await dg_connection.start(options)

        # Open a microphone stream on the default input device
        microphone = Microphone(dg_connection.send)
        microphone.start()

        # Wait for the transcription to complete
        await transcription_complete.wait()

        # Wait for the microphone to close
        microphone.finish()

        # Indicate that we've finished
        await dg_connection.finish()
       
    except Exception as e:
        print(f"Could not open socket: {e}")
        return
