import subprocess  # For running system commands
import os  # For environment variables and file operations
import signal  # For handling signals (not used in this script, but imported for potential use)
import asyncio  # For asynchronous programming
from dotenv import load_dotenv  # For loading environment variables
import shutil  # For file operations
import requests  # For making HTTP requests
import time  # For time-related functions
import threading
import clipboard
import json
import base64
import io
import threading
from PIL import Image
from PIL import ImageGrab
import shutil  # Import shutil for checking executables
from pynput import keyboard

from pynput import keyboard

from threading import Event


import re
import random
import requests
import time
from pyautogui import screenshot
import sounddevice as sd
import soundfile as sf


from api_configs.configs import *

from stream_tts import stream_audio_from_text
from stream_asr import get_transcript
from wake_words import get_wake_words, WakeWordEngine

# Import LangChain components
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_together import Together  #pip install langchain-together
from llm_definition import get_llm
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain



from florence2 import handle_captioning_florence2
from florence2 import handle_ocr_florence2
from florence2 import send_image_for_captioning_florence2
from florence2 import send_image_for_ocr_florence2

from hyprlab import send_image_for_captioning_and_ocr_hyprlab_gpt4o
from dl_yt_subtitles import download_youtube_video_info, extract_and_concat_subtitle_text, find_first_youtube_url


florence2_server_url = "http://213.173.96.19:5002/" 
HYPRLAB_API_KEY= "hypr-lab-xxxx" 


llm_config = get_llm_config()

tts_config = get_tts_config()

tts_api = tts_config["default_api"]

tts_model= tts_config["apis"][tts_api]["model"]

tts_api_key= tts_config["apis"][tts_api]["api_key"]



def get_caption_from_clipboard_gpt4o_hyprlab():
    # Check clipboard content

    try:
       content = ImageGrab.grabclipboard()
    except:
        content = clipboard.paste()
        print(type(content))
        if isinstance(content, str):
            if "https://www.youtu" in content and len(content)<100:
                video_metadata= download_youtube_video_info(find_first_youtube_url(content))
                subtitle_text= extract_and_concat_subtitle_text(str(video_metadata))
                print(subtitle_text)
                print(len(subtitle_text))
                return subtitle_text [:6000] 
                
            else:
              print("Returning text from the clipboard...")
              return content
    print(content)
    print(type(content))
    
    if isinstance(content, Image.Image):
        print("Processing an image from the clipboard...")
        if content.mode != 'RGB':
            content = content.convert('RGB')
            
        # Save image to a byte array
        img_byte_arr = io.BytesIO()
        content.save(img_byte_arr, format='JPEG', quality=60)
        img_byte_arr = img_byte_arr.getvalue()


        # Send image for captioning and return the result
        combined_caption = send_image_for_captioning_and_ocr_hyprlab_gpt4o(img_byte_arr, HYPRLAB_API_KEY)

        print(combined_caption)
   
  
        return combined_caption

    else:
        return "No image or text data found in the clipboard."

# Functions `handle_captioning` and `handle_ocr` need to be defined elsewhere in your code.
# They should update the `results` dictionary with keys 'caption' and 'ocr' respectively.

def get_caption_from_screenshot_gpt4o_hyprlab():


    # Take a screenshot and open it with PIL
    print("Taking a screenshot...")
    screenshot_image = screenshot()  # Uses PyAutoGUI to take a screenshot
    width, height = screenshot_image.size
    new_height = 500
    new_width = int((new_height / height) * width)
    
    # Resizing with the correct resampling filter
    resized_image = screenshot_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Save the resized image as JPEG
    img_byte_arr = io.BytesIO()
    resized_image.save(img_byte_arr, format='JPEG', quality=70)
    screenshot_image.save(img_byte_arr, format='JPEG', quality=70)
    img_byte_arr = img_byte_arr.getvalue()

    # Send image for captioning and return the result
    combined_caption = send_image_for_captioning_and_ocr_hyprlab_gpt4o(img_byte_arr, HYPRLAB_API_KEY)

    print(combined_caption)
   
  
    return combined_caption



def get_caption_from_clipboard_florence2():
    # Check clipboard content

    try:
       content = ImageGrab.grabclipboard()
    except:
        content = clipboard.paste()
        print(type(content))
        if isinstance(content, str):
            print("Returning text from the clipboard...")
            return content
    print(content)
    print(type(content))
    
    if isinstance(content, Image.Image):
        print("Processing an image from the clipboard...")
        if content.mode != 'RGB':
            content = content.convert('RGB')
            
        # Save image to a byte array
        img_byte_arr = io.BytesIO()
        content.save(img_byte_arr, format='JPEG', quality=60)
        img_byte_arr = img_byte_arr.getvalue()

        results = {}
        
        # Define tasks for threads
        thread1 = threading.Thread(target=handle_captioning_florence2, args=(img_byte_arr, results))
        thread2 = threading.Thread(target=handle_ocr_florence2, args=(img_byte_arr, results))

        # Start threads
        thread1.start()
        thread2.start()

        # Wait for threads to complete
        thread1.join()
        thread2.join()

        # Combine results and return
        combined_caption = results.get('caption', '') + "\nOCR RESULTS:\n" + results.get('ocr', '')
        return combined_caption

    else:
        return "No image or text data found in the clipboard."

# Functions `handle_captioning` and `handle_ocr` need to be defined elsewhere in your code.
# They should update the `results` dictionary with keys 'caption' and 'ocr' respectively.

def get_caption_from_screenshot_florence2():


    # Take a screenshot and open it with PIL
    print("Taking a screenshot...")
    screenshot_image = screenshot()  # Uses PyAutoGUI to take a screenshot
    #width, height = screenshot_image.size
    #new_height = 800
    #new_width = int((new_height / height) * width)
    
    # Resizing with the correct resampling filter
    #resized_image = screenshot_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Save the resized image as JPEG
    img_byte_arr = io.BytesIO()
    #resized_image.save(img_byte_arr, format='JPEG', quality=60)
    screenshot_image.save(img_byte_arr, format='JPEG', quality=60)
    img_byte_arr = img_byte_arr.getvalue()

    # Send image for captioning and return the result
    #caption = send_image_for_captioning(img_byte_arr)
    #ocr_result= send_image_for_ocr(img_byte_arr)
    #print(ocr_result)
    #caption += "\nOCR RESULTS:\n"+ocr_result
    
    results = {}
    
    thread1 = threading.Thread(target=handle_captioning_florence2, args=(img_byte_arr, results))
    thread2 = threading.Thread(target=handle_ocr_florence2, args=(img_byte_arr, results))

    # Start threads
    thread1.start()
    #time.sleep(2)
    thread2.start()

    # Wait for threads to complete
    thread1.join()
    thread2.join()
    print(results)
    # Combine results and print
    combined_caption = results['caption'] + "\nOCR RESULTS:\n"+ results['ocr']
        
    return combined_caption



def open_site(url):
    # Use subprocess.Popen to open the browser
    process = subprocess.Popen(['xdg-open', url])
    
    # Wait for 2 seconds
    time.sleep(1)
    
    # Kill the process
    process.terminate()  # Safely terminate the process
    # If terminate doesn't kill the process, you can use kill():
    # process.kill()
    
def extract_urls_to_open(input_string):
    # Define a regular expression pattern to find URLs within <open-url> tags
    pattern = r"<open-url>(https?://[^<]+)</open-url>"
    
    # Use re.findall to extract all occurrences of the pattern
    urls = re.findall(pattern, input_string)
    
    return urls


def extract_questions_to_send_to_askorkg(input_string):
    # Define a regular expression pattern to find content within <open-askorkg>...</open-orkg> tags
    pattern = r"<open-askorkg>(.*?)</open-askorkg>"
    
    # Use re.findall to extract all occurrences of the pattern
    contents = re.findall(pattern, input_string)
    
    # Return the content of the first tag pair, or None if there are no matches
    return contents[0] if contents else None


def extract_questions_to_send_to_wikipedia(input_string):
    # Define a regular expression pattern to find content within <open-askorkg>...</open-orkg> tags
    pattern = r"<open-wikipedia>(.*?)</open-wikipedia>"
    
    # Use re.findall to extract all occurrences of the pattern
    contents = re.findall(pattern, input_string)
    
    # Return the content of the first tag pair, or None if there are no matches
    return contents[0] if contents else None
    





# Load environment variables from .env file
load_dotenv()


# Define LanguageModelProcessor class
class LanguageModelProcessor:
    def __init__(self):
        # Initialize the language model (LLM)
        
        #self.llm =Together(model="mistralai/Mixtral-8x7B-Instruct-v0.1", max_tokens=400, together_api_key=os.getenv("TOGETHER_API_KEY"))#  ChatGroq(temperature=0, model_name="llama3-8b-8192", groq_api_key=os.getenv("GROQ_API_KEY"))
        # Alternatively, use OpenAI models (commented out)
        # self.llm = ChatOpenAI(temperature=0.5, model_name="gpt-4-0125-preview", openai_api_key=os.getenv("OPENAI_API_KEY"))
        # self.llm = ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo-0125", openai_api_key=os.getenv("OPENAI_API_KEY"))

        # Determine which language model to use based on the configuration
        
        self.llm= get_llm(llm_config)
        



        # Initialize conversation memory
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Load system prompt from file
        with open('system_prompt.txt', 'r') as file:
            system_prompt = file.read().strip()
        
        # Create chat prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{text}")
        ])

        # Create conversation chain
        self.conversation = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory
        )

    def process(self, text):
        # Add user message to memory
        self.memory.chat_memory.add_user_message(text)

        # Record start time
        start_time = time.time()

        # Get response from LLM
        response = self.conversation.invoke({"text": text})
        
        # Record end time
        end_time = time.time()

        # Add AI response to memory
        self.memory.chat_memory.add_ai_message(response['text'])

        # Calculate elapsed time
        elapsed_time = int((end_time - start_time) * 1000)
        print(f"LLM ({elapsed_time}ms): {response['text']}")
        return response['text']


class TextToSpeech:

    def __init__(self):
        self.player_process = None
        self.should_stop = False
        self.listener = None

    @staticmethod
    def is_installed(lib_name: str) -> bool:
        """Check if a command exists in the system's path"""
        return shutil.which(lib_name) is not None

    def stop(self):
        self.should_stop = True
        if self.player_process:
            self.player_process.terminate()
            self.player_process = None
        if self.listener:
            self.listener.stop()  # Stop the keyboard listener

    def on_activate(self):
        print("Hotkey activated - stopping TTS.")
        self.stop()

    def speak(self, text, stop_event: Event):
        if not self.is_installed("ffplay"):
            raise ValueError("ffplay not found, necessary to stream audio.")

        # Setup hotkey listener
        with keyboard.GlobalHotKeys({
                '<ctrl>+<shift>': self.on_activate}) as self.listener:
            player_command = ["ffplay", "-autoexit", "-", "-nodisp"]
            self.player_process = subprocess.Popen(
                player_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            try:
                audio_stream_generator = stream_audio_from_text(text, tts_api_key, tts_model)
                for chunk in audio_stream_generator:
                    if stop_event.is_set() or self.should_stop:
                        break
                    if chunk:
                        try:
                            self.player_process.stdin.write(chunk)
                            self.player_process.stdin.flush()
                        except BrokenPipeError:
                            print("TTS playback stopped.")
                            break
            finally:
                if self.player_process and self.player_process.stdin:
                    self.player_process.stdin.close()
                if self.player_process:
                    self.player_process.wait()
                self.player_process = None




class ConversationManager:
    def __init__(self):
        self.transcription_response = ""
        self.llm = LanguageModelProcessor()
        self.tts = TextToSpeech()
        self.stop_event = asyncio.Event()
        self.conversation_active = False

    async def start_conversation(self):
        self.conversation_active = True
        await self.main()

    async def speak_response(self, response):
        tts_task = asyncio.to_thread(self.tts.speak, response, self.stop_event)
        try:
            await tts_task
        except Exception as e:
            print(f"TTS error: {e}")

    async def main(self):
        def handle_full_sentence(full_sentence):
            self.transcription_response = full_sentence

        while self.conversation_active:
            self.stop_event.clear()
            self.tts = TextToSpeech()  # Create a new TTS instance for each response
            
            print("Listening for your command...")
            await get_transcript(handle_full_sentence)
            
            if "goodbye" in self.transcription_response.lower():
                self.conversation_active = False
                break
            
            # Process the transcription and generate a response
            llm_response = self.llm.process(self.transcription_response)
            
            # Handle URL opening
            extracted_url_to_open = extract_urls_to_open(llm_response)
            if extracted_url_to_open:
                open_site(extracted_url_to_open[0])
                llm_response = random.choice([
                    "Sure! Let me know if there's anything else you need.",
                    "All set! Anything else you'd like to explore?",
                    "The site has been opened! Feel free to ask more questions.",
                    "Done! Can I assist you with anything else today?",
                    "The link is now open! Let me know if you need further assistance."
                ])

            # Handle Ask ORKG
            question_for_askorkg = extract_questions_to_send_to_askorkg(llm_response)
            if question_for_askorkg:
                open_site(f"https://ask.orkg.org/search?query={question_for_askorkg}")
                llm_response = random.choice([
                    "Sure! I will use the Ask Open Knowledge Graph service to analyze the question: {0}",
                    "Got it! Let's see what Ask Open Knowledge Graph has on: {0}",
                    "I'm on it! Checking Ask Open Knowledge Graph for information about: {0}",
                    "Excellent question! I'll consult Ask Open Knowledge Graph about: {0}",
                    "One moment! I'll look that up on Ask Open Knowledge Graph for you about: {0}"
                ]).format(question_for_askorkg)

            # Handle Wikipedia
            question_for_wikipedia = extract_questions_to_send_to_wikipedia(llm_response)
            if question_for_wikipedia:
                open_site(f"https://en.wikipedia.org/w/index.php?search={question_for_wikipedia}")
                llm_response = random.choice([
                    "Sure! Here are the Wikipedia search results for: {0}",
                    "Let me pull up Wikipedia for you to explore: {0}",
                    "Checking Wikipedia for: {0}. Here's what I found!",
                    "I'll search Wikipedia for that. Hold on: {0}",
                    "One moment, I'm getting the information from Wikipedia on: {0}"
                ]).format(question_for_wikipedia)

            print(f"AI: {llm_response}")
            await self.speak_response(llm_response)
            
            self.transcription_response = ""

        print("Conversation ended. Listening for wake words again...")


async def main():
    conversation_manager = ConversationManager()
    wake_words = get_wake_words()
    wake_word_engine = WakeWordEngine(wake_words, conversation_manager.start_conversation)
    wake_word_engine.initialize()
    print("Listening for wake words...")
    await wake_word_engine.detect()

if __name__ == "__main__":
    asyncio.run(main()) 
    
'''
To dos:
- move wake word code to wake_word.py
- move skills to skills folder
'''
