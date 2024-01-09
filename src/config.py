import os
from dotenv import load_dotenv
load_dotenv(override=True)

GOOGLE_MODEL = 'gemini-pro'
GOOGLE_MODEL_VISION = 'gemini-pro-vision'
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

OPENAI_MODEL = "gpt-3.5-turbo"
OPENAI_MODEL_VISION = "gpt-4-vision-preview"
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

PICOVOICE_ACCESS_KEY = os.getenv('PICOVOICE_ACCESS_KEY')
PICOVOICE_KEYWORD_PATH = os.getenv('PICOVOICE_KEYWORD_PATH')

MESSAGES = {
    "pressSpace": "Press and hold space to speak",
    "loadingModel": "Loading...",
    "noAudioInput": "Error: No sound input!"
}

WHISPER_RECOGNITION = {
    "model": "distil-whisper/distil-large-v2"  # distil-whisper/distil-large-v2 or distil-whisper/distil-small.en or distil-whisper/distil-medium.en
}

LLM = {
    "model": "bakllava",  # "openai" or "gemini" or <ollama model name> (ex: "openchat" or "bakllava")
    "systemPrompt": """
        You are a weather caster.
        No matter what question you ask, you always answer it by relating it to the weather.
        Answer in up to 15 words at most.
    """,
    # stream and url parameters are not applicable for gemini model
    "stream": False,  # for ollama and openai model
    "url": "http://localhost:11434/api/chat",  # for ollama model
    "timeout": 20,  # for ollama model
}

FRAMES_DIR = "./frames"
FRAME_TEMP_DIR = "./temp"
FRAME_TEMP_FILE_NAME = "temp.jpg"

CONVERSATION = {
    "greeting": "Hi, how can I help you?"
}