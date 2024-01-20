import os
import io

import pyttsx3
import replicate
from openai import OpenAI
from pydantic import BaseModel
from pydub import AudioSegment
from pydub.playback import play
from dotenv import load_dotenv
load_dotenv(override=True)

class Speech(BaseModel):
    text: str
    emotion_text: str = "Happy"
    voice_id: str = "com.apple.voice.compact.en-US.Samantha"
    speaker_id: str = "8051"
    language: str = "English"


def text_to_speech(speech: Speech):
    tts = pyttsx3.init("nsss")
    tts.setProperty('rate', 170)
    tts.setProperty('voice', speech.voice_id)
    tts.say(speech.text)
    tts.runAndWait()
    tts.stop()

    # いい音声を探す時に使う
    # tts = pyttsx3.init("nsss")
    # voices = tts.getProperty('voices')
    # for voice in voices:
    #     if voice.languages == ["en_US"]:
    #         print(voice, voice.id)
    #         tts.setProperty('voice', voice.id)
    #         tts.say(text)
    #         tts.runAndWait()
    #         tts.stop()


def text_to_speech_emotivoice_replicate(speech: Speech):
    output = replicate.run(
        "bramhooimeijer/emotivoice:261b541053a0a30d922fd61bb47fbbc669941cb84f96a8f0042f14e8ad34f494",
        # api_token=os.getenv('REPLICATE_API_TOKEN'),
        input={
            "prompt": speech.emotion_text,
            "content": speech.text,
            "speaker": speech.speaker_id,
            "language": speech.language
        }
    )
    print(output)

def text_to_speech_openai(speech: Speech):
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=speech.text,
    )

    # バイナリ形式のレスポンス内容をバイトストリームに変換します
    byte_stream = io.BytesIO(response.content)

    # バイトストリームからオーディオデータを読み込みます
    audio = AudioSegment.from_file(byte_stream, format="mp3")

    # オーディオを再生します
    play(audio)

    # オーディオデータをファイルに書き込みます
    audio.export("output.mp3", format="mp3")

    # 成功メッセージを返します
    return {"message": "音声の再生と保存が成功しました"}


if __name__ == "__main__":
    text_to_speech(Speech(text="Hello world! I'm happy to see you!", voice_id="com.apple.voice.compact.en-US.Samantha"))
    # text_to_speech_emotivoice_replicate(Speech(text="Hello world! I'm happy to see you!"))
    # print(text_to_speech_openai(Speech(text="Hello world! I'm happy to see you!"))["message"])
