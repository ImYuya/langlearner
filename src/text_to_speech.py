import pyttsx3

# 日本語
# voice_id = "com.apple.voice.compact.ja-JP.Kyoko"

# 英語
voice_id = "com.apple.voice.compact.en-US.Samantha"
# voice_id = "com.apple.eloquence.en-US.Reed"

def text_to_speech(text, voice_id="com.apple.voice.compact.en-US.Samantha"):
    tts = pyttsx3.init("nsss")
    tts.setProperty('rate', 170)
    tts.setProperty('voice', voice_id)
    tts.say(text)
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


if __name__ == "__main__":
    speech_to_text("Hello world!", voice_id)
