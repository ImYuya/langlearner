import threading
import queue
import time
from datetime import datetime, timezone, timedelta
import os

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import sounddevice as sd
import numpy as np
import pyaudio
import wave

import config
from llm_transcription import ask_llm
from speech_to_text import speech_to_text

# グローバル変数で実行状態を管理
running = True
dev_params = {'input_user_text': False, 'output_record_wav': False, 'text_to_speech': True}


INPUT_FORMAT = pyaudio.paInt16
INPUT_CHANNELS = 1
INPUT_RATE = 16000
INPUT_CHUNK = 1024
INPUT_BLOCK = 1.5 * INPUT_RATE  # 無音判定用のブロックサイズ（秒数 * サンプリングレート）
SILENCE_THRESHOLD = 0.15  # 無音判定の閾値


def save_waveform_to_file(waveform: np.ndarray, file_path: str):
    """波形データをWAVファイルとして保存する。"""
    # PyAudioの設定に合わせて、16ビット整数に変換
    waveform_int16 = np.int16(waveform * 32767)
    
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(INPUT_CHANNELS)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(INPUT_FORMAT))
        wf.setframerate(INPUT_RATE)
        wf.writeframes(waveform_int16.tobytes())

def waveform_from_mic(q_record, device_index):
    global running, dev_params
    """マイクから波形を取得する。"""
    audio = pyaudio.PyAudio()

    def callbackStream():
        stream = audio.open(
            format=INPUT_FORMAT,
            channels=INPUT_CHANNELS,
            rate=INPUT_RATE,
            input=True,
            frames_per_buffer=INPUT_CHUNK,
            input_device_index=device_index
        )
        return stream

    while running:
        try:
            block_index = 0
            output_index = 0
            frames_block = []
            frames = []
            SUM_INPUT_CHUNK = 0
            stream = callbackStream()
            while True:
                data = stream.read(INPUT_CHUNK)
                SUM_INPUT_CHUNK += INPUT_CHUNK
                frames_block.append(data)
                frames.append(data)
                if SUM_INPUT_CHUNK >= INPUT_BLOCK:
                    waveform_block = np.frombuffer(b''.join(frames_block), np.int16).astype(np.float32) / 32768.0
                    # 無音判定
                    if np.max(np.abs(waveform_block)) < SILENCE_THRESHOLD:
                        print('無音')
                        waveform = np.frombuffer(b''.join(frames), np.int16).astype(np.float32) / 32768.0
                        print('ストリームを停止します。')
                        stream.stop_stream()
                        stream.close()
                        if block_index != 0:
                            q_record.put(waveform)  # 録音データをキューに追加
                            if dev_params["output_record_wav"]:
                                print('音声データを保存します。')
                                file_path = create_chat_file(folder_path="./uploads/sample", file_name=f"output_{output_index}.wav")
                                save_waveform_to_file(waveform, file_path)
                            output_index += 1
                        block_index = 0
                        frames = []
                        frames_block = []
                        SUM_INPUT_CHUNK = 0
                        print('新規ストリームを開始します。')
                        stream = callbackStream()
                    else:
                        print('音声あり')
                        if dev_params["output_record_wav"]:
                            file_path = create_chat_file(folder_path="./uploads/sample", file_name=f'output_block_{output_index}-{block_index}.wav')
                            save_waveform_to_file(waveform_block, file_path)
                        block_index += 1
                        frames_block = []
                        SUM_INPUT_CHUNK = 0
        except KeyboardInterrupt:
            running = False
            print('キーボード割り込みを検知しました。')
            print('ストリームを停止します。')
            stream.stop_stream()
            stream.close()
            audio.terminate()

def transcribe_audio(q_record, pipe, q_transcribe, max_buffer_time, max_buffer_size, filename):
    global running, dev_params
    buffer_content = ""
    last_output_time = time.time()

    while running:
        try:
            audio_data = q_record.get(timeout=1)  # 1秒間待ってからキューから取得
        except queue.Empty:
            continue

        result = pipe(audio_data)
        text = result["text"]
        buffer_content += text + " "
        current_time = time.time()

        # バッファの内容を出力するタイミングを判断
        if (current_time - last_output_time > max_buffer_time) or (len(buffer_content) >= max_buffer_size):
            stripped_content = buffer_content.strip()
            if stripped_content not in ["you", "I"]:  # youとIのみ認識した場合には除外
                q_transcribe.put(stripped_content)
                with open(filename, "a") as file:
                    file.write('User: ' + text + "\n")
            else:
                print(f"you or I: {buffer_content}")
            
            buffer_content = ""
            last_output_time = current_time
        del audio_data  # 処理後にメモリから削除

def input_user_text(q_transcribe, filename):
    global running
    while running:
        user_input = input("User Input for debug: ")
        q_transcribe.put(user_input)
        with open(filename, "a") as file:
            file.write('User: ' + user_input + "\n")

def assistant_transcription(q_transcribe, filename, q_assistant):
    global running
    first_transcription = True
    while running:
        try:
            text = q_transcribe.get(timeout=1)  # 1秒間待ってからキューから取得
        except queue.Empty:
            continue

        try:
            # ここでAssistantの返答を作成する
            if first_transcription:
                text = f"System: {config.LLM['systemPrompt']} \n User: {text}"
            # chatbot = ask_llm(text, image_path='./temp/temp.jpg')
            chatbot = ask_llm(text, image_path=None)
            # print(chatbot)
            # print("=========================================")
            assistant_text = chatbot[-1][1]['text']
            q_assistant.put(assistant_text)
            with open(filename, "a") as file:
                file.write('Assistant: ' + assistant_text + "\n")
            first_transcription = False
        except Exception as e:
            print(f"Error in assistant_transcription: {e}")

def create_pipe_for_speech_recognition():
    # モデルとプロセッサの設定
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    # model_id = "distil-whisper/distil-small.en"
    model_id = config.WHISPER_RECOGNITION

    # 指定されたmodel_idを使用して、事前学習済みの音声認識モデルをロードします
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
    # モデルを指定されたデバイス（CPUまたはGPU）に移動します。モデルの計算はこのデバイス上で行われます。
    model.to(device)
    # 指定されたmodel_idを使用して、事前学習済みのプロセッサをロードします。プロセッサは、音声データをモデルが理解できる形式に変換するために使用されます。
    processor = AutoProcessor.from_pretrained(model_id)

    # 音声認識のパイプラインの設定
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        torch_dtype=torch_dtype,
        device=device,
    )

    return pipe

def create_chat_file(folder_path="./uploads", file_name=f"{datetime.now(tz=timezone(timedelta(hours=+9))).strftime('%Y-%m-%d_%H%M')}.txt"):
    # 現在のディレクトリのパスを取得
    current_directory = os.getcwd()

    # folder_pathのパスを生成
    directory_path = os.path.join(current_directory, folder_path)

    # folder_pathが存在しない場合は再帰的に作成
    os.makedirs(directory_path, exist_ok=True)

    # テキストファイルの名前を生成（folder_path内）
    file_path = os.path.join(directory_path, file_name)

    return file_path


def main():
    global running, dev_params
    dev_params = {'input_user_text': True, 'output_record_wav': False, 'text_to_speech': False}  # デバッグ用

    # 録音、文字起こし、Assistant出力、文字入力のキューの設定
    q_record = queue.Queue(maxsize=10)  # キューのサイズ制限を設定
    q_transcribe = queue.Queue(maxsize=10)  # キューのサイズ制限を設定
    q_assistant = queue.Queue(maxsize=10)  # キューのサイズ制限を設定

    # ファイル名の設定
    filename = create_chat_file()

    if not dev_params["input_user_text"]:
        # 利用可能なオーディオデバイスのリスト表示
        print("利用可能なオーディオデバイス:")
        print(sd.query_devices())

        pipe = create_pipe_for_speech_recognition()

        # デバイスの指定
        device_index = None  # 適切なデバイスインデックスを設定するか、Noneのままにしてデフォルトを使用

        # バッファの設定
        max_buffer_time = 3  # バッファの内容を出力する時間間隔（秒）
        max_buffer_size = 300  # バッファの最大文字数

        # 並行処理数の最適化
        num_transcription_threads = 10  # 文字起こしスレッドの数を増やす

        # 録音スレッドの開始
        record_thread = threading.Thread(target=waveform_from_mic, args=(q_record, device_index))
        record_thread.start()

        # 文字起こしスレッドの開始
        transcription_threads = []
        # 指定された数のスレッドを作成
        for _ in range(num_transcription_threads):
            thread = threading.Thread(target=transcribe_audio, args=(q_record, pipe, q_transcribe, max_buffer_time, max_buffer_size, filename))
            # スレッドの開始
            thread.start()
            # スレッドをリストに追加
            transcription_threads.append(thread)
    
    else:  # 【デバッグ用】ユーザーpromptを文字入力
        # 文字入力用スレッド作成
        input_text_thread = threading.Thread(target=input_user_text, args=(q_transcribe, filename))
        input_text_thread.start()

    # Assistant出力スレッドの開始
    assistant_thread = threading.Thread(target=assistant_transcription, args=(q_transcribe, filename, q_assistant))
    assistant_thread.start()

    try:
        while True:
            time.sleep(0.1)
            try:
                assistant_text = q_assistant.get(timeout=1)  # 1秒間待ってからキューから取得
                if dev_params["text_to_speech"]:
                    speech_to_text(assistant_text)
            except queue.Empty:
                continue
    except KeyboardInterrupt:
        running = False
        if hasattr(record_thread, 'join') and isinstance(record_thread, threading.Thread):  # スレッドが存在し、スレッドである場合
            record_thread.join()
        for t in transcription_threads:
            if hasattr(t, 'join') and isinstance(t, threading.Thread):  # スレッドが存在し、スレッドである場合
                t.join()
        if hasattr(assistant_thread, 'join') and isinstance(assistant_thread, threading.Thread):  # スレッドが存在し、スレッドである場合
            assistant_thread.join()
        if hasattr(input_text_thread, 'join') and isinstance(input_text_thread, threading.Thread): # スレッドが存在し、スレッドである場合
            input_text_thread.join()
        print("\nRecording and transcription stopped.")

if __name__ == "__main__":
    main()
