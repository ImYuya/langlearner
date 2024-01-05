import threading
import queue
import time
from datetime import datetime, timezone, timedelta
import os

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

from llm_transcription import ask_llm
from speech_to_text import speech_to_text

# グローバル変数で実行状態を管理
running = True
in_dev = False

# 利用可能なオーディオデバイスのリスト表示
print("利用可能なオーディオデバイス:")
print(sd.query_devices())

def is_silent(data, threshold=0.05):
    """無音を検出する。"""
    return np.max(np.abs(data)) < threshold

def contains_voice(data, threshold=0.05):
    """音声が含まれているかをチェックする。"""
    return np.max(np.abs(data)) >= threshold

def record_audio(q_record, max_record_duration, silence_duration, device_index):
    global running, in_dev
    with sd.InputStream(samplerate=16000, channels=1, callback=None, dtype='float32', device=device_index):
        while running:
            audio_data = sd.rec(int(max_record_duration * 16000), samplerate=16000, channels=1)
            sd.wait()  # 録音が完了するまで待機
            audio_data = np.squeeze(audio_data)
            now = datetime.now(tz=timezone(timedelta(hours=+9))).strftime("%Y-%m-%d_%H%M%S")

            # 録音データをファイルに出力　※デバッグ用
            if in_dev:
                file_path = create_chat_file(folder_path="./uploads/sample", file_name=f"raw_{now}.wav")
                write(file_path, 16000, audio_data)

            # 録音データに音声が含まれていない場合はスキップ
            if not contains_voice(audio_data):
                continue

            # 無音を検出して録音を早期終了
            for i in range(0, len(audio_data), int(16000 * silence_duration)):
                window = audio_data[i:i + int(16000 * silence_duration)]
                if is_silent(window):
                    audio_data = audio_data[:i]
                    break

            q_record.put(audio_data)  # 録音データをキューに追加
            del audio_data  # 処理後にメモリから削除

def transcribe_audio(q_record, pipe, q_transcribe, max_buffer_time, max_buffer_size, filename):
    global running, in_dev
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
            now = datetime.now(tz=timezone(timedelta(hours=+9))).strftime("%Y-%m-%d_%H%M%S")
            stripped_content = buffer_content.strip()
            if stripped_content not in ["you", "I"]:  # youとIのみ認識した場合には除外
                q_transcribe.put(stripped_content)
                with open(filename, "a") as file:
                    file.write('User: ' + text + "\n")
                
                # 録音データをファイルに出力　※デバッグ用
                if in_dev:
                    file_path = create_chat_file(folder_path="./uploads/sample", file_name=f"transcribed_{now}.wav")
                    write(file_path, 16000, audio_data)
            else:
                print(f"you or I: {buffer_content}")
                # 録音データをファイルに出力　※デバッグ用
                if in_dev:
                    file_path = create_chat_file(folder_path="./uploads/sample", file_name=f"you_{now}.wav")
                    write(file_path, 16000, audio_data)
            
            buffer_content = ""
            last_output_time = current_time
        del audio_data  # 処理後にメモリから削除

def assistant_transcription(response_buffer, filename, q_assistant):
    global running
    while running:
        try:
            text = response_buffer.get(timeout=1)  # 1秒間待ってからキューから取得
        except queue.Empty:
            continue

        try:
            # ここでAssistantの返答を作成する
            # chatbot = ask_llm(text, image_path='./temp/temp.jpg')
            chatbot = ask_llm(text, image_path=None)
            # print(chatbot)
            # print("=========================================")
            assistant_text = chatbot[-1][1]['text']
            q_assistant.put(assistant_text)
            with open(filename, "a") as file:
                file.write('Assistant: ' + assistant_text + "\n")
        except Exception as e:
            print(f"Error in assistant_transcription: {e}")

def create_pipe_for_speech_recognition():
    # モデルとプロセッサの設定
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    # model_id = "distil-whisper/distil-small.en"
    model_id = "distil-whisper/distil-large-v2"

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
    global running, in_dev
    in_dev = True
    pipe = create_pipe_for_speech_recognition()

    # デバイスの指定
    device_index = None  # 適切なデバイスインデックスを設定するか、Noneのままにしてデフォルトを使用

    # 録音時間の設定
    max_record_duration = 10  # 最大10秒
    silence_duration = 8   # 無音と判断する期間（秒）

    # 録音、文字起こし、Assistant出力のキューの設定
    q_record = queue.Queue(maxsize=10)  # キューのサイズ制限を設定
    q_transcribe = queue.Queue(maxsize=10)  # キューのサイズ制限を設定
    q_assistant = queue.Queue(maxsize=10)  # キューのサイズ制限を設定

    # バッファの設定
    max_buffer_time = 3  # バッファの内容を出力する時間間隔（秒）
    max_buffer_size = 300  # バッファの最大文字数

    # 並行処理数の最適化
    num_transcription_threads = 10  # 文字起こしスレッドの数を増やす

    # ファイル名の設定
    filename = create_chat_file()

    # 録音スレッドの開始
    record_thread = threading.Thread(target=record_audio, args=(q_record, max_record_duration, silence_duration, device_index))
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

    # Assistant出力スレッドの開始
    assistant_thread = threading.Thread(target=assistant_transcription, args=(q_transcribe, filename, q_assistant))
    assistant_thread.start()

    try:
        while True:
            time.sleep(0.1)
            try:
                assistant_text = q_assistant.get(timeout=1)  # 1秒間待ってからキューから取得
                speech_to_text(assistant_text)
            except queue.Empty:
                continue
    except KeyboardInterrupt:
        running = False
        record_thread.join()
        for t in transcription_threads:
            t.join()
        q_transcribe.join()
        print("\nRecording and transcription stopped.")

if __name__ == "__main__":
    main()