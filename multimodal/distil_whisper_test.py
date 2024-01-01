import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import sounddevice as sd
import numpy as np
import sys
import threading
import queue
import time
from datetime import datetime
import os

# グローバル変数で実行状態を管理
running = True

# 利用可能なオーディオデバイスのリスト表示
print("利用可能なオーディオデバイス:")
print(sd.query_devices())

def is_silent(data, threshold=0.01):
    """無音を検出する。"""
    return np.max(np.abs(data)) < threshold

def contains_voice(data, threshold=0.02):
    """音声が含まれているかをチェックする。"""
    return np.max(np.abs(data)) > threshold

def record_audio(q, max_record_duration, silence_duration, device_index):
    global running
    with sd.InputStream(samplerate=16000, channels=1, callback=None, dtype='float32', device=device_index):
        while running:
            audio_data = sd.rec(int(max_record_duration * 16000), samplerate=16000, channels=1)
            sd.wait()  # 録音が完了するまで待機
            audio_data = np.squeeze(audio_data)

            # 録音データに音声が含まれていない場合はスキップ
            if not contains_voice(audio_data):
                continue

            # 無音を検出して録音を早期終了
            for i in range(0, len(audio_data), int(16000 * silence_duration)):
                window = audio_data[i:i + int(16000 * silence_duration)]
                if is_silent(window):
                    audio_data = audio_data[:i]
                    break

            q.put(audio_data)  # 録音データをキューに追加
            del audio_data  # 処理後にメモリから削除

def transcribe_audio(q, pipe, output_buffer, max_buffer_time, max_buffer_size):
    global running
    buffer_content = ""
    last_output_time = time.time()

    while running:
        try:
            audio_data = q.get(timeout=1)  # 1秒間待ってからキューから取得
        except queue.Empty:
            continue

        result = pipe(audio_data)
        text = result["text"]
        buffer_content += text + " "
        current_time = time.time()

        # バッファの内容を出力するタイミングを判断
        if (current_time - last_output_time > max_buffer_time) or (len(buffer_content) >= max_buffer_size):
            output_buffer.put(buffer_content.strip())
            buffer_content = ""
            last_output_time = current_time
        del audio_data  # 処理後にメモリから削除

def output_transcription(output_buffer, filename):
    global running
    while running:
        try:
            text = output_buffer.get(timeout=1)  # 1秒間待ってからキューから取得
        except queue.Empty:
            continue

        with open(filename, "a") as file:
            file.write(text + "\n")

def main():
    global running
    # モデルとプロセッサの設定
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "distil-whisper/distil-small.en"

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

    # デバイスの指定
    device_index = None  # 適切なデバイスインデックスを設定するか、Noneのままにしてデフォルトを使用

    # 録音時間の設定
    max_record_duration = 3  # 最大3秒
    silence_duration = 0.7    # 無音と判断する期間（秒）

    # 録音と文字起こしのキュー
    q = queue.Queue(maxsize=10)  # キューのサイズ制限を設定
    output_buffer = queue.Queue(maxsize=10)  # キューのサイズ制限を設定

    # バッファの設定
    max_buffer_time = 3  # バッファの内容を出力する時間間隔（秒）
    max_buffer_size = 300  # バッファの最大文字数

    # 並行処理数の最適化
    num_transcription_threads = 10  # 文字起こしスレッドの数を増やす

    # 現在のディレクトリのパスを取得
    current_directory = os.getcwd()

    # 'uploads'フォルダのパスを生成
    uploads_directory = os.path.join(current_directory, 'uploads')

    # 'uploads'フォルダが存在しない場合は作成
    if not os.path.exists(uploads_directory):
        os.makedirs(uploads_directory)

    # テキストファイルの名前を生成（'uploads'フォルダ内）
    filename = os.path.join(uploads_directory, datetime.now().strftime("%Y-%m-%d_%H%M.txt"))

    # 録音スレッドの開始
    # ここでは、新しいスレッドを作成しています。このスレッドは、音声の録音を担当します。
    # 'target'パラメータには、スレッドが実行する関数（この場合は'record_audio'関数）を指定します。
    # 'args'パラメータには、'target'で指定した関数に渡す引数をタプル形式で指定します。
    # この例では、'q'（録音データを格納するキュー）、'max_record_duration'（最大録音時間）、
    # 'silence_duration'（無音と判断する期間）、'device_index'（録音デバイスのインデックス）を引数として渡しています。
    record_thread = threading.Thread(target=record_audio, args=(q, max_record_duration, silence_duration, device_index))
    record_thread.start()

    # 文字起こしスレッドの開始
    # 文字起こしスレッドの初期化
    transcription_threads = []
    # 指定された数のスレッドを作成
    for _ in range(num_transcription_threads):
        # 各スレッドは、音声データを文字に変換する役割を持つ
        # 'target'パラメータには、スレッドが実行する関数（この場合は'transcribe_audio'関数）を指定します。
        # 'args'パラメータには、'target'で指定した関数に渡す引数をタプル形式で指定します。
        # この例では、'q'（録音データを格納するキュー）、'pipe'（音声データを処理するパイプライン）、
        # 'output_buffer'（文字起こし結果を格納するバッファ）、'max_buffer_time'（バッファの内容を出力する時間間隔）、
        # 'max_buffer_size'（バッファの最大文字数）を引数として渡しています。
        thread = threading.Thread(target=transcribe_audio, args=(q, pipe, output_buffer, max_buffer_time, max_buffer_size))
        # スレッドの開始
        thread.start()
        # スレッドをリストに追加
        transcription_threads.append(thread)

    # 出力スレッドの開始
    # 出力スレッドの初期化と開始
    # 'target'パラメータには、スレッドが実行する関数（この場合は'output_transcription'関数）を指定します。
    # 'args'パラメータには、'target'で指定した関数に渡す引数をタプル形式で指定します。
    # この例では、'output_buffer'（文字起こし結果を格納するバッファ）と'filename'（テキストファイルの名前）を引数として渡しています。
    output_thread = threading.Thread(target=output_transcription, args=(output_buffer, filename))
    output_thread.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        running = False
        record_thread.join()
        for t in transcription_threads:
            t.join()
        output_thread.join()
        print("\nRecording and transcription stopped.")

if __name__ == "__main__":
    main()


