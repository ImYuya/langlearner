import os
import cv2

from collections import deque
from datetime import datetime
import PIL.Image
import google.generativeai as genai
from google.generativeai.types.generation_types import BlockedPromptException
from dotenv import load_dotenv

import sys
sys.path.append('src')
import config

# .envファイルの内容を読み込見込む
load_dotenv(override=True)
genai.configure(api_key=config.GOOGLE_API_KEY)

def wrap_text(text, line_length):
    """テキストを指定された長さで改行する"""
    words = text.split(' ')
    lines = []
    current_line = ''

    for word in words:
        if len(current_line) + len(word) + 1 > line_length:
            lines.append(current_line)
            current_line = word
        else:
            current_line += ' ' + word

    lines.append(current_line)  # 最後の行を追加
    return lines


def add_text_to_frame(frame, text):
    # テキストを70文字ごとに改行
    wrapped_text = wrap_text(text, 70)

    # テキストのフォントとサイズ
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0  # フォントサイズを大きくする
    color = (255, 255, 255)  # 白色
    outline_color = (0, 0, 0)  # 輪郭の色（黒）
    thickness = 2
    outline_thickness = 4  # 輪郭の太さ
    line_type = cv2.LINE_AA

    # 各行のテキストを画像に追加
    for i, line in enumerate(wrapped_text):
        position = (10, 30 + i * 30)  # 各行の位置を調整（より大きい間隔）

        # テキストの輪郭を描画
        cv2.putText(frame, line, position, font, font_scale, outline_color, outline_thickness, line_type)

        # テキストを描画
        cv2.putText(frame, line, position, font, font_scale, color, thickness, line_type)


def save_frame(frame, filename, directory='./frames'):
    # ディレクトリが存在しない場合は作成
    if not os.path.exists(directory):
        os.makedirs(directory)
    # ファイル名のパスを作成
    filepath = os.path.join(directory, filename)
    # フレームを保存
    cv2.imwrite(filepath, frame)


def save_temp_frame(frame, filename, directory='./temp'):
    # ディレクトリが存在しない場合は作成
    if not os.path.exists(directory):
        os.makedirs(directory)
    # ファイル名のパスを作成
    filepath = os.path.join(directory, filename)
    # フレームを保存
    cv2.imwrite(filepath, frame)
    return filepath  # 保存したファイルのパスを返す


def send_frame_with_text_to_gemini(frame, previous_texts, timestamp, user_input, client):
    temp_file_path = save_temp_frame(frame, "temp.jpg")
    img = PIL.Image.open(temp_file_path)

    # 過去のテキストをコンテキストとして結合
    context = ' '.join(previous_texts)

    # システムメッセージの追加
    system_message = "System Message - Your identity: Gemini, you are a smart, kind, and helpful AI assistant."

    # Geminiモデルの初期化
    model = client.GenerativeModel('gemini-pro-vision')

    # モデルに画像とテキストの指示を送信
    prompt = f"{system_message}\nGiven the context: {context} and the current time: {timestamp}, please respond to the following message without repeating the context in Japanese, using no more than 20 words. Message: {user_input}"
    
    try:
        print(f"{prompt=}, {img=}")
        response = model.generate_content([prompt, img], stream=True)
        print(f"{response=}")
        response.resolve()
        # 生成されたテキストを返す
        return response.text
    
    except BlockedPromptException as e:
        print("AI response was blocked due to safety concerns. Please try a different input.")
        return "AI response was blocked due to safety concerns."
    
    # return "Hello, I am Gemini. I am a smart, kind, and helpful AI assistant."


def main():
    try:
        video = cv2.VideoCapture(0)
        if not video.isOpened():
            raise IOError("カメラを開くことができませんでした。")

        previous_texts = deque(maxlen=5)

        loop_count = 0
        while True:  # 無限ループに変更
            # ウェイトを追加
            user_input = f"""How are you?"""
            print(f"{user_input=}")

            # 画像処理とAI応答のコード
            success, frame = video.read()  # カメラからフレームを読み込む
            if not success:
                print("フレームの読み込みに失敗しました。")
                break  # フレームの読み込みに失敗した場合、ループを抜ける

            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 現在のタイムスタンプを取得

            # Gemini AIモデルにフレームとユーザーの入力を送信し、応答を生成
            generated_text = send_frame_with_text_to_gemini(frame, previous_texts, timestamp, user_input, genai)
            print(f"Timestamp: {timestamp}, Generated Text: {generated_text}")

            # 過去のテキストを更新
            # previous_texts.append(f"[{timestamp}] Message: {user_input}, Generated Text: {generated_text}")
            previous_texts.append(f"Timestamp: {timestamp}\nUser Message: {user_input}\nYour Response: {generated_text}\n")

            # 生成されたテキストをフレームに追加
            text_to_add = f"{timestamp}: {generated_text}"
            add_text_to_frame(frame, text_to_add)  # フレームにテキストを追加

            # フレームを保存
            filename = f"{timestamp}.jpg"
            save_frame(frame, filename)  # 画像として保存

            # AIの応答を音声に変換して再生
            print(f"{text_to_add=}")

            loop_count += 1


    except IOError as e:
        raise e
                
    finally:
        video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()