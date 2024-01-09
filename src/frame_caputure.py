import os
import time
from datetime import datetime

import cv2
import config


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


def save_frame(frame, filename, directory=config.FRAMES_DIR):
    # ディレクトリが存在しない場合は作成
    if not os.path.exists(directory):
        os.makedirs(directory)
    # ファイル名のパスを作成
    filepath = os.path.join(directory, filename)
    # フレームを保存
    cv2.imwrite(filepath, frame)


def save_temp_frame(frame, filename, directory=config.FRAME_TEMP_DIR):
    # ディレクトリが存在しない場合は作成
    if not os.path.exists(directory):
        os.makedirs(directory)
    # ファイル名のパスを作成
    filepath = os.path.join(directory, filename)
    # フレームを保存
    cv2.imwrite(filepath, frame)
    return filepath  # 保存したファイルのパスを返す


def capture_image():
    try:
        video = cv2.VideoCapture(0)
        if not video.isOpened():
            raise IOError("カメラを開くことができませんでした。")
        # 0.2秒 sleep を入れる
        time.sleep(0.2)

        success, frame = video.read()  # カメラからフレームを読み込む
        if not success:
            print("フレームの読み込みに失敗しました。")

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 現在のタイムスタンプを取得

        # imageファイルのパス取得
        temp_image_path = save_temp_frame(frame, config.FRAME_TEMP_FILE_NAME)  # 一時ファイルとしてフレームを保存
    except Exception as e:
        raise e
    finally:
        video.release()
        cv2.destroyAllWindows()

    return frame, temp_image_path, timestamp


def save_image(frame, timestamp, generated_text):
    # 生成されたテキストをフレームに追加
    text_to_add = f"{timestamp}: {generated_text}"
    add_text_to_frame(frame, text_to_add)  # フレームにテキストを追加

    # フレームを保存
    filename = f"{timestamp}.jpg"
    save_frame(frame, filename)  # 画像として保存


if __name__ == "__main__":
    # imageを取得
    frame, temp_image_path, timestamp = capture_image()
    generated_text = "Hello World!" # TODO: replace with gemini call, then iput text and image.
    save_image(frame, timestamp, generated_text)

    print("フレームを保存しました。")
    print("Done!")