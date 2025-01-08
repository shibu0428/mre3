import time
import pyautogui
import pydirectinput
import sys

def main():
    key_to_press = 't'         # 連打したいキーに変更（例: 'w', 'a', 's', 'd', 'space'）
    press_interval = 0.2       # キーを連打する間隔（秒）
    press_duration = 10        # キーを連打する秒数
    prep_time = 5              # 準備時間（秒）

    print(f"プログラムは{prep_time}秒の準備時間を開始します。")
    print("準備ができたら、ゲームウィンドウをアクティブにしてください。")
    time.sleep(prep_time)  # 準備時間

    print(f"キー「{key_to_press}」の連打を{press_duration}秒間開始します。")
    start_time = time.time()
    end_time = start_time + press_duration

    try:
        while time.time() < end_time:
            pydirectinput.click()
            #pydirectinput.press(key_to_press)
            time.sleep(press_interval)
    except KeyboardInterrupt:
        print("ユーザーによってプログラムが中断されました。")
        sys.exit()

    print("キーの連打を終了しました。")

if __name__ == "__main__":
    main()
