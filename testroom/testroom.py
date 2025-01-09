import time
import pyautogui
import pygetwindow as gw


def main():
    key_to_press = 'esc'
    press_interval = 1
    press_duration = 10
    prep_time = 5

    print("プログラムは5秒の準備時間を開始します。")
    time.sleep(prep_time)
    
    print(f"キー「{key_to_press}」の連打を{press_duration}秒間開始します。")
    start_time = time.time()
    end_time = start_time + press_duration

    try:
        while time.time() < end_time:
            pyautogui.keyDown(key_to_press)
            time.sleep(press_interval)
            pyautogui.keyUp(key_to_press)
    except KeyboardInterrupt:
        print("ユーザーによってプログラムが中断されました。")

    print("キーの連打を終了しました。")

if __name__ == "__main__":
    main()
