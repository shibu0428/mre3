import time
import pyautogui
import pydirectinput
import sys

def main():
    start=time.time()
    pydirectinput.click()
    end=time.time()
    print(end-start)

if __name__ == "__main__":
    main()
