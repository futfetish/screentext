import os
import subprocess
from screenshot import take_screenshot
from getText import getText

shot = take_screenshot()
if shot is None:
    print("Скрин отменён")
    exit()

try:
    text = getText(shot)

    print(text)

    subprocess.run(["wl-copy"], input=text, text=True)

finally:
    if shot and os.path.exists(shot):
        os.remove(shot)
