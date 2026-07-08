import os
import subprocess
from screenshot import take_screenshot

shot = take_screenshot()
if shot is None:
    print("Скрин отменён")
    exit()

try:
    text = subprocess.check_output(
        ["tesseract", shot, "stdout", "-l", "eng+jpn+rus"],
        text=True
    )

    print(text)

    subprocess.run(
        ["wl-copy"],
        input=text,
        text=True
    )

finally:
    if shot and os.path.exists(shot):
        os.remove(shot)