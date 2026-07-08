import re
import subprocess


def fixJapaneseSpaces(text):
    text = re.sub(
        r"(?<=[\u3040-\u30ff\u3400-\u9fff])\s+(?=[\u3040-\u30ff\u3400-\u9fff])",
        "",
        text,
    )
    return text


def getText(path):
    text = subprocess.check_output(
        ["tesseract", path, "stdout", "-l", "eng+jpn+rus"], text=True
    )

    text = fixJapaneseSpaces(text)

    return text
